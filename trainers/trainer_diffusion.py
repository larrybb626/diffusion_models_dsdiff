import os
import time
from functools import partial
from inspect import isfunction

import SimpleITK as sitk
import lightning.pytorch as pl
import numpy as np
import torch
from monai.data import Dataset, CacheDataset, decollate_batch, DataLoader, pad_list_data_collate
from monai.metrics import MAEMetric, SSIMMetric
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR

from training_project.training_transform import *
from training_project.utils.image_pool import ImagePool
from training_project.utils.progress_bar import printProgressBar
from training_project.utils.save_tensor_img import tensor2im
# from ldm_v1.modules.diffusionmodules.openaimodel import UNetModel as UNet_Latent
from training_project.utils.util import init_weights


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def get_duration_time_str(s_time, e_time):
    h, remainder = divmod((e_time - s_time), 3600)  # 小时和余数
    m, s = divmod(remainder, 60)  # 分钟和秒
    time_str = "%02d h:%02d m:%02d s" % (h, m, s)
    return time_str


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


class DiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        # =============================dataset===================================
        self.val_ds = None
        self.train_ds = None
        self.test_ds = None
        self.num_workers = config.num_workers
        self.train_batch_size = config.train_batch_size
        self.val_batch_size = config.val_batch_size
        self.test_batch_size = 16
        self.num_samples = config.num_samples
        self.val_num_samples = config.val_num_samples
        self.dataset_type = config.dataset_type
        self.val_transforms = None
        self.train_transforms = None
        self.test_transforms = None
        self.crop_size = config.crop_size
        # =============================fold======================================
        self.fold_K = config.fold_K
        self.fold_idx = config.fold_idx
        # ================================net====================================
        input_channels = len(config.train_keys)
        output_channel = 1
        self.net_G = UNet(in_channel=input_channels + output_channel, out_channel=1,
                          inner_channel=64,
                          channel_mults=[
                              1,
                              2,
                              4,
                              8
                          ],
                          attn_res=[16],
                          num_head_channels=32,
                          res_blocks=2,
                          dropout=0.2,
                          image_size=320
                          )
        # self.net_G = UNet_Latent(image_size=0,
        #                          in_channels=input_channels + output_channel,
        #                          out_channels=output_channel,
        #                          model_channels=320,
        #                          attention_resolutions=[4,2,1],
        #                          num_res_blocks=2,
        #                          channel_mult=[1,2,4,4],
        #                          num_head_channels=32,
        #                          )
        init_weights(self.net_G)
        # ================================loss&metric============================
        self.criterion_dict = None
        self.loss_weight_dict = config.loss_weight_dict
        self.mae_metric = MAEMetric(reduction="mean", get_not_nans=False)
        self.ssim_metric = SSIMMetric(spatial_dims=2)

        self.best_val_mae = 1000
        self.best_val_ssim = 0
        self.best_val_epoch = 0
        self.loss_fn = nn.MSELoss()
        # =================================optimiser======================================
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # ====================================lr===========================================
        self.max_lr = config.lr
        self.min_lr = config.lr_low
        self.lr_list = {"lr": []}
        # =============================training setting=====================================
        self.random_state = config.seed
        self.random_prob = config.augmentation_prob
        self.max_epochs = config.num_epochs
        self.warmup_epochs = config.lr_warm_epoch
        self.cos_epochs = config.lr_cos_epoch
        # self.metric_values = {"MAE": []}
        self.epoch_loss_values = []
        # self.training_loss = {"loss": [], "MSEloss": [], "ms_ssim_loss": [], "monai_ssmi_loss": []}
        # self.training_ms_ssim = {"ms_ssim_loss": []}
        self.fake_pool = ImagePool(config.pool_size)
        # =============================training variable====================================
        self.train_s_time = 0
        self.train_e_time = 0
        self.val_s_time = 0
        self.val_e_time = 0
        self.predict_tic = None
        self.predict_toc = None
        self.keys = config.train_keys
        self.beta_schedule = {
            "train": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 0.01
            },
            "test": {
                "schedule": "linear",
                "n_timestep": 1000,
                "linear_start": 1e-4,
                "linear_end": 0.09
            }
        }
        # =============================文件地址=======================================
        self.data_dir = config.h5_2d_img_dir
        self.train_dir = os.path.join(self.data_dir, "images_tr")
        self.test_dir = os.path.join(self.data_dir, "images_ts")
        self.template_dir = config.filepath_img
        self.log_pic_dir = os.path.join(config.root_dir, "loss_pic")
        self.result_dir = config.root_dir
        self.record_file = os.path.join(config.root_dir, "log_txt.txt")
        self.pred_result_dir = os.path.join(self.result_dir, "pred_nii")
        # if not os.path.exists(self.pred_result_dir):
        #     os.makedirs(self.pred_result_dir)
        if not os.path.exists(self.log_pic_dir):
            os.makedirs(self.log_pic_dir)
        # 初始化 training output saving
        # self.training_step_output = []
        self.validation_step_outputs = []
        # 搞搞时间步
        self.prepare_noise_schedule(phase="train")

    def print_to_txt(self, *args):
        print(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()

    def do_split(self, K, fold):
        """
        :reg_param K: 分几折
        :reg_param fold: 第几折，从1开始
        :return:分折的病人id列表[train,val]
        """
        fold_train = []
        fold_test = []

        kf = KFold(n_splits=K, random_state=self.random_state, shuffle=True)
        id_list = sorted(os.listdir(self.train_dir))
        for train_index, test_index in kf.split(id_list):
            fold_train.append(np.array(id_list)[train_index])
            fold_test.append(np.array(id_list)[test_index])

        train_id = fold_train[fold - 1]
        test_id = fold_test[fold - 1]
        self.print_to_txt(f'train_id:{len(train_id)}||valid_id:{len(test_id)}')
        return [train_id, test_id]

    def get_data_dict(self, id_list):
        # 输入id的list获取数据字典
        data_dict = []
        for id_num in id_list:
            layer_list = sorted(os.listdir(os.path.join(self.train_dir, id_num)))
            # layer_list = [layer_list] * 4
            for layer in layer_list:  # 头尾不要?
                new_data_dict = {"path": os.path.join(self.train_dir, id_num, layer)}
                data_dict.append(new_data_dict)
        return data_dict

    def get_test_data_dict(self, ):
        # 输入test_id的list获取数据字典
        data_dict = []
        id_list = sorted(os.listdir(self.test_dir))
        for id_num in id_list:
            layer_list = sorted(os.listdir(os.path.join(self.test_dir, id_num)))
            for layer in layer_list:
                new_data_dict = {"path": os.path.join(self.test_dir, id_num, layer)}
                data_dict.append(new_data_dict)
        return data_dict

    def get_dataset(self, data_list, transform, mode="train", dataset_type="normal"):
        """
        :param data_list:
        :param transform:
        :param mode: "train" or "val"
        :param dataset_type: "normal" or "cache"
        :return:
        """
        if mode == "train":
            if dataset_type == "normal":
                self.train_ds = Dataset(
                    data=data_list,
                    transform=transform,
                )
            elif dataset_type == "cache":
                self.train_ds = CacheDataset(
                    data=data_list,
                    transform=transform,
                    # cache_num=300,
                    cache_rate=0.6,
                    num_workers=self.num_workers,
                )
        elif mode == "val":
            if dataset_type == "normal":
                self.val_ds = Dataset(
                    data=data_list,
                    transform=transform
                )
            elif dataset_type == "cache":
                self.val_ds = CacheDataset(
                    data=data_list,
                    transform=transform,
                    # cache_num=100,
                    cache_rate=1,
                    num_workers=self.num_workers,
                )
        elif mode == "test":
            if dataset_type == "normal":
                self.test_ds = Dataset(
                    data=data_list,
                    transform=transform
                )
            elif dataset_type == "cache":
                self.test_ds = CacheDataset(
                    data=data_list,
                    transform=transform,
                    # cache_num=100,
                    cache_rate=1,
                    num_workers=self.num_workers,
                )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_mae"] = self.best_val_mae
        checkpoint["best_metric"] = self.best_val_ssim
        checkpoint["best_val_epoch"] = self.best_val_epoch
        checkpoint["criterion_dict"] = self.criterion_dict

    def on_load_checkpoint(self, checkpoint):
        self.best_val_mae = checkpoint["best_mae"]
        self.best_val_ssim = checkpoint["best_metric"]
        self.best_val_epoch = checkpoint["best_val_epoch"]
        self.criterion_dict = checkpoint["criterion_dict"]

    def prepare_noise_schedule(self, phase):
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
                extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
                extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def prepare_data(self):
        # prepare data
        # 根据dir 获取train：0 val：1
        print("preparing data with val")
        # datasets = sorted(os.listdir(self.train_dir))
        datasets = self.do_split(self.fold_K, self.fold_idx)
        train_dict = self.get_data_dict(datasets[0])
        val_dict = self.get_data_dict(datasets[1])
        test_dict = self.get_test_data_dict()
        self.train_transforms = get_2d_train_transform_diff(keys=self.keys, random_prob=self.random_prob)
        self.val_transforms = get_2d_val_transform_diff(keys=self.keys)
        self.test_transforms = get_2d_test_transform(keys=self.keys)
        # 获取dataset 方法内直接赋值self.train_ds, self.val_ds, self.test_ds
        self.get_dataset(train_dict, self.train_transforms, mode="train", dataset_type=self.dataset_type)
        self.get_dataset(val_dict, self.val_transforms, mode="val", dataset_type=self.dataset_type)
        self.get_dataset(test_dict, self.test_transforms, mode="test", dataset_type=self.dataset_type)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return val_loader

    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.test_ds,
            batch_size=4,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return pred_loader

    # def configure_losses(self):
    #     self.criterion_dict = {}
    #     for loss_name, weight in self.loss_weight_dict.items():
    #         # self.training_loss.update({loss_name: []})
    #         self.criterion_dict.update({loss_name: loss_picker(loss_name)})
    #     self.print_to_txt("loss&weight |", self.loss_weight_dict)

    # 方法名都是固定的
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self._model.parameters(), betas=(self.beta1, self.beta2), lr=self.max_lr,
        #                               weight_decay=1e-5)
        optimizer = torch.optim.Adam(self.net_G.parameters(), betas=(self.beta1, self.beta2), lr=self.max_lr)
        return ({"optimizer": optimizer,
                 "lr_scheduler": {"scheduler": CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr),
                                  "interval": "epoch"}},)

    def on_train_start(self):
        self.print_to_txt("||start with||\n", self.config)
        # self.configure_losses()

    def on_train_epoch_start(self):
        self.print_to_txt("⭐epoch: {}⭐".format(self.current_epoch))
        # 起始时间
        self.train_s_time = time.time()

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=y_0.device).long()
        # gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        # sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        # sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        gamma_t1 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = gamma_t1.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.net_G(torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), sample_gammas)
            loss = self.loss_fn(mask * noise, mask * noise_hat)
        else:
            noise_hat = self.net_G(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, is_ddim=False, ddim_step=50):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)
        yt_shape = list(y_cond.shape)
        yt_shape[1] = 1
        y_t = default(y_t, lambda: torch.randn(yt_shape, device=self.device))
        ret_arr = y_t
        if is_ddim:
            y_t, ret_arr = self.p_sample_ddim(model=self.net_G, y_t=y_t, y_cond=y_cond, batch_size=b,
                                              ddim_timesteps=ddim_step,
                                              ddim_discr_method="uniform", ddim_eta=1, clip_denoised=True,
                                              sample_inter=sample_inter)
        else:
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
                y_t = self.p_sample(y_t, t, y_cond=y_cond)
                if mask is not None:
                    y_t = y_0 * (1. - mask) + mask * y_t
                if i % sample_inter == 0:
                    ret_arr = torch.cat([ret_arr, y_t], dim=2)
                print_content = "{}/{} validation processing......".format(i + 1, self.num_timesteps)
                printProgressBar(i, self.num_timesteps - 1, content=print_content)
        return y_t, ret_arr

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
                extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # gamma_t1 = extract(self.gammas, t, x_shape=(1, 1))
        # sqrt_gamma_t2 = extract(self.gammas, t+1, x_shape=(1, 1))
        # noise_level = (sqrt_gamma_t2 - gamma_t1) * torch.rand((y_t.shape[0], 1), device=y_t.device) + gamma_t1
        # noise_level = noise_level.view(y_t.shape[0], -1).to(y_t.device)
        # a = time.time()
        y_0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=self.net_G(torch.cat([y_cond, y_t], dim=1), noise_level))
        # b = time.time()
        # print(b-a)
        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
                sample_gammas.sqrt() * y_0 +
                (1 - sample_gammas).sqrt() * noise
        )

    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def p_sample_ddim(self, model, y_t, y_cond, batch_size=8, ddim_timesteps=50, ddim_discr_method="uniform",
                      ddim_eta=0.0, clip_denoised=True, sample_inter=1):
        # make ddim timestep sequence
        assert sample_inter >= 1, 'num_timesteps must greater than sample_num'
        if ddim_discr_method == 'uniform':
            c = self.num_timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.num_timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.num_timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        sample_img = y_t
        ret_arr = y_t
        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.gammas, t, sample_img.shape)
            alpha_cumprod_t_prev = extract(self.gammas, prev_t, sample_img.shape)

            # 2. predict noise using model
            pred_noise = model(torch.cat([y_cond, sample_img], dim=1), alpha_cumprod_t)

            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, sample_img], dim=2)
            print_content = "{}/{} validation processing......".format(i + 1, ddim_timesteps)
            printProgressBar(i, ddim_timesteps - 1, content=print_content)
        return sample_img, ret_arr

    def predict_forward(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, is_ddim=True, ddim_step=50):
        return self.restoration(y_cond, y_t, y_0, mask, sample_num, is_ddim=is_ddim, ddim_step=ddim_step)[0]

    def training_step(self, batch, batch_idx):
        real_A, real_B = (batch["image"], batch["t1ce"])
        loss = self.forward(y_0=real_B, y_cond=real_A)
        loss_dict = {"loss": loss, }
        return loss_dict

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        # todo: need to change while loss change
        self.log_dict(outputs)
        self.train_e_time = time.time()
        time_str = get_duration_time_str(s_time=self.train_s_time, e_time=self.train_e_time)
        loss_print_content = {}
        for loss_n, loss_v in outputs.items():
            loss_print_content.update({loss_n: "%.4f" % loss_v.item()})
        print_content = "{} / {} {}  || Training cost: {}".format(batch_idx + 1,
                                                                  len(self.train_dataloader()),
                                                                  loss_print_content,
                                                                  time_str)
        printProgressBar(batch_idx, len(self.train_dataloader()) - 1, content=print_content)

    def on_train_epoch_end(self):
        # epoch lr schedule
        # sch1, sch2 = self.lr_schedulers()
        # if self.current_epoch > self.max_epochs - self.cos_epochs:
        #     sch1.step()
        #     sch2.step()

        self.train_e_time = time.time()
        time_str = get_duration_time_str(s_time=self.train_s_time, e_time=self.train_e_time)
        lr = self.optimizers().optimizer.param_groups[0]['lr']
        self.print_to_txt(
            "Epoch Done || lr: {} || Epoch cost: {}".format(lr, time_str))
        self.log("lr", lr)

    # def on_validation_epoch_start(self):
    #     print('?????????????????????????')
    def validation_step(self, batch, batch_idx):
        images, labels = (batch["image"], batch["t1ce"])

        output, visual_img = self.restoration(images, is_ddim=True)

        if "SSIM_loss" in self.loss_weight_dict.keys():
            loss = self.criterion_dict["SSIM_loss"](output, labels)
        elif "Perceptual_loss" in self.loss_weight_dict.keys():
            loss = self.criterion_dict["Perceptual_loss"].forward(output, labels).mean()
        # 每15个step保存图片
        # save_image_2d(images, output, labels, self.result_dir, batch_idx, every_n_step=50, mode="Val")
        if batch_idx % 1 == 0:
            for b in range(images.shape[0]):
                _labels = labels[b:b + 1, ...]
                _output = output[b:b + 1, ...]
                _visual_img = visual_img[b:b + 1, ...]
                img_realB = tensor2im(_labels)
                img_fakeB = tensor2im(_output)
                img_restore = tensor2im(_visual_img)
                self.logger.experiment.add_image("val_real_B", img_realB, self.global_step + b, dataformats="HWC")
                self.logger.experiment.add_image("val_fake_B", img_fakeB, self.global_step + b, dataformats="HWC")
                self.logger.experiment.add_image("restore_procedure", img_restore, self.global_step + b,
                                                 dataformats="HWC")

            # self.log_dict({"val_real_B": img_realB,
            #                "val_fake_B": img_fakeB})
        # data_range = torch.max(labels) - torch.min(labels)
        outputs = [i for i in decollate_batch(output)]
        labels = [i for i in decollate_batch(labels)]
        self.mae_metric(y_pred=outputs, y=labels)
        self.ssim_metric(y_pred=outputs, y=labels)
        # outputs = [(i - torch.mean(i)) / (torch.std(i) / 400) + 2048 for i in outputs]
        # labels = [(i - torch.mean(i)) / (torch.std(i) / 400) + 2048 for i in labels]
        # self.ssim_metric = SSIMMetric(spatial_dims=2, data_range=data_range)

        self.validation_step_outputs.append({"val_loss": loss, "val_number": len(outputs)})

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0):
        printProgressBar(batch_idx, len(self.val_dataloader()) - 1,
                         content="{}/{} validation processing......".format(batch_idx + 1, len(self.val_dataloader())))

    def on_validation_epoch_end(self):
        val_loss, num_items, val_ssim = 0, 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_ssim = self.ssim_metric.aggregate().item()
        mean_val_mae = self.mae_metric.aggregate().item()
        self.mae_metric.reset()
        self.ssim_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)

        if mean_val_ssim > self.best_val_ssim:
            self.best_val_ssim = mean_val_ssim
            self.best_val_epoch = self.current_epoch
        if mean_val_mae < self.best_val_mae:
            self.best_val_mae = mean_val_mae
        self.print_to_txt(
            f"val_loss: {mean_val_loss:.4f}"
            f" || current mean SSIM: {mean_val_ssim:.4f}"
            f" || best mean SSIM: {self.best_val_ssim:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.print_to_txt(
            f"current mean MAE: {mean_val_mae:.4f}"
            f" || best mean MAE: {self.best_val_mae:.4f} "
        )
        # self.metric_values["MAE"].append(mean_val_mae)
        self.log_dict({
            "val_ssim": mean_val_ssim,
            "val_mae": mean_val_mae,
            "val_loss": mean_val_loss,
        })
        # self.log("best",self.best_val_mae)
        # print_logger(self.metric_values, self.log_pic_dir)
        self.validation_step_outputs.clear()

    def on_predict_start(self):
        self.predict_tic = time.time()
        file_list = sorted(os.listdir(self.test_dir))
        self.pred_dict = dict(zip(file_list, [{} for i in range(len(file_list))]))
        # self.prepare_noise_schedule(phase="test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        path, images, labels = (batch["path"], batch["image"], batch["t1ce"])
        id_num = [p.split("/")[-2] for p in path]
        slice_idx = [int(os.path.basename(p).split(".")[0].split("_")[-1]) for p in path]
        roi_x = int(np.ceil(images.shape[2] / 32) * 32)
        roi_y = int(np.ceil(images.shape[3] / 32) * 32)
        roi_size = (roi_x, roi_y)
        sw_batch_size = self.test_batch_size
        # outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.predict_forward(is_ddim=True,ddim_step=200), overlap=0.25,
        #                                    mode='gaussian')
        outputs = self.predict_forward(images, is_ddim=True, ddim_step=50)

        return id_num, outputs, slice_idx

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # 临时文件-2d
        for id_, slice_i, img in zip(outputs[0], outputs[2], outputs[1]):
            output_img = img[0, :, :].cpu().numpy()
            self.pred_dict[id_].update({str(slice_i): output_img})
            # temp_write_path = os.path.join(self.pred_result_dir, "temp", str(id_))
            # if not os.path.exists(temp_write_path):
            #     os.makedirs(temp_write_path)
            # with h5py.File(os.path.join(temp_write_path, "{}.h5".format(slice_i)), "w") as h5file:
            #     h5file["array"] = output_img
            #     h5file.close()
        printProgressBar(batch_idx + 1, len(self.predict_dataloader()), content="predicting......\n")

    def on_predict_end(self):
        self.predict_toc = time.time()
        time_str = get_duration_time_str(s_time=self.predict_tic, e_time=self.predict_toc)
        print("predicting cost:", time_str)
        # 处理临时文件
        print("Converting 2d to 3d")
        template_path = os.path.join(self.template_dir, os.path.basename(self.test_dir))
        for idx, id_num in enumerate(self.pred_dict.keys()):
            # all_slice = sorted(os.listdir(os.path.join(temp_dir, id_num)))
            template_nii = sitk.ReadImage(os.path.join(template_path, id_num, "T1CE.nii.gz"))
            template_array = sitk.GetArrayFromImage(template_nii)
            pred_array = np.zeros_like(template_array)
            for slice_idx, slice_img in self.pred_dict[id_num].items():
                pred_array[int(slice_idx)] = slice_img
            pred_nii = sitk.GetImageFromArray(pred_array)
            pred_nii.CopyInformation(template_nii)
            sitk.WriteImage(pred_nii, os.path.join(self.pred_result_dir, "{}_pred.nii.gz".format(id_num)))
            printProgressBar(idx, len(self.pred_dict) - 1,
                             content="{}/{} making prediction nii......".format(idx + 1, len(self.pred_dict)))
        # print("remove temp file......")
        # shutil.rmtree(os.path.join(self.pred_result_dir, "temp"))
        print("Done")
