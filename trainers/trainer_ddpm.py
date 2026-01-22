import os
import sys
import time
from inspect import isfunction

import SimpleITK as sitk
import lightning.pytorch as pl
import numpy as np
import torch
from monai.data import Dataset, CacheDataset, decollate_batch, DataLoader, pad_list_data_collate
from monai.metrics import MAEMetric, SSIMMetric
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DDPM, DiffusionWrapper_for_other_model
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.dpm_solver_new import DPMSolverSampler
from ldm.modules.diffusionmodules.util import noise_like
from training_project.training_transform import *
from training_project.utils.progress_bar import printProgressBar
from training_project.utils.save_tensor_img import tensor2im
from training_project.utils.util import print_options


def get_duration_time_str(s_time, e_time):
    h, remainder = divmod((e_time - s_time), 3600)  # 小时和余数
    m, s = divmod(remainder, 60)  # 分钟和秒
    time_str = "%02d h:%02d m:%02d s" % (h, m, s)
    return time_str


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPMModel(DDPM, pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.save_hyperparameters()
        self.first_stage_key = "t1ce"
        self.sampler_setting = config.sampler_setting
        # =============================dataset===================================
        self.val_ds = None
        self.train_ds = None
        self.test_ds = None
        self.test_num = None
        self.num_workers = config.num_workers
        self.train_batch_size = config.train_batch_size
        self.val_batch_size = config.val_batch_size
        self.dataset_type = config.dataset_type
        self.val_transforms = None
        self.train_transforms = None
        self.test_transforms = None
        # =============================fold======================================
        self.fold_K = config.fold_K
        self.fold_idx = config.fold_idx
        # ================================net====================================
        input_channels = len(config.train_keys)
        output_channel = 1
        # ================================loss&metric============================
        self.criterion_dict = None
        self.mae_metric = MAEMetric(reduction="mean", get_not_nans=False)
        self.ssim_metric = SSIMMetric(spatial_dims=2)

        self.best_val_mae = 1000
        self.best_val_ssim = 0
        self.best_val_epoch = 0
        # =================================optimiser======================================
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # ====================================lr===========================================
        self.max_lr = config.lr
        self.learning_rate = self.max_lr
        self.min_lr = config.lr_low
        # =============================training setting=====================================
        self.random_state = config.seed
        self.random_prob = config.augmentation_prob
        self.max_epochs = config.num_epochs
        self.warmup_epochs = config.lr_warm_epoch
        self.cos_epochs = config.lr_cos_epoch
        self.epoch_loss_values = []
        # =============================training variable====================================
        self.train_s_time = 0
        self.train_e_time = 0
        self.val_s_time = 0
        self.val_e_time = 0
        self.predict_tic = None
        self.predict_toc = None
        self.keys = config.train_keys
        # =============================文件地址=======================================
        self.data_dir = config.h5_2d_img_dir
        self.train_dir = os.path.join(self.data_dir, "images_tr")
        self.test_dir = os.path.join(self.data_dir, "images_ts")
        self.template_dir = config.filepath_img
        self.result_dir = config.root_dir
        self.record_file = os.path.join(config.root_dir, "log_txt.txt")
        self.pred_result_dir = os.path.join(self.result_dir, "pred_nii")
        # 初始化 training output saving
        # self.training_step_output = []
        self.validation_step_outputs = []

        if self.model.conditioning_key == "crossattn-adm":
            print("Loading v2-1 ckpt")
            state_dict = torch.load(
                sys.argv[0].split("/newnas")[
                    0] + "/newnas/MJY_file/CE-MRI/stable_diffusion_2-1_unclip/sd21-unclip-h.ckpt")["state_dict"]
            new_state_dict = {}
            first_stage_dict = {}
            for k, v in state_dict.items():
                if "model.diffusion_model" in k:
                    new_state_dict.update({k[6:]: v})
                if "first_stage_model" in k:
                    first_stage_dict.update({k[len("first_stage_model."):]: v})
            # self.model.load_state_dict(new_state_dict, strict=False)
            self.first_stage_model.load_state_dict(first_stage_dict, strict=True)

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
                new_data_dict = {"path": os.path.join(self.train_dir, id_num, layer), "txt": ""}
                data_dict.append(new_data_dict)
        return data_dict

    def get_test_data_dict(self, ):
        # 输入test_id的list获取数据字典
        data_dict = []
        id_list = sorted(os.listdir(self.test_dir))[:self.test_num]
        for id_num in id_list:
            layer_list = sorted(os.listdir(os.path.join(self.test_dir, id_num)))
            for layer in layer_list:
                new_data_dict = {"path": os.path.join(self.test_dir, id_num, layer), "txt": ""}
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
                    cache_rate=0.8,
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
                    cache_rate=0,
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
            num_workers=self.num_workers if sys.gettrace() is None else 1,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return val_loader

    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return pred_loader

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        print("Setting up CosineAnnealingLR scheduler...")
        scheduler = [
            {
                'scheduler': CosineAnnealingLR(opt, self.max_epochs, eta_min=self.min_lr),
                'interval': 'epoch',
            }]
        return [opt], scheduler

    def on_train_start(self):
        self.print_to_txt("||start with||\n", print_options(self.config))

    def on_train_epoch_start(self):
        self.print_to_txt("⭐epoch: {}⭐".format(self.current_epoch))
        # 起始时间
        self.train_s_time = time.time()

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
        if self.use_ema:
            self.model_ema(self.model)

    def on_train_epoch_end(self):
        self.train_e_time = time.time()
        time_str = get_duration_time_str(s_time=self.train_s_time, e_time=self.train_e_time)
        lr = self.optimizers().optimizer.param_groups[0]['lr']
        self.print_to_txt(
            "Epoch Done || lr: {} || Epoch cost: {}".format(lr, time_str))
        self.log("lr", lr)

    def get_input(self, batch, k, **kwargs):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        cond = batch[kwargs["cond_k"]]
        return x, cond

    def shared_step(self, batch):
        x, c = self.get_input(batch, self.first_stage_key, cond_k="image")
        loss, loss_dict = self(x, c_concat=[c])
        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def p_losses(self, x_start, t, noise=None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, **kwargs)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, sampler="ddim", pred_mode=False,
                   ddim_eta=0, **kwargs):
        log = dict()
        x, c = self.get_input(batch, self.first_stage_key, cond_k="image")
        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        c = c.to(self.device)[:N]
        log["inputs"] = x
        ddim_steps = kwargs["ddim_steps"]
        shape = x.shape[1:]
        if sample:
            if sampler == "dpm":
                dpm_sampler = DPMSolverSampler(self)
                samples, intermediates = dpm_sampler.sample(ddim_steps, N,
                                                            shape, c, **kwargs)
                if intermediates:
                    for idx, img in enumerate(intermediates["x_inter"]):
                        intermediates["x_inter"][idx] = img[:int(0.25 * img.shape[0]), ...]
                log["samples"] = samples
                if not pred_mode and intermediates:
                    log["denoise_row"] = self._get_rows_from_list(intermediates["x_inter"])
            elif sampler == "ddim":
                ddim_sampler = DDIMSampler(self)

                # eta=0 ddim/eta=1 ddpm mekf
                samples, intermediates = ddim_sampler.sample(ddim_steps, N,
                                                             shape, c, verbose=False, eta=ddim_eta, **kwargs)
                for idx, img in enumerate(intermediates["x_inter"]):
                    intermediates["x_inter"][idx] = img[:int(0.25 * img.shape[0]), ...]
                log["samples"] = samples
                if not pred_mode:
                    log["denoise_row"] = self._get_rows_from_list(intermediates["x_inter"])
            else:
                # get denoise row
                # with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, cond=c, shape=shape, return_intermediates=True, )

                log["samples"] = samples
                if not pred_mode:
                    log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def sample(self, batch_size=16, cond=None, shape=None, return_intermediates=False, *args, **kwargs):

        return self.p_sample_loop((batch_size, shape[0], shape[1], shape[2],), cond,
                                  return_intermediates=return_intermediates)

    def p_sample_loop(self, shape, cond=None, return_intermediates=False, *args, **kwargs):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond,
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    def p_sample(self, x, t, cond=None, clip_denoised=True, repeat_noise=False, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, *args, **kwargs):
        model_out = self.apply_model(x, t, cond)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == "v":
            x_recon = self.predict_start_from_z_and_v(x, t, model_out)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def validation_step(self, batch, batch_idx):
        images, labels = (batch["image"], batch["t1ce"])

        _, loss_dict_no_ema = self.shared_step(batch)
        # with self.ema_scope():
        #     _, loss_dict_ema = self.shared_step(batch)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        if self.current_epoch % 1 == 0:
            log = self.log_images(batch=batch, use_ema_scope=False, N=len(batch["image"]),
                                  sampler=self.sampler_setting.sampler, ddim_steps=self.sampler_setting.sample_steps)

            for b in range(int(images.shape[0] * 0.25)):
                _input = log['inputs'][b:b + 1, ...]
                _output = log["samples"][b:b + 1, ...]
                _input = tensor2im(_input)
                _output = tensor2im(_output)

                self.logger.experiment.add_image("fake_img", _output, self.global_step + batch_idx + b,
                                                 dataformats="HWC")

                self.logger.experiment.add_image("real_img", _input, self.global_step + batch_idx + b,
                                                 dataformats="HWC")

            _rec_img = tensor2im(log['denoise_row'].unsqueeze(0))
            self.logger.experiment.add_image("reconstruction_img", _rec_img, self.global_step + batch_idx,
                                             dataformats="HWC")
            outputs = [i for i in decollate_batch(log["samples"])]
            labels = [i for i in decollate_batch(log['inputs'])]
            self.mae_metric(y_pred=outputs, y=labels)
            self.ssim_metric(y_pred=outputs, y=labels)

        # self.validation_step_outputs.append({"val_number": len(outputs)})

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0):
        printProgressBar(batch_idx, len(self.val_dataloader()) - 1,
                         content="{}/{} validation processing......".format(batch_idx + 1, len(self.val_dataloader())))

    def on_validation_epoch_end(self):
        val_loss, num_items, mean_val_ssim = 0, 0, 0
        mean_val_ssim = self.ssim_metric.aggregate().item()
        mean_val_mae = self.mae_metric.aggregate().item()
        self.mae_metric.reset()
        self.ssim_metric.reset()
        if mean_val_ssim > self.best_val_ssim:
            self.best_val_ssim = mean_val_ssim
            self.best_val_epoch = self.current_epoch
        if mean_val_mae < self.best_val_mae:
            self.best_val_mae = mean_val_mae
        self.print_to_txt(
            f"val_loss: No use"
            f" || current mean SSIM: {mean_val_ssim:.4f}"
            f" || best mean SSIM: {self.best_val_ssim:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.print_to_txt(
            f"current mean MAE: {mean_val_mae:.4f}"
            f" || best mean MAE: {self.best_val_mae:.4f} "
        )

        self.log_dict({
            "val/ssim": mean_val_ssim,
            "val/mae": mean_val_mae,
        })

    def on_predict_start(self):
        self.predict_tic = time.time()
        file_list = sorted(os.listdir(self.test_dir))
        self.pred_dict = dict(zip(file_list, [{} for i in range(len(file_list))]))
        # self.prepare_noise_schedule(phase="test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        path, images, labels = (batch["path"], batch["image"], batch["t1ce"])
        id_num = [p.split("/")[-2] for p in path]
        slice_idx = [int(os.path.basename(p).split(".")[0].split("_")[-1]) for p in path]
        # roi_x = int(np.ceil(images.shape[2] / 32) * 32)
        # roi_y = int(np.ceil(images.shape[3] / 32) * 32)
        # roi_size = (roi_x, roi_y)
        # sw_batch_size = 8
        outputs = self.log_images(batch=batch, use_ema_scope=False, N=len(batch["image"]),
                                  unconditional_guidance_scale=1.0, sampler=self.sampler_setting.sampler,
                                  ddim_steps=self.sampler_setting.sample_steps,
                                  pred_mode=True, ddim_use_original_steps=self.sampler_setting.ddim_use_original_steps,
                                  ddim_eta=self.sampler_setting.ddim_eta)
        outputs = outputs["samples"]
        return id_num, outputs, slice_idx

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # 临时文件-2d
        for id_, slice_i, img in zip(outputs[0], outputs[2], outputs[1]):
            output_img = img.cpu().numpy()
            self.pred_dict[id_].update({str(slice_i): output_img})
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
