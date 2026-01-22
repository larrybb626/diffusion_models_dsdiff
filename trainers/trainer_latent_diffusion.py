import os
import sys
import time
from contextlib import nullcontext
from inspect import isfunction

import SimpleITK as sitk
import lightning.pytorch as pl
import numpy as np
import torch
from diffusers import AutoencoderKL
from monai.data import Dataset, CacheDataset, decollate_batch, DataLoader, pad_list_data_collate, SmartCacheDataset
from monai.metrics import MAEMetric, SSIMMetric
from monai.optimizers import WarmupCosineSchedule
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import ImageEmbeddingConditionedLatentDiffusion, LatentDiffusion
from ldm.models.diffusion.dpm_solver_new import DPMSolverSampler
from ldm.util import log_txt_as_img
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


class LatentDiffusionModel(ImageEmbeddingConditionedLatentDiffusion, pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__(embedding_key="image", *args, **kwargs)
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
        self.val_dir = os.path.join(self.data_dir, "images_val") if getattr(self.config, "data_name",
                                                                            "prostate") == 'BraTs' else self.train_dir
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

    def instantiate_first_stage(self, config, ckpt=None):
        model = AutoencoderKL.from_single_file(
            sys.argv[0].split("/newnas")[0] + "/newnas/MJY_file/vae-ft-mse-840000-ema-pruned.ckpt",
            original_config="../configs/v1-inference.yaml")
        if ckpt:
            print(f"use {ckpt} to load first stage model")
            state_dict = torch.load(ckpt, map_location="cuda:4")["state_dict"]
            keys = list(state_dict.keys())
            for k in keys:
                if "loss" in k:
                    state_dict.pop(k)
            model.load_state_dict(state_dict)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def get_input(self, batch, k, cond_key=None, bs=None, **kwargs):
        outputs = LatentDiffusion.get_input(self, batch, k, bs=bs, **kwargs)
        z, c = outputs[0], outputs[1]
        img = batch[self.embed_key][:bs]
        if img.shape[1] > 3:
            img = list(torch.chunk(img, len(self.keys), 1))
        # img = rearrange(img, 'b h w c -> b c h w')
        c_adm = None
        if "adm" in self.model.conditioning_key:
            c_adm = self.embedder(img)
            if self.noise_augmentor is not None:
                c_adm, noise_level_emb = self.noise_augmentor(c_adm)
                # assume this gives embeddings of noise levels
                c_adm = torch.cat((c_adm, noise_level_emb), 1)
            if self.training:
                c_adm = torch.bernoulli((1. - self.embedding_dropout) * torch.ones(c_adm.shape[0],
                                                                                   device=c_adm.device)[:,
                                                                        None]) * c_adm

        all_conds = {"c_crossattn": [c], "c_adm": c_adm}
        if self.model.conditioning_key == "concat":
            c_cat = list()
            if not isinstance(img, list):
                img = [img]
            self.concat_keys = [self.embed_key]
            for ck in range(len(img)):
                cc = img[ck]
                if bs is not None:
                    cc = cc[:bs]
                    cc = cc.to(self.device)
                cc = self.encode_first_stage(cc).latent_dist
                cc = cc.sample()
                cc = self.get_first_stage_encoding(cc)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            all_conds.update({"c_concat": [c_cat]})
        noutputs = [z, all_conds]
        noutputs.extend(outputs[2:])
        return noutputs

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

    def get_data_dict(self, id_list, data_dir):
        # 输入id的list获取数据字典
        data_dict = []
        for id_num in id_list:
            layer_list = sorted(os.listdir(os.path.join(data_dir, id_num)))
            # layer_list = [layer_list] * 4
            for layer in layer_list:  # 头尾不要?
                new_data_dict = {"path": os.path.join(data_dir, id_num, layer), "txt": ""}
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
            # if getattr(self.config, "data_name", "prostate") == 'BraTs' and sys.gettrace() is None:
            #     self.train_ds = SmartCacheDataset(
            #         data=data_list,
            #         transform=transform,
            #         # cache_num=300,
            #         cache_rate=0.2,
            #     )
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

    def on_load_checkpoint(self, checkpoint):
        self.best_val_mae = checkpoint["best_mae"]
        self.best_val_ssim = checkpoint["best_metric"]
        self.best_val_epoch = checkpoint["best_val_epoch"]

    def prepare_data(self):
        # prepare data
        # 根据dir 获取train：0 val：1
        print("preparing data with val")
        # datasets = sorted(os.listdir(self.train_dir))
        print("preparing data with val")
        if getattr(self.config, "data_name", "prostate") == 'BraTs':
            datasets = (sorted(os.listdir(self.train_dir)), sorted(os.listdir(self.val_dir)))
            self.print_to_txt(f'train_id:{len(datasets[0])}||valid_id:{len(datasets[1])}')
        else:
            datasets = self.do_split(self.fold_K, self.fold_idx)
        # datasets = self.do_split(self.fold_K, self.fold_idx)
        train_dict = self.get_data_dict(datasets[0], self.train_dir)
        val_dict = self.get_data_dict(datasets[1], self.val_dir)
        test_dict = self.get_test_data_dict()
        self.train_transforms = get_2d_rgb_train_transform(keys=self.keys, random_prob=self.random_prob)
        self.val_transforms = get_2d_rgb_val_transform(keys=self.keys)
        self.test_transforms = get_2d_rgb_test_transform(keys=self.keys)
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
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        print("Setting up CosineAnnealingLR scheduler...")
        scheduler = CosineAnnealingLR(opt, self.max_epochs, eta_min=self.min_lr) \
            if self.warmup_epochs == 0 else \
            WarmupCosineSchedule(opt, cycles=0.5, warmup_steps=self.warmup_epochs, t_total=self.max_epochs)
        scheduler = [
            {
                'scheduler': scheduler,
                'interval': 'epoch',
            }]
        return [opt], scheduler

    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, is_ddim=False):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)
        yt_shape = list(y_cond.shape)
        yt_shape[1] = 1
        y_t = default(y_t, lambda: torch.randn(yt_shape, device=self.device))
        y_t, ret_arr = self.p_sample_ddim(model=self.net_G, y_t=y_t, y_cond=y_cond, batch_size=b, ddim_timesteps=50,
                                          ddim_discr_method="quad", ddim_eta=0, clip_denoised=True,
                                          sample_inter=sample_inter)
        return y_t, ret_arr

    # may not use
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

    def on_train_start(self):
        self.print_to_txt("||start with||\n", print_options(self.config))

    def on_train_epoch_start(self):
        self.print_to_txt("⭐epoch: {}⭐".format(self.current_epoch))
        # 起始时间
        self.train_s_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
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

    def log_images(self, batch, N=8, n_row=4, **kwargs):
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True,
                                           return_original_cond=True)
        log["inputs"] = x
        log["reconstruction"] = xrec
        assert self.model.conditioning_key is not None
        assert self.cond_stage_key in ["caption", "txt"]
        xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
        log["conditioning"] = xc

        if self.model.conditioning_key != "concat":
            uc = self.get_unconditional_conditioning(N, kwargs.get('unconditional_guidance_label', ''))
        else:
            uc = None
        unconditional_guidance_scale = kwargs.get('unconditional_guidance_scale', 5.)

        uc_ = {"c_crossattn": [uc], "c_adm": c["c_adm"]}
        if self.model.conditioning_key == "concat":
            uc_.update({"c_concat": c["c_concat"]})
        ema_scope = self.ema_scope if kwargs.get('use_ema_scope', True) else nullcontext
        with ema_scope(f"Sampling"):
            samples_cfg, _ = self.sample_log(cond=c, batch_size=N, sampler=self.sampler_setting.sampler,
                                             ddim_steps=self.sampler_setting.sample_steps,
                                             pred_mode=True,
                                             ddim_use_original_steps=self.sampler_setting.ddim_use_original_steps,
                                             ddim_eta=self.sampler_setting.get("ddim_eta", 0.),
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_,
                                             clip_denoised=kwargs.get('clip_denoised', True))
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samplescfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
        return log

    def sample_log(self, cond, batch_size, sampler, ddim_steps, **kwargs):
        shape = (self.channels, self.image_size, self.image_size)
        actual_shape = cond['c_concat'][0].shape[2:]
        if tuple(actual_shape) != shape[1:]:
            shape = (self.channels, *actual_shape)
        if sampler == "dpm":
            dpm_sampler = DPMSolverSampler(self)
            samples, intermediates = dpm_sampler.sample(ddim_steps, batch_size,
                                                        shape, cond, **kwargs)
        elif sampler == "ddim":
            ddim_sampler = DDIMSampler(self)
            # eta=0 ddim/eta=1 ddpm mekf
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, eta=kwargs.get("ddim_eta"),
                                                         **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)
        return samples, intermediates

    # def on_validation_epoch_start(self):
    #     print('?????????????????????????')
    # def validation_step(self, batch, batch_idx):
    #     images, labels = (batch["image"], batch["t1ce"])
    #
    #     output, visual_img = self.restoration(images, is_ddim=False)
    #
    #     # 每15个step保存图片
    #     # save_image_2d(images, output, labels, self.result_dir, batch_idx, every_n_step=50, mode="Val")
    #     if batch_idx % 1 == 0:
    #         for b in range(images.shape[0]):
    #             _labels = labels[b:b + 1, ...]
    #             _output = output[b:b + 1, ...]
    #             _visual_img = visual_img[b:b + 1, ...]
    #             img_realB = tensor2im(_labels)
    #             img_fakeB = tensor2im(_output)
    #             img_restore = tensor2im(_visual_img)
    #             self.logger.experiment.add_image("val_real_B", img_realB, self.global_step + b, dataformats="HWC")
    #             self.logger.experiment.add_image("val_fake_B", img_fakeB, self.global_step + b, dataformats="HWC")
    #             self.logger.experiment.add_image("restore_procedure", img_restore, self.global_step + b,
    #                                              dataformats="HWC")
    #
    #         # self.log_dict({"val_real_B": img_realB,
    #         #                "val_fake_B": img_fakeB})
    #     # data_range = torch.max(labels) - torch.min(labels)
    #     outputs = [i for i in decollate_batch(output)]
    #     labels = [i for i in decollate_batch(labels)]
    #     self.mae_metric(y_pred=outputs, y=labels)
    #     self.ssim_metric(y_pred=outputs, y=labels)
    #     # outputs = [(i - torch.mean(i)) / (torch.std(i) / 400) + 2048 for i in outputs]
    #     # labels = [(i - torch.mean(i)) / (torch.std(i) / 400) + 2048 for i in labels]
    #     # self.ssim_metric = SSIMMetric(spatial_dims=2, data_range=data_range)
    #
    #     self.validation_step_outputs.append({"val_number": len(outputs)})
    @ torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, labels = (batch["image"], batch["t1ce"])

        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        if self.current_epoch % 1 == 0:
            log = self.log_images(batch=batch, use_ema_scope=False, N=len(batch["txt"]),
                                  unconditional_guidance_scale=1.0, clip_denoised=self.clip_denoised)
            # print(log)
            for k in log:
                if "samplescfg" in k:
                    samplescfg_key = k
                    break
            for b in range(images.shape[0]):
                _input = log['inputs'][b:b + 1, ...]
                _output = log[samplescfg_key].sample[b:b + 1, ...]
                _rec_img = log['reconstruction'].sample[b:b + 1, ...]
                _input = tensor2im(_input)
                _output = tensor2im(_output)
                _rec_img = tensor2im(_rec_img)
                self.logger.experiment.add_image("fake_img", _output, self.global_step + b, dataformats="HWC")
                self.logger.experiment.add_image("reconstruction_img", _rec_img, self.global_step + b,
                                                 dataformats="HWC")
                self.logger.experiment.add_image("real_img", _input, self.global_step + b,
                                                 dataformats="HWC")
            outputs = [i for i in decollate_batch(log[samplescfg_key].sample)]
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
        outputs = self.log_images(batch=batch, use_ema_scope=False, N=len(batch["txt"]),
                                  unconditional_guidance_scale=1.0, clip_denoised=self.clip_denoised)
        for k in outputs:
            if "samplescfg" in k:
                samplescfg_key = k
                break
        outputs = outputs[samplescfg_key].sample
        return id_num, outputs, slice_idx

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # 临时文件-2d
        for id_, slice_i, img in zip(outputs[0], outputs[2], outputs[1]):
            output_img = ((img[0, :, :] + img[1, :, :] + img[2, :, :]) / 3).cpu().numpy()
            output_img = np.clip(output_img, -1, 1)
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
            ce_name = "T1CE.nii.gz" if getattr(self.config, "data_name", 'prostate') == 'prostate' else "ce.nii.gz"
            template_nii = sitk.ReadImage(os.path.join(template_path, id_num, ce_name))
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
