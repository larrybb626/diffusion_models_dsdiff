# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：trainer_use_gaussian_diff.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/11/4 11:23 
"""
import os
import shutil
import sys
import time

import SimpleITK as sitk
import lightning.pytorch as pl
import numpy as np
import torch
from monai.data import Dataset, CacheDataset, decollate_batch, DataLoader, pad_list_data_collate
from monai.metrics import MAEMetric, SSIMMetric
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR

from Disc_diff.guided_diffusion.resample import create_named_schedule_sampler
from UNet_DS_Diff.model import MD_Dis_content
from ldm.models.diffusion.ddpm import DiffusionWrapper
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from loss_function.contrastive_loss import ContrastiveLoss
from training_project.training_transform import *
from training_project.utils.progress_bar import printProgressBar
from training_project.utils.save_tensor_img import tensor2im
from training_project.utils.script_util import sr_create_model_and_diffusion
from training_project.utils.util import print_options
from UNet_DS_Diff.DiT_models import DiT_models


def get_duration_time_str(s_time, e_time):
    h, remainder = divmod((e_time - s_time), 3600)  # 小时和余数
    m, s = divmod(remainder, 60)  # 分钟和秒
    time_str = "%02d h:%02d m:%02d s" % (h, m, s)
    return time_str


class TryTrainerDiffusion_adv(pl.LightningModule):
    def __init__(self, config):
        super(TryTrainerDiffusion_adv, self).__init__()
        self.automatic_optimization = False
        self.config = config
        self.save_hyperparameters()
        self.first_stage_key = "t1ce"
        self.sampler_setting = config.sampler_setting
        # =============================dataset===================================
        self.val_ds = None
        self.train_ds = None
        self.test_ds = None
        self.test_num = None
        self.include_test = config.include_test
        self.num_workers = config.num_workers
        self.train_batch_size = config.train_batch_size
        self.val_batch_size = config.val_batch_size
        self.dataset_type = config.dataset_type
        self.val_transforms = None
        self.train_transforms = None
        self.test_transforms = None
        # ============================model and diffusion============================
        self.learn_sigma = config.learn_sigma
        self.learn_logvar = False
        config.unet_config.params.out_channels = 1 if not self.learn_sigma else 2
        _, self.diffusion = sr_create_model_and_diffusion(config)
        if getattr(config, 'use_edge', False):
            if not hasattr(config.unet_config.params, 'use_edge') or not config.unet_config.params.use_edge:
                print("WARNING: Force setting config.unet_config.params.use_edge to True because config.use_edge is True")
            config.unet_config.params.use_edge = True

        model_config = config.unet_config if getattr(config,"model_type",'unet') == 'unet' else config.ViT_config
        self.model = DiffusionWrapper(model_config, config.conditioning_key)
        self.adv_s = MD_Dis_content()
        self.adv_c = MD_Dis_content()
        self.adv_a = MD_Dis_content()
        self.adv_l = MD_Dis_content()
        # self.model = DiT_models['DiT-B/4'](input_size=320,in_channels=4)
        self.schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, self.diffusion)
        self.clip_denoised = config.clip_denoised
        # =============================fold======================================
        self.fold_K = config.fold_K
        self.fold_idx = config.fold_idx
        # ================================net====================================
        input_channels = len(config.train_keys)
        output_channel = 1
        # ================================loss&metric============================
        self.criterion_dict = None
        self.distance_type = config.disentangle_distance
        self.mae_metric = MAEMetric(reduction="mean", get_not_nans=False)
        self.ssim_metric = SSIMMetric(spatial_dims=2)
        self.get_contrastive_loss = ContrastiveLoss(contrast_mode='all', contrastive_method='cl')
        self.contrast_lambda = config.contrast_lambda
        self.best_val_mae = 1000
        self.best_val_ssim = 0
        self.best_val_epoch = 0
        # =================================optimiser======================================
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.use_ema = config.use_ema
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
        self.val_dir = os.path.join(self.data_dir, "images_val") if self.config.data_name == 'BraTs' else self.train_dir
        # self.val_dir = self.train_dir
        self.test_dir = os.path.join(self.data_dir, "images_ts")
        self.template_dir = config.filepath_img
        self.result_dir = config.root_dir
        self.record_file = os.path.join(config.root_dir, "log_txt.txt")
        self.pred_result_dir = os.path.join(self.result_dir, "pred_nii")
        # 初始化 training output saving
        # self.training_step_output = []
        self.validation_step_outputs = []

    def prepare_data(self):
        # prepare data
        # 根据dir 获取train：0 val：1
        print("preparing data with val")
        if self.config.data_name == 'BraTs':
            datasets = (sorted(os.listdir(self.train_dir)), sorted(os.listdir(self.val_dir)))
            self.print_to_txt(f'train_id:{len(datasets[0])}||valid_id:{len(datasets[1])}')
        else:
            datasets = self.do_split(self.fold_K, self.fold_idx)
        # datasets = self.do_split(self.fold_K, self.fold_idx)
        train_dict = self.get_data_dict(datasets[0], self.train_dir)
        print(len(train_dict), "train data")
        val_dict = self.get_data_dict(datasets[1], self.val_dir)
        test_dict = self.get_test_data_dict()
        self.train_transforms = get_2d_train_transform_diff(keys=self.keys, random_prob=self.random_prob,
                                                            use_edge=getattr(self.config, 'use_edge', False))
        self.val_transforms = get_2d_val_transform_diff(keys=self.keys,
                                                        use_edge=getattr(self.config, 'use_edge', False))
        self.test_transforms = get_2d_test_transform(keys=self.keys, use_edge=getattr(self.config, 'use_edge', False))
        # 获取dataset 方法内直接赋值self.train_ds, self.val_ds, self.test_ds
        self.get_dataset(train_dict, self.train_transforms, mode="train", dataset_type=self.dataset_type)
        self.get_dataset(val_dict, self.val_transforms, mode="val", dataset_type=self.dataset_type)
        self.get_dataset(test_dict, self.test_transforms, mode="test", dataset_type=self.dataset_type)

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
                    cache_rate=1,
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
            # if dataset_type == "normal":
            self.test_ds = Dataset(
                data=data_list,
                transform=transform
            )
            # elif dataset_type == "cache":
            #     self.test_ds = CacheDataset(
            #         data=data_list,
            #         transform=transform,
            #         # cache_num=100,
            #         cache_rate=1,
            #         num_workers=self.num_workers,
            #     )

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
        if self.include_test:
            train_id = np.stack([train_id, test_id], axis=0)
        # 把train_id 保存到txt
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

    def get_val_data_dict(self, id_list):
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
        opt_adv = torch.optim.AdamW(params, lr=lr, betas=(self.beta1, self.beta2))
        return [opt], scheduler

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_mae"] = self.best_val_mae
        checkpoint["best_metric"] = self.best_val_ssim
        checkpoint["best_val_epoch"] = self.best_val_epoch

    def on_load_checkpoint(self, checkpoint):
        self.best_val_mae = checkpoint["best_mae"]
        self.best_val_ssim = checkpoint["best_metric"]
        self.best_val_epoch = checkpoint["best_val_epoch"]

    def on_train_start(self):
        self.print_to_txt("||start with||\n", print_options(self.config))
        # self.model.to(self.device)
        print("Save PyBackup model.py")
        model_dir = self.config.unet_config.target
        model_dir = "../" + "/".join(model_dir.split(".")[:-1]) + '.py'
        shutil.copy(model_dir, os.path.join(self.result_dir, "model.py"))

    def on_train_epoch_start(self):
        self.print_to_txt("⭐epoch: {}⭐".format(self.current_epoch))
        # 起始时间
        self.train_s_time = time.time()

    def training_step(self, batch, batch_idx):
        # for k in batch:
        #     if isinstance(batch[k], MetaTensor):
        #         batch[k] = batch[k].as_tensor()
        x = batch["image"]
        y = batch["t1ce"]
        _cond = dict(t1=x[:, [0]], t2=x[:, [1]], dwi=x[:, [2]])
        if "edge" in batch.keys():
            edge = batch["edge"]
            _cond.update(edge=edge)
            if self.global_step % 100 == 0:
                record_edge = tensor2im(edge[0:1])
                self.logger.experiment.add_image("edge", record_edge, self.global_step,
                                                 dataformats="HWC")

        t, weights = self.schedule_sampler.sample(x.shape[0], self.device)
        # 'contrast' or 'eu'
        losses = self.diffusion.training_losses(self.model, y, t, model_kwargs=_cond, disentangle=self.distance_type,
                                                disen_lambda=self.contrast_lambda)  # dict{'loss':... ,'mse': ...}
        if self.contrast_lambda > 0 and self.distance_type:
            loss_disen = sum([v for k, v in losses.items() if 'disen' in k])
            losses['loss'] = self.contrast_lambda * loss_disen + losses['loss']
        # save heatmap of contrast
        contrast_map = losses.pop('contrast_map', None)
        if contrast_map is not None:
            for k, v in contrast_map.items():
                self.logger.experiment.add_image(f"contrast_map/{k}", v, self.global_step, dataformats="HWC")
        # 出来之后去平均
        for k, v in losses.items():
            losses[k] = v.mean()

        self.log_dict(losses, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return losses

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        # todo: need to change while loss change
        # self.log_dict(outputs)
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

    def on_validation_start(self) -> None:

        self.sample_fn = self.diffusion.ddim_sample_loop if self.sampler_setting.sampler == 'ddim' else \
            (
                self.diffusion.dpm_solver_sample_loop if self.sampler_setting.sampler == 'dpm++' else self.diffusion.p_sample_loop)

    def validation_step(self, batch, batch_idx):
        images, labels = (batch["image"], batch["t1ce"])
        _cond = dict(c_concat=[images])
        if "edge" in batch.keys():
            edge = batch["edge"]
            _cond['c_concat'].append(edge)
        # with self.ema_scope():
        #     _, loss_dict_ema = self.shared_step(batch)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        if self.current_epoch % 1 == 0:
            sample = self.sample_fn(
                self.model,
                (images.shape[0], 1, images.shape[-1], images.shape[-1]),
                clip_denoised=self.clip_denoised,
                model_kwargs=_cond,
                # progress=True
            )
            for b in range(int(images.shape[0] * 0.25)):
                _input = labels[b:b + 1, ...]
                _output = sample[b:b + 1, ...]
                _input = tensor2im(_input)
                _output = tensor2im(_output)

                self.logger.experiment.add_image("fake_img", _output, self.global_step + batch_idx + b,
                                                 dataformats="HWC")

                self.logger.experiment.add_image("real_img", _input, self.global_step + batch_idx + b,
                                                 dataformats="HWC")
            outputs = [i for i in decollate_batch(sample)]
            labels = [i for i in decollate_batch(labels)]
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
        if self.sampler_setting.sample_steps != int(self.config.timestep_respacing):
            self.config.timestep_respacing = str(self.sampler_setting.sample_steps)
            self.diffusion = sr_create_model_and_diffusion(self.config)[1]
        self.sample_fn = self.diffusion.ddim_sample_loop if self.sampler_setting.sampler == 'ddim' else \
            (
                self.diffusion.dpm_solver_sample_loop if self.sampler_setting.sampler == 'dpm' else self.diffusion.p_sample_loop)
        print(self.sample_fn)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        path, images, labels = (batch["path"], batch["image"], batch["t1ce"])
        _cond = dict(c_concat=[images])
        if "edge" in batch.keys():
            edge = batch["edge"]
            _cond['c_concat'].append(edge)
        id_num = [p.split("/")[-2] for p in path]
        slice_idx = [int(os.path.basename(p).split(".")[0].split("_")[-1]) for p in path]
        # roi_x = int(np.ceil(images.shape[2] / 32) * 32)
        # roi_y = int(np.ceil(images.shape[3] / 32) * 32)
        # roi_size = (roi_x, roi_y)
        # sw_batch_size = 8
        B, _, H, W = images.shape
        outputs = self.sample_fn(
            self.model,
            (B, 1, H, W),
            clip_denoised=self.clip_denoised,
            model_kwargs=_cond,
            # progress=True
        )
        # outputs = outputs["samples"]
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
            ce_name = "T1CE.nii.gz" if self.config.data_name == 'prostate' else "ce.nii.gz"
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
