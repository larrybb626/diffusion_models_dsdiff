import os
import shutil
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

from Disc_diff.guided_diffusion.gaussian_diffusion import L1_Charbonnier_loss
from Disc_diff.guided_diffusion.losses import discretized_gaussian_log_likelihood
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import DDPM
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.dpm_solver_new import DPMSolverSampler
from ldm.modules.diffusionmodules.util import noise_like, mean_flat
from ldm.modules.distributions.distributions import normal_kl
from loss_function.contrastive_loss import ContrastiveLoss
from training_project.training_transform import *
from training_project.utils.progress_bar import printProgressBar
from training_project.utils.save_tensor_img import tensor2im
from training_project.utils.util import print_options, get_heatmap


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


class DSDiffModel(DDPM, pl.LightningModule):
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
        self.conclude_test = False
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
        self.get_contrastive_loss = ContrastiveLoss(contrast_mode='all', contrastive_method='cl')
        self.contrast_lambda = config.contrast_lambda
        self.contrast = False  # 欧氏距离或对比损失
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
        # self.val_dir = os.path.join(self.data_dir, "images_val") if self.config.data_name == 'BraTs' else self.train_dir
        self.val_dir = self.train_dir
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
        # if self.config.data_name == 'BraTs':
        #     datasets = (sorted(os.listdir(self.train_dir)), sorted(os.listdir(self.val_dir)))
        #     self.print_to_txt(f'train_id:{len(datasets[0])}||valid_id:{len(datasets[1])}')
        # else:
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
        if self.conclude_test:
            train_id = np.stack([train_id, test_id], axis=0)
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
        return [opt], scheduler

    def get_disentangle_loss(self, feature, label, contrast=True, temperature=0.1):
        assert feature.shape[0] == label.shape[0]
        if contrast:
            return self.get_contrastive_loss(feature, label, temperature=temperature)
        else:
            label = torch.cat(torch.unbind(label, dim=1), dim=0)
            label = label.contiguous().view(-1, 1).to(self.device)
            feature = torch.cat(torch.unbind(feature, dim=1), dim=0)
            feature = feature.view(feature.shape[0], -1)
            logits = torch.cdist(feature, feature, p=1)
            logits = logits / feature.shape[1]
            mask = torch.eq(label, label.T)
            perfect_logit = torch.where(mask == 1, torch.tensor(1.0), torch.tensor(-1.0))
            eye_mask = torch.eye(label.shape[0], dtype=torch.bool).to(self.device)
            numerator = (logits * ~eye_mask * mask).sum()
            denominator = (logits * ~mask).sum()
            loss = torch.div(numerator, denominator)
            return loss, logits, perfect_logit

    def get_loss(self, pred, target, mean=True, contrast=True, **kwargs):
        content_style_loss = None
        anatomy_lesion_loss = None
        hs_list_dict = pred[1]
        pred = pred[0]
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == 'charbonnie':
            loss = L1_Charbonnier_loss()(target, pred)
            if mean:
                loss = loss.mean()
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        anatomy_feature = torch.stack(hs_list_dict['anatomy'], dim=1)  # [[b, c, h, w], ...]->[b, n, c, h, w]
        lesion_feature = torch.stack(hs_list_dict['lesion'], dim=1)  # [b, n, c, h, w]
        style_feature = torch.stack(hs_list_dict['style'], dim=1)
        content_feature = torch.stack(hs_list_dict['content'], dim=1)
        # n_style_content = torch.stack(hs_list_dict['n_style_content'],
        #                               dim=1)  # [[h_style, h_n_style, h_share_content, h_n_content]]
        c_s_feature = torch.cat([content_feature, style_feature], dim=1)
        a_l_feature = torch.cat([anatomy_feature, lesion_feature], dim=1)

        # # todo 一次做完c-s-a-l
        # c_s_a_l_feature = torch.cat([c_s_feature, a_l_feature], dim=1)
        # c_s_a_l_label = []
        # for b in range(content_feature.shape[0]):
        #     c_label = [b * 3] * content_feature.shape[1]
        #     s_label = [-1 - j for j in range(style_feature.shape[1])]
        #     a_label = [3 * b + 1] * anatomy_feature.shape[1]
        #     l_label = [3 * b + 2] * lesion_feature.shape[1]
        #     c_s_a_l_label.append(
        #         torch.tensor(c_label + s_label + a_label + l_label))
        # c_s_a_l_label = torch.stack(c_s_a_l_label)
        # c_s_a_l_loss, logit_c_s_a_l_s, perfect_logit_c_s_a_l = self.get_contrastive_loss(c_s_a_l_feature, c_s_a_l_label,
        #                                                                                  temperature=0.05)
        # C-S disentangle
        c_s_label = []
        for b in range(content_feature.shape[0]):
            c_s_label.append(
                torch.tensor([b] * content_feature.shape[1] + [-1 - j for j in range(style_feature.shape[1])]))
        # c_s_label: [[0,0,0,-1,-2,-3],[1,1,1,-1,-2,-3],...]
        c_s_label = torch.stack(c_s_label)

        # todo S-A-L disentangle
        s_a_l_feature = torch.cat([style_feature, a_l_feature], dim=1)
        s_a_l_label = []
        for b in range(s_a_l_feature.shape[0]):
            label_a = 2 * b
            label_l = 2 * b + 1
            s_a_l_label.append(
                torch.tensor(
                    [-1 - j for j in range(style_feature.shape[1])] + [label_a] * anatomy_feature.shape[1] + [
                        label_l] *
                    lesion_feature.shape[1]))
            # a_l_label: [[-1,-2,-3,0,0,1,1],[-1,-2,-3,2,2,3,3],...]
        s_a_l_label = torch.stack(s_a_l_label)

        n_style_content_label = []
        # for i in range(n_style_content.shape[0]):
        #     n_style_content_label.append(
        #         torch.tensor([-1, -1] + [i] * (n_style_content.shape[1] // 2)))  # [-1,-1, 0,0],[-1,-1,1,1],...
        # n_style_content_label = torch.stack(n_style_content_label)

        content_style_loss, logit, perfect_logit = self.get_disentangle_loss(c_s_feature, c_s_label, contrast,
                                                                             temperature=0.05)
        c_s_heatmap = get_heatmap(logit.detach())
        perfect_c_s_heatmap = get_heatmap(perfect_logit)
        style_anatomy_lesion_loss, logit, perfect_logit = self.get_disentangle_loss(s_a_l_feature, s_a_l_label,
                                                                                    contrast)
        s_a_l_heatmap = get_heatmap(logit.detach())
        perfect_heatmap = get_heatmap(perfect_logit)

        # n_style_content_loss, logit, perfect_logit = self.get_disentangle_loss(n_style_content,
        #                                                                        n_style_content_label, contrast)
        # n_style_content_heatmap = get_heatmap(logit.detach())
        # perfect_n_style_content_heatmap = get_heatmap(perfect_logit)
        # content_loss, logit_c, perfect_logit_c = self.get_contrastive_loss(content_feature, c_label, temperature=0.05)
        # style_loss, logit_s, perfect_logit_s = self.get_contrastive_loss(style_feature, s_label, temperature=0.05)
        # save heatmap

        # self.logger.experiment.add_image("heatmap/s_a_l_heatmap", s_a_l_heatmap, self.global_step, dataformats="HWC")
        # self.logger.experiment.add_image("heatmap/s_a_l_perfect_heatmap", perfect_heatmap, self.global_step,
        #                                  dataformats="HWC")
        # # self.logger.experiment.add_image("heatmap/c_s_a_l_heatmap", c_s_a_l_loss_heatmap, self.global_step,
        # #                                  dataformats="HWC")
        # # self.logger.experiment.add_image("heatmap/c_s_a_l_perfect_heatmap", perfect_c_s_a_l_loss_heatmap,
        # #                                  self.global_step,
        # #                                  dataformats="HWC")
        # self.logger.experiment.add_image("heatmap/c_s_heatmap", c_s_heatmap, self.global_step, dataformats="HWC")
        # self.logger.experiment.add_image("heatmap/c_s_perfect_heatmap", perfect_c_s_heatmap, self.global_step,
        #                                  dataformats="HWC")
        # self.logger.experiment.add_image("heatmap/n_s_c_heatmap", n_style_content_heatmap, self.global_step,
        #                                     dataformats="HWC")
        # self.logger.experiment.add_image("heatmap/n_s_c_perfect_heatmap", perfect_n_style_content_heatmap,
        #                                     self.global_step,
        #                                     dataformats="HWC")
        # # self.log("train/c_s_a_l_loss", c_s_a_l_loss)
        # self.log("train/c-s_loss", content_style_loss)
        # self.log("train/s-a-l_loss", style_anatomy_lesion_loss)
        # self.log("train/n_style_content_loss", n_style_content_loss)

        # total disentangle loss
        disentangle_loss = {"c-s": content_style_loss, "s-a-l": style_anatomy_lesion_loss}

        return loss, disentangle_loss

    def on_train_start(self):
        self.print_to_txt("||start with||\n", print_options(self.config))
        # self.model.to(self.device)

    def on_train_epoch_start(self):
        self.print_to_txt("⭐epoch: {}⭐".format(self.current_epoch))
        # 起始时间
        self.train_s_time = time.time()

    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val
        # for k in batch:
        #     if isinstance(batch[k], MetaTensor):
        #         batch[k] = batch[k].as_tensor()
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        ret_dict = dict()
        ret_dict["loss"] = loss
        for k in loss_dict:
            ret_dict[k] = loss_dict[k]
        return ret_dict

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
        log_prefix = 'train' if self.training else 'val'

        # loss_tuple = self.get_loss(model_out, target, mean=False)
        # loss_simple = loss_tuple[0].mean(dim=[1, 2, 3])
        # loss_dict.update({f'{log_prefix}/loss_simple': loss_simple.mean()})
        # self.logvar = self.logvar.to(t.device)
        # logvar_t = self.logvar[t].to(self.device)
        # loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # if self.learn_logvar:
        #     loss_dict.update({f'{log_prefix}/loss_gamma': loss.mean()})
        #     loss_dict.update({'logvar': self.logvar.data.mean()})
        # # loss_dict.update({f'{log_prefix}/c-s-a-l_loss': loss_tuple[1]["c-s-a-l"]})
        # for k in loss_tuple[1]:
        #     loss_dict[f'{log_prefix}/{k}'] = loss_tuple[1][k]
        # # loss_vlb = (self.lvlb_weights[t] * self._vb_terms_bpd(model=None,x_start=x_start, x_t=x_noisy, t=t,**kwargs)["output"]).mean()
        # loss_vlb = (self.lvlb_weights[t] * loss).mean()
        # loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})
        #
        # loss = loss.mean() + self.original_elbo_weight * loss_vlb + self.contrast_lambda * (
        #     sum([v for v in loss_tuple[1].values()]))
        # # loss = self.contrast_lambda * (loss_tuple[1]["c-s"]+loss_tuple[1]["a-l"])
        # loss_dict.update({f'{log_prefix}/loss': loss})

        loss_tuple = self.get_loss(model_out, target, mean=False, contrast=self.contrast)
        loss = loss_tuple[0].mean(dim=[1, 2, 3])
        log_prefix = 'train' if self.training else 'val'
        # loss_dict.update({f'{log_prefix}/c-s_loss': loss_tuple[1]["c-s"]})
        # loss_dict.update({f'{log_prefix}/a-l_loss': loss_tuple[1]["a-l"]})
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})

        # loss_dict.update({f'{log_prefix}/c-s-a-l_loss': loss_tuple[1]["c-s-a-l"]})
        for k in loss_tuple[1]:
            loss_dict[f'{log_prefix}/{k}'] = loss_tuple[1][k]
        loss_simple = loss.mean() * self.l_simple_weight
        # loss_vlb = (self.lvlb_weights[t] * self._vb_terms_bpd(model=None,x_start=x_start, x_t=x_noisy, t=t,**kwargs)["output"]).mean()
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb + self.contrast_lambda * (
            sum([v for v in loss_tuple[1].values()]))
        # loss = self.contrast_lambda * (loss_tuple[1]["c-s"]+loss_tuple[1]["a-l"])
        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, **kwargs
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(x_t, t, clip_denoised=clip_denoised, cond=kwargs)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out[0], out[2]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out[0], log_scales=0.5 * out[2]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": None}

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
