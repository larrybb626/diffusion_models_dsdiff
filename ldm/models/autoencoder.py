import os
import random
import sys
import time
from contextlib import contextmanager

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL as hugging_face_autoencoder
from monai.data import Dataset, CacheDataset, DataLoader, pad_list_data_collate
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config
from training_project.training_transform import *
from training_project.utils.progress_bar import printProgressBar
from training_project.utils.save_tensor_img import tensor2im
from training_project.utils.util import get_duration_time_str


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False,
                 train_from_hgf=False,
                 only_finetune_decoder=False,
                 training_opt=None
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        if not train_from_hgf:
            ddconfig["in_channels"] = 1
            ddconfig["out_ch"] = 1
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.finetune_decoder = only_finetune_decoder
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if train_from_hgf:
            self.init_from_hgf()
        self.init_train_opt(training_opt)
        self.automatic_optimization = False

    def init_train_opt(self, training_opt):
        self.fold_K = training_opt.fold_K
        self.fold_idx = training_opt.fold_idx
        self.num_workers = training_opt.num_workers
        self.train_batch_size = training_opt.train_batch_size
        self.val_batch_size = training_opt.val_batch_size
        self.keys = training_opt.train_keys
        self.random_state = training_opt.seed
        self.random_prob = training_opt.augmentation_prob
        self.data_dir = training_opt.h5_2d_img_dir
        self.train_dir = os.path.join(self.data_dir, "images_tr")
        self.test_dir = os.path.join(self.data_dir, "images_ts")
        self.template_dir = training_opt.filepath_img
        self.result_dir = training_opt.root_dir
        self.pred_result_dir = os.path.join(self.result_dir, "pred_nii")
        self.dataset_type = training_opt.dataset_type
        self.learning_rate = training_opt.lr
        self.max_epochs = training_opt.num_epochs
        self.max_steps = training_opt.num_steps
        self.min_lr = training_opt.lr_low
        self.best_rec_loss = 1000
        self.best_val_epoch = 0

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_from_hgf(self):
        hugging_face_model = hugging_face_autoencoder.from_single_file(
            sys.argv[0].split("/newnas")[0] + "/newnas/MJY_file/vae-ft-mse-840000-ema-pruned.ckpt",
            config_file="../../configs/v1-inference.yaml")
        self.encoder = hugging_face_model.encoder
        self.decoder = hugging_face_model.decoder
        self.quant_conv = hugging_face_model.quant_conv
        self.post_quant_conv = hugging_face_model.post_quant_conv

        print(f"Restored from vae-ft-mse-840000-ema-pruned")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if self.finetune_decoder:
            z = z.detach()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        keys_len = len(self.keys)
        # 随机从self.keys中取两个
        # now_key = random.sample(self.keys, 2)
        # 固定取顺序前后两个
        # now_key = [self.keys[self.global_step % keys_len], self.keys[(self.global_step + 1) % keys_len]]
        # following_input = [batch[l] for l in now_key]
        # x = torch.cat([x] + following_input)
        following_input = [batch[l] for l in self.keys]
        # x = torch.cat([x] + following_input)
        if x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_metric"] = self.best_rec_loss
        checkpoint["best_val_epoch"] = self.best_val_epoch

    def on_load_checkpoint(self, checkpoint):
        self.best_rec_loss = checkpoint["best_metric"]
        self.best_val_epoch = checkpoint["best_val_epoch"]

    def on_train_epoch_start(self) -> None:
        self.s_time = time.time()
        print("Epoch ", self.current_epoch)
    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)

        reconstructions, posterior = self(inputs)
        optimizer_0, optimizer_1 = self.optimizers()
        # =======optimizer_0  vae==========
        self.toggle_optimizer(optimizer_0)
        optimizer_idx = 0
        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        optimizer_0.zero_grad()
        self.manual_backward(aeloss)
        optimizer_0.step()
        self.untoggle_optimizer(optimizer_0)
        # =======optimizer_1  disc==========
        self.toggle_optimizer(optimizer_1)
        optimizer_idx = 1
        # train the discriminator
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        optimizer_1.zero_grad()
        self.manual_backward(discloss)
        optimizer_1.step()
        self.untoggle_optimizer(optimizer_1)
        self.e_time = time.time()
        print_content = "{} / {} aeloss={}; discloss={} || Epoch {}  {}".format(batch_idx + 1,
                                                                                len(self.train_dataloader()),
                                                                                aeloss,
                                                                                discloss,
                                                                                self.current_epoch,
                                                                                get_duration_time_str(
                                                                                    s_time=self.s_time,
                                                                                    e_time=self.e_time))
        printProgressBar(batch_idx, len(self.train_dataloader()) - 1, content=print_content)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

    def on_train_epoch_end(self) -> None:

        print("lr: ", self.optimizers()[0].optimizer.param_groups[0]['lr'])

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        # with self.ema_scope():
        #     log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        log_image = self.log_images(batch)
        if batch_idx < 2:
            for b in range(int(log_image['inputs'].shape[0])):
                _input = log_image['inputs'][b:b + 1, ...]
                _output = log_image["samples"][b:b + 1, ...]
                _rec_img = log_image['reconstructions'][b:b + 1, ...]
                _input = tensor2im(_input)
                _output = tensor2im(_output)
                _rec_img = tensor2im(_rec_img)
                self.logger.experiment.add_image("sample_img", _output, self.global_step + batch_idx * b + b,
                                                 dataformats="HWC")
                self.logger.experiment.add_image("reconstruction_img", _rec_img, self.global_step + batch_idx + b,
                                                 dataformats="HWC")
                self.logger.experiment.add_image("real_img", _input, self.global_step + batch_idx + b,
                                                 dataformats="HWC")
        rec_loss = log_dict[0]["val/rec_loss"].item()
        print_content = "{} / {} rec_loss: {} ||".format(batch_idx + 1,
                                               len(self.val_dataloader()),
                                               rec_loss)
        printProgressBar(batch_idx, len(self.val_dataloader()) - 1, content=print_content)
        return log_dict

    def on_validation_end(self) -> None:
        print("validation end")

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val" + postfix)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + postfix)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.ae_loss_list.append(log_dict_ae)
        return log_dict_ae,log_dict_disc

    def on_validation_epoch_start(self) -> None:
        self.ae_loss_list = []
    def on_validation_epoch_end(self) -> None:
        result_dict = {}
        for ae_loss_dict in self.ae_loss_list:
            for k,v in ae_loss_dict.items():
                result_dict[k] = result_dict.get(k, 0) + v.item()
        if result_dict["val/rec_loss"] < self.best_rec_loss:
            self.best_rec_loss = result_dict["val/rec_loss"]
            self.best_val_epoch = self.current_epoch
        print("val result: ", result_dict)
        print("best rec_loss: ", self.best_rec_loss, " at epoch: ", self.best_val_epoch)

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
            self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        scheduler = [
            {
                'scheduler': CosineAnnealingLR(opt_ae, self.max_steps//2, eta_min=self.min_lr),
                'interval': 'step',
            }, {
                'scheduler': CosineAnnealingLR(opt_disc, self.max_steps//2, eta_min=self.min_lr),
                'interval': 'step',
            }]
        return [opt_ae, opt_disc], scheduler

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        # colorize with random projection
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"] = self.decode(torch.randn_like(posterior_ema.sample()))
                    log["reconstructions_ema"] = xrec_ema
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

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
        print(f'train_id:{len(train_id)}||valid_id:{len(test_id)}')
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
        id_list = sorted(os.listdir(self.test_dir))[:1]
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
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return val_loader

    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.test_ds,
            batch_size=8,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return pred_loader


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
