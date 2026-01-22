import os
import re
import sys

import lightning.pytorch as pl
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.profilers import AdvancedProfiler
from monai.utils import set_determinism
from omegaconf import OmegaConf

from configs.train_config import config
from trainers.trainer_latent_diffusion import LatentDiffusionModel

if __name__ == "__main__":
    config = OmegaConf.load(config.config_file)
    torch.multiprocessing.set_sharing_strategy('file_system')
    set_determinism(config.seed)
    seed_everything(config.seed, workers=True)
    # 设置好路径
    dir_prefix = sys.argv[0].split("/newnas")[0]
    config.filepath_img = os.path.join(dir_prefix, config.filepath_img)
    config.h5_2d_img_dir = os.path.join(dir_prefix, config.h5_2d_img_dir)
    config.result_path = os.path.join(dir_prefix, config.result_path)
    # 设置任务名和对应的路径
    # CE_MRI_simulate_1_2d_fold5-1
    task_name = config.Task_name + '_' + str(config.Task_id) + '_' + config.net_mode + '_fold' + str(
        config.fold_K) + "-" + str(config.fold_idx)
    print("===================={}=====================".format(task_name))
    root_dir = os.path.join(config.result_path, task_name)
    config.root_dir = os.path.join(config.result_path, task_name)
    # config.record_file = os.path.join(config.root_dir, "log_txt.txt")
    # ===============================set up GPU======================================================
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    # =====================================set up loggers and checkpoints======================================================
    log_dir = os.path.join(root_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)
    # tensorboard --logdir = log_dir
    # ===================================callback======================================================
    ckpt_dir = os.path.join(root_dir, "checkpoint")
    loss_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="val_loss_best",
        monitor='val/loss',
        mode="min",
        save_last=False,
        save_top_k=1,
        save_weights_only=True,
    )
    best_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch}",
        monitor='val/ssim',
        mode="max",
        save_last=False,
        save_top_k=1,
        save_weights_only=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="checkpoint",
        every_n_epochs=config.checkpoint_epoch,
        save_on_train_epoch_end=True
    )
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )
    # ================================== config opt ====================================================================
    config_opt = OmegaConf.load("../configs/v2-1-stable-unclip-h-inference.yaml")["model"]["params"]
    # config.embedder_config = config_opt["model"]["params"].pop("embedder_config",None)
    # =================================initialise Lightning's trainer.==================================================
    profiler = AdvancedProfiler(dirpath=root_dir, filename="perf_logs")
    trainer = pl.Trainer(
        # default_root_dir=root_dir,
        accelerator='gpu',
        devices=[config.cuda_idx],
        max_epochs=config.num_epochs,
        check_val_every_n_epoch=config.val_step,
        logger=tb_logger,
        enable_checkpointing=True,
        log_every_n_steps=1,
        callbacks=[best_callback, checkpoint_callback],
        deterministic="warn",
        enable_progress_bar=False,
        # =====dev option=====
        num_sanity_val_steps=0,
        # fast_dev_run=1,
        # limit_train_batches=1,
        limit_val_batches=8,
        # limit_train_batches=300,
        # profiler=profiler,
    )
    # ===================configure net===================================
    def configurate_vae(net):
        if config.vae_local_pretrained:
            print("Loading VAE Local Pretrained")
            vae_ckpt = os.path.join(dir_prefix,"newnas/MJY_file/CE-MRI/train_result", "VAE_4_fold5-1", "checkpoint", "checkpoint-v1.ckpt")
            net.instantiate_first_stage(config=None, ckpt=vae_ckpt)
        print("Network initiate finished")
    # ========================search for ckpt==============================
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_list = os.listdir(ckpt_dir)
    pattern = r"checkpoint(-v\d+)?\.ckpt"
    ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
    # 新训练
    if not ckpt_file:
        print("========== No checkpoint to resume, start a new train ==========")

        # print("Loading v2-1 ckpt")
        # state_dict = torch.load(
        #     "/home/user15/sharedata/newnas/MJY_file/CE-MRI/stable_diffusion_2-1_unclip/sd21-unclip-h.ckpt")
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if "model.diffusion_model" in k:
        #         new_state_dict.update({k: v})
        # unet.load_state_dict(new_state_dict, strict=False)
        unet = LatentDiffusionModel(config, **config_opt)
        configurate_vae(unet)
        trainer.fit(unet)
    # 断点恢复
    else:
        versions = [re.search(r"v(\d+)", file).group(1) for file in ckpt_file if re.search(r"v\d+", file)]
        sorted_versions = sorted(versions, key=lambda x: int(x))
        ckpt_to_resume = f"checkpoint-v{sorted_versions[-1]}.ckpt" if sorted_versions else "checkpoint.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
        hyper_parameters = torch.load(ckpt_path)["hyper_parameters"]
        unet = LatentDiffusionModel(**hyper_parameters)
        configurate_vae(unet)
        trainer.fit(unet, ckpt_path=ckpt_path)
