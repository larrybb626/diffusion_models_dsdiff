# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models
@File    ：train_main_with_gaussian_diff.py
@IDE     ：PyCharm
@Author  ：MJY
@Date    ：2024/11/4 11:18

"""
import os
import sys
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)

# 如果脚本在子文件夹(如 training_project)里，可能需要多取一层 os.path.dirname
# 如果脚本已经在根目录，project_root 就是根目录
if project_root not in sys.path:
    sys.path.append(project_root)

# 打印一下确认路径，方便排查
print(f"==== 项目根目录已锁定为: {project_root} ====")

import shutil

from trainers.trainer_use_gaussian_diff import TryTrainerDiffusion
import lightning.pytorch as pl
import lightning as l
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import AdvancedProfiler
from monai.utils import set_determinism
from omegaconf import OmegaConf
from core.JY_Network import JunyangFramework
from configs.train_config import config
from trainers.trainer_ds_diff import DSDiffModel

import re


# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # print("Please enter the password:")
    # password = input()
    Junyang = JunyangFramework("Junyang is the best!")  # enter your password
    config = Junyang.get_config(OmegaConf.load(config.config_file))
    assert config.Task_name == "PET_synthesis"
    # torch.multiprocessing.set_sharing_strategy('file_system')
    set_determinism(config.seed)
    seed_everything(config.seed, workers=True)
    # 设置好路径
    dir_prefix = sys.argv[0].split("/nas")[0]  # change 现在没有newnas了，改成nas吧
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
        monitor='val_loss',
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
    # ================================== config opt ====================================================================
    config_opt = OmegaConf.load('/nas_3/LaiRuiBin/Changhai/code/dsfr_diffusion/configs/v2-1-cddpm-ds-disc-openai-diffusion.yaml')["model"]["params"]
    input_channels = min((len(config.train_keys) + 1) ,config_opt.unet_config.params.in_channels)
    output_channels = 1
    config_opt["unet_config"]["params"]["in_channels"] = input_channels
    config_opt["unet_config"]["params"]["out_channels"] = output_channels
    # config.embedder_config = config_opt["model"]["params"].pop("embedder_config",None)
    # =================================initialise Lightning's trainer.======================================================
    profiler = AdvancedProfiler(dirpath=root_dir, filename="perf_logs")
    trainer = pl.Trainer(
        # default_root_dir=root_dir,
        # strategy="deepspeed",
        accelerator='gpu',
        devices=list(config.cuda_idx),
        max_epochs=config.num_epochs,
        check_val_every_n_epoch=config.val_step,
        logger=tb_logger,
        enable_checkpointing=True,
        log_every_n_steps=1,
        callbacks=[best_callback, checkpoint_callback],
        deterministic="warn",
        enable_progress_bar=False,
        # =====dev option=====
        # precision="bf16-mixed",
        num_sanity_val_steps=0,
        # fast_dev_run=1,
        # limit_train_batches=1,
        limit_val_batches=8,
        # limit_train_batches=300,
        # profiler=profiler,
    )
    # ===================configure net===================================
    # 合并两个config字典
    config = OmegaConf.merge(config, config_opt)

    # unet = Junyang.get_model(TryTrainerDiffusion(config))
    # unet = torch.compile(unet)
    # ========================search for ckpt==============================
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_list = os.listdir(ckpt_dir)
    pattern = r"checkpoint(-v\d+)?\.ckpt"
    ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]

    print("Save PyBackup model.py")
    model_dir = config.unet_config.target
    model_dir = "../" + "/".join(model_dir.split(".")[:-1]) + '.py'
    # shutil.copy(model_dir, os.path.join(config.root_dir, "model.py"))
    # 确保使用我们在脚本开头定义的 PROJECT_ROOT
    model_dir = os.path.join(project_root, "UNet_DS_Diff", "model.py")
    # 新训练
    if not ckpt_file:
        print("========== No checkpoint to resume, start a new train ==========")
        unet = Junyang.get_model(TryTrainerDiffusion(config))
        trainer.fit(unet)
    # 断点恢复
    # else:  # 俊杨原来的代码
    #     versions = [re.search(r"v(\d+)", file).group(1) for file in ckpt_file if re.search(r"v\d+", file)]
    #     sorted_versions = sorted(versions, key=lambda x: int(x))
    #     ckpt_to_resume = f"checkpoint-v{sorted_versions[-1]}.ckpt" if sorted_versions else "checkpoint.ckpt"
    #     ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
    #     hyper_parameters = torch.load(ckpt_path)["hyper_parameters"]
    #     unet = Junyang.get_model(TryTrainerDiffusion(**hyper_parameters))
    #     trainer.fit(unet, ckpt_path=ckpt_path)
    else:
        versions = [re.search(r"v(\d+)", file).group(1) for file in ckpt_file if re.search(r"v\d+", file)]
        sorted_versions = sorted(versions, key=lambda x: int(x))
        ckpt_to_resume = f"checkpoint-v{sorted_versions[-1]}.ckpt" if sorted_versions else "checkpoint.ckpt"
        if not os.path.exists(os.path.join(ckpt_dir, ckpt_to_resume)):  # 因为我后面更改了保存模型和命名规则，每2个epoch保存一次，所以这里再加一个判断
            pattern_new = r"checkpoint-epoch=(\d+)\.ckpt"
            epoch_ckpts = []
            for f in ckpt_list:
                m = re.match(pattern_new, f)
                if m:
                    epoch_ckpts.append((int(m.group(1)), f))
            if epoch_ckpts:
                # 按 epoch 排序，取最大的
                epoch_ckpts.sort(key=lambda x: x[0])
                ckpt_to_resume = epoch_ckpts[-1][1]
        ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
        hyper_parameters = torch.load(ckpt_path)["hyper_parameters"]
        unet = Junyang.get_model(TryTrainerDiffusion(**hyper_parameters))
        trainer.fit(unet, ckpt_path=ckpt_path)



