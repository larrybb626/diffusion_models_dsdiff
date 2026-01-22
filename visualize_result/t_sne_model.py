# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：t_sne_model.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/3/11 12:44 
"""
import os
import re
import sys

import torch
from lightning import seed_everything
from monai.utils import set_determinism
from omegaconf import OmegaConf

from trainers.trainer_use_gaussian_diff import TryTrainerDiffusion

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    config = OmegaConf.load("../configs/inference_config.yaml")
    torch.multiprocessing.set_sharing_strategy('file_system')
    set_determinism(config["seed"])
    seed_everything(config["seed"], workers=True)
    # ==========path============
    Task_name = config.Task_name
    task_id = config.Task_id
    fold_idx = config.fold_idx
    ckpt_name = config.ckpt_name
    dir_prefix = sys.argv[0].split("/newnas")[0]
    config.result_path = os.path.join(dir_prefix, config.result_path)
    # ===============model setting==============
    task_name = "{}_{}_{}_fold5-{}".format(Task_name, task_id, config.net_mode, fold_idx)
    result_path = config.result_path
    # ================search for best==============================
    ckpt_dir = os.path.join(result_path, task_name, "checkpoint")
    ckpt_list = os.listdir(ckpt_dir)
    if ckpt_name == "best":
        pattern = r"{}(-epoch=\d+)?\.ckpt".format(ckpt_name)
        ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
        versions = [re.search(r"epoch=(\d+)", file).group(1) for file in ckpt_file if re.search(r"epoch=\d+", file)]
        sorted_versions = sorted(versions, key=lambda x: int(x))
        ckpt_to_resume = f"{ckpt_name.split('.')[0]}-epoch={sorted_versions[-1]}.ckpt" if sorted_versions else ckpt_name
    else:
        pattern = r"{}(-v\d+)?\.ckpt".format(ckpt_name)
        ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
        versions = [re.search(r"v(\d+)", file).group(1) for file in ckpt_file if re.search(r"v\d+", file)]
        sorted_versions = sorted(versions, key=lambda x: int(x))
        ckpt_to_resume = f"{ckpt_name}-v{sorted_versions[-1]}.ckpt" if sorted_versions else ckpt_name + ".ckpt"

    ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
    print(ckpt_to_resume)
    model = TryTrainerDiffusion.load_from_checkpoint(ckpt_path,
                                                     map_location="cuda:{}".format(int(config.cuda_idx))
                                                     )
    dataloader = model.predict_dataloader()
    model = model.model
    
    for batch in dataloader:
        path, images, labels = (batch["path"], batch["image"], batch["t1ce"])
        noise = torch.randn(*images.shape, device='cuda:0')
        images = images.to('cuda:0')
        t = 0
        x, y = batch
        y_hat = model(x)
        print(y_hat.shape)
        break
