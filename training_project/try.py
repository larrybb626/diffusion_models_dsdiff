# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：try.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/4/16 1:15 
"""
import os
import sys

import h5py
import numpy as np
import torch
import tqdm

from Disc_diff.guided_diffusion.image_datasets import dataset_config

# sys.path.append('/home/user15/sharedata/newnas/MJY_file/code/diffusion_model')
# from trainers.trainer_diffusion import DiffusionModel

# from training_project.utils.My_callback import CheckPointSavingMetric
from segmentation_models_pytorch import Unet

import re
import diffusers

import threading

def cpu_stress():
    while True:
        pass  # 无限循环，占用 CPU

# 创建与 CPU 核心数相同的线程
num_threads = 8  # 根据你的 CPU 核心数调整
threads = []

for _ in range(num_threads):
    thread = threading.Thread(target=cpu_stress)
    thread.start()
    threads.append(thread)

# 等待线程执行
for thread in threads:
    thread.join()


