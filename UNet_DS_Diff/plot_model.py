# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：plot_model.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/3/11 10:29 
"""
import argparse

import torch
import yaml
from omegaconf import OmegaConf
from torchinfo import summary
from itertools import zip_longest

from Disc_diff.guided_diffusion.script_util import add_dict_to_argparser, sr_create_model_and_diffusion
from ldm.models.diffusion.ddpm import DiffusionWrapper
# 这里替换为你的模型导入方式
from model import DSUnetModel
from Disc_diff.guided_diffusion.unet import UNet_disc_Model


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to YAML configuration file",
                    default="../Disc_diff/config/config_train_prostate.yaml")
args = parser.parse_args()

# Load the configuration from the YAML file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)
# Add the configuration values to the argument parser
add_dict_to_argparser(parser, config)
# 创建模型实例
config_1 = parser.parse_args()
config = OmegaConf.load("../configs/v2-1-cddpm-ds-disc-openai-diffusion.yaml").model.params
config_2 = config.unet_config
model1 = DiffusionWrapper(config_2, config.conditioning_key).diffusion_model
model2,_ = sr_create_model_and_diffusion(config_1)

# 获取模型结构字符串
summary1 = str(model1).split("\n")
summary2 = str(model2).split("\n")

# 计算最长行长度，使其对齐
max_len_1 = max(len(line) for line in summary1)
summary1_lines = [line.ljust(max_len_1) for line in summary1]

# 提取参数信息
params1 = [f"{name}: {param.numel()} params" for name, param in model1.named_parameters()]
params2 = [f"{name}: {param.numel()} params" for name, param in model2.named_parameters()]

# 计算参数部分的最长长度
max_len_2 = max(len(line) for line in params1) if params1 else 0
params1_lines = [line.ljust(max_len_2) for line in params1]
params2_lines = params2

# 组合结构信息
combined_lines = [f"{a}    {b}" for a, b in zip_longest(summary1_lines, summary2, fillvalue=" ")]

# 组合参数信息
combined_params = [f"{a}    {b}" for a, b in zip_longest(params1_lines, params2_lines, fillvalue=" ")]

# 保存到文件
with open("/home/user15/sharedata/newnas_1/MJY_file/comparison.txt", "w") as f:
    f.write("模型结构对比:\n")
    f.write("\n".join(combined_lines))
    f.write("\n\n模型参数对比:\n")
    f.write("\n".join(combined_params))

print("模型结构与参数对比已保存至 comparison.txt")