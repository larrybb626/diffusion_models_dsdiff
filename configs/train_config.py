# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：train_config.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/3/11 21:45 
"""
import argparse
import yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # 嵌套字典也转换为对象
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


parser = argparse.ArgumentParser()
# =============================偶尔改的参数=============================
# dataset_type
parser.add_argument('--config_file', type=str,
                    default=r"/nas_3/LaiRuiBin/Changhai/code/dsfr_diffusion/configs/train_config.yaml")
# # result&save
# parser.add_argument('--dir_prefix', type=str, default=r'/home/user15/sharedata/')
# parser.add_argument('--dir_prefix', type=str, default=r'/data/newnas_1/LiuWenxi/Changhai/results/2025_0417/logs')

# parser.add_argument('--result_path', type=str, default=r'newnas1/MJY_file/diffusion/train_result/')  # 结果保存地址
# parser.add_argument('--result_path', type=str, default=r'/data/newnas_1/LiuWenxi/Changhai/results/2025_0417/result')  # 结果保存地址
#
# parser.add_argument('--filepath_img', type=str,default=r'newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm')
# parser.add_argument('--h5_2d_img_dir', type=str, default=r'newnas/MJY_file/CE-MRI/PCa_new/h5_data_2d_pre_320320_01norm')

args = parser.parse_args()

# model hyper-parameters
# 加载 YAML 文件
try:
    with open(args.config_file, 'r', encoding='utf-8') as file:
        yaml_config = yaml.safe_load(file)  # 加载 YAML 文件内容
except FileNotFoundError:
    raise FileNotFoundError(f"YAML 配置文件未找到: {args.config_file}")
except yaml.YAMLError as e:
    raise ValueError(f"YAML 文件解析错误: {e}")

# 将 YAML 配置转换为对象
yaml_config_obj = Config(yaml_config)

# 将命令行参数转换为对象
args_dict = vars(args)  # 将 argparse.Namespace 转换为字典
args_config_obj = Config(args_dict)

# 合并 YAML 配置和命令行参数
class MergedConfig:
    def __init__(self, yaml_config, args_config):
        # 先加载 YAML 配置
        for key, value in yaml_config.__dict__.items():
            setattr(self, key, value)
        # 再加载命令行参数（覆盖 YAML 配置）
        for key, value in args_config.__dict__.items():
            setattr(self, key, value)

# 创建最终的 config 对象
config = MergedConfig(yaml_config_obj, args_config_obj)
