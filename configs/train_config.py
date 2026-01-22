# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：train_config.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/3/11 21:45 
"""
import argparse

parser = argparse.ArgumentParser()
# =============================偶尔改的参数=============================
# dataset_type
parser.add_argument('--config_file', type=str, default="../configs/train_config.yaml")
# # result&save
parser.add_argument('--dir_prefix', type=str, default=r'/home/user15/sharedata/')
parser.add_argument('--result_path', type=str, default=r'newnas1/MJY_file/diffusion/train_result/')  # 结果保存地址
parser.add_argument('--filepath_img', type=str,
                    default=r'newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm')
parser.add_argument('--h5_2d_img_dir', type=str, default=r'newnas/MJY_file/CE-MRI/PCa_new/h5_data_2d_pre_320320_01norm')
# model hyper-parameters
config = parser.parse_args()
