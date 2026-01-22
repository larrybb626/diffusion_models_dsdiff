# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：disc_diff_trainer.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/5/7 17:09 
"""
from Disc_diff.guided_diffusion.script_util import sr_create_model_and_diffusion
from Disc_diff.guided_diffusion.resample import create_named_schedule_sampler
from omegaconf import OmegaConf
from Disc_diff.scripts.super_res_train import create_argparser
if __name__ == '__main__':
    args = create_argparser().parse_args()
    model, diffusion = sr_create_model_and_diffusion(args)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)