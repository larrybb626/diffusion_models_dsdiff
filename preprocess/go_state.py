# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：go_state.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/3/15 18:34 
"""
import os

import pandas as pd
if __name__ == "__main__":
    txt_path = os.path.join("/home/user15/sharedata/newnas/MJY_file/CE-MRI", "peizhun_error.txt")
    with open(txt_path, "r") as f:
        # 读取全部内容
        file_contents = f.read()
    bad_list = file_contents.split("\n")
    whole_dir = "/home/user15/sharedata/newnas_1/MJY_file/diffusion/train_result"
    metric_name = ["MS-SSIM", "PSNR", "NRMSE"]
    # 示例：使用附件表格中的 NRMSE 数据
    # 这里假设每组数据是多个实验的分布，实际应用中需提供完整数据集
    labels = [
        "DDPM+DDIM", "DDPM+DPM-Solver",
        "LDM-oVAE+DDIM", "LDM-oVAE+DPM-Solver",
        "LDM-sVAE+DDIM", "LDM-sVAE+DPM-Solver",
        "DiT-oVAE+DDIM",
        "DiT-oVAE+DPM-Solver",
        "DiT-sVAE+DDIM", "DiT-sVAE+DPM-Solver",
        "DiT-nVAE+DDIM", "DiT-nVAE+DPM-Solver"
    ]
    exp_name = [
        ("94_ds_diff", "ddim"),
        ("94_ds_diff", "dpm"),
        ("13_ldm", "ddim"),
        ("13_ldm", "dpm"),
        ("19_ldm", "ddim"),
        ("19_ldm", "dpm"),
        ("29_dit", "ddim"),
        ("29_dit", "dpm"),
        ("18_dit", "ddim"),
        ("18_dit", "dpm"),
        ("108_ds_diff", "ddim"),
        ("108_ds_diff", "dpm"),
    ]
    labels = [
        "DDPM",
        "w/ MS-UNet",
        "w/ SADM",
        "w/ $L_{dis}$ λ=0.1",
        "w/ $L_{dis}$ λ=0.5",
        "w/ $L_{dis}$ λ=1",
        "w/ $L_{dis}^{contrast}$ λ=0.1",
        "w/ $L_{dis}^{contrast}$ λ=0.5",
        "w/ $L_{dis}^{contrast}$ λ=1",
        "w/ Cross-Attn",
        "w/ H-Cross-Attn",
    ]
    exp_name = [
        ("94_ds_diff", "ddim"),
        ("55_ds_diff", "ddim"),
        ("49_ds_diff", "ddim"),
        ("61_ds_diff", "ddim"),
        ("139_ds_diff", "ddim"),
        ("59_ds_diff", "ddim"),
        ("142_ds_diff", "ddim"),
        ("140_ds_diff", "ddim"),
        ("143_ds_diff", "ddim"),
        ("85_ds_diff", "ddim"),
        ("95_ds_diff", "ddim"),
    ]
    labels = [
        "w/o SG",
        "Sobel",
        "Canny",
        "Laplacian",
        "SFG",
        "Sobel+SFG",
    ]
    exp_name = [
        ("139_ds_diff", "ddim"),
        ("144_ds_diff", "ddim"),
        ("105_ds_diff", "ddim"),
        ("145_ds_diff", "ddim"),
        ("134_ds_diff", "ddim"),
        ("156_ds_diff", "ddim"),
    ]
    excel_dirs = []
    for i, sample in exp_name:
        if "dit" not in i:
            excel_dirs.append(
                whole_dir + "/CE_MRI_synthesis_{}_fold5-1/pred_nii_{}_20_eta0_checkpoint_metric.xlsx".format(i, sample)
            )
        else:
            excel_dirs.append(
                os.path.dirname(os.path.dirname(
                    whole_dir)) + "/SOTA_models/DiT_test/0{}-DiT-XL-2/{}20/itk_result/_metric.xlsx".format(
                    i.split("_")[0], sample)
            )
        if not os.path.exists(excel_dirs[-1]):
            raise FileNotFoundError("File not found: {}".format(excel_dirs[-1]))
    labels = ['cGAN', 'ResViT', 'DisC-Diff', 'SD3', 'DS-Diff']
    excel_dirs = [
    '/home/user15/sharedata/newnas_1/MJY_file/SOTA_models/mulD_RegGAN/CE_MRI_simulate_PCa_63_pix2pix_mulD_fold5-1/pred_nii_metric.xlsx',
    '/home/user15/sharedata/newnas_1/MJY_file/SOTA_models/SOTA_GAN/ResVit_all_Seq_noval_new_pretrain/result/_metric.xlsx',
    '/home/user15/sharedata/newnas_1/MJY_file/SOTA_models/DiscDiff_test/Prostate_v_predict_7e4/itk_save_dir/_metric.xlsx',
    '/home/user15/sharedata/newnas_1/MJY_file/SOTA_models/controlnet-model/controlnet_result/checkpoint-50000/itk_result/_metric.xlsx',
    '/home/user15/sharedata/newnas_1/MJY_file/diffusion/train_result/CE_MRI_synthesis_154_ds_diff_fold5-1/pred_nii_ddim_20_eta0_checkpoint_metric.xlsx',
]
    for mcn in metric_name:
        data = []
        for excel_dir in excel_dirs:
            dp = pd.read_excel(excel_dir)[mcn.split("-")[-1].lower()].values[1:]
        data.append(dp)
    all_ssim = pd.DataFrame()
    all_psnr = pd.DataFrame()
    all_nrmse = pd.DataFrame()
    for excel_dir, label in zip(excel_dirs, labels):
        ms_ssim = pd.read_excel(excel_dir)["ssim"].values[1:]
        psnr = pd.read_excel(excel_dir)["psnr"].values[1:]
        nrmse = pd.read_excel(excel_dir)["nrmse"].values[1:]
        all_ssim[label] = ms_ssim
        all_psnr[label] = psnr
        all_nrmse[label] = nrmse
    with pd.ExcelWriter("/home/user15/sharedata/newnas_1/MJY_file/statistic analysis/4_2_all_metric.xlsx") as writer:
        all_ssim.to_excel(writer, sheet_name="MS-SSIM", index=False)
        all_psnr.to_excel(writer, sheet_name="PSNR", index=False)
        all_nrmse.to_excel(writer, sheet_name="NRMSE", index=False)


