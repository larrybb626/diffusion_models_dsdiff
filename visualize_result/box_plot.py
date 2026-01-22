# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：box_plot.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/2/25 20:52 
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_box_plot(data, labels, metric_name, save_dir, file_name="boxplot.png", dpi=300, ):
    """
    创建并保存箱线图，用于展示多组数据的分布。

    参数:
    - data: list of arrays, 每组数据作为一个数组
    - labels: list of str, 每组数据的标签
    - save_dir: str, 保存图像的目录路径
    - file_name: str, 保存的图像文件名，默认为 "boxplot.png"
    - dpi: int, 图像分辨率，默认为 300
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")

    # 设置 Seaborn 风格和调色板
    sns.set_style("whitegrid")
    sns.set_palette("muted")
    """
    "deep"：深色调
    "muted"：柔和色调
    "pastel"：柔和的浅色
    "bright"：鲜艳色调
    "dark"：深色
    custom_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD", "#D4A5A5", "#9B59B6", "#3498DB", "#E74C3C", "#2ECC71"]
    """

    # 创建图像
    plt.figure(figsize=(10, 8))

    # 将数据转换为 DataFrame，方便 Seaborn 处理
    df = pd.DataFrame({label: pd.Series(group) for label, group in zip(labels, data)})
    sns.boxplot(data=df,width=0.7,
                # flierprops=dict(marker='o', color='red', alpha=0.5),  # 异常值颜色
                # medianprops=dict(color='black', linewidth=2),  # 中位数线颜色
                # boxprops=dict(edgecolor='darkblue', linewidth=1.5)  # 箱体边框颜色
                )

    # plt.yscale('log')  # 使用对数坐标轴
    # 添加标题和轴标签（根据附件数据定制）
    # plt.title(f"Boxplot of {metric_name} Across Models for Prostate", fontsize=16)
    plt.xlabel("Models", fontsize=20)
    # plt.ylim(0.5, 0.2)
    plt.ylabel(metric_name, fontsize=20)
    plt.xticks(fontsize=20,rotation=30, ha="right")  # 旋转标签以避免重叠
    plt.yticks(fontsize=20)
    # 保存图像到指定路径
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"图像已保存到: {save_path}")

    # 显示图像
    plt.show()


if __name__ == "__main__":
    # 定义保存路径
    plot_save_dir = "/home/user15/sharedata/newnas_1/MJY_file/visualize_all/box_plot"
    whole_dir = "/home/user15/sharedata/newnas_1/MJY_file/diffusion/train_result"
    metric_name = ["MS-SSIM","PSNR","NRMSE"]
    # 示例：使用附件表格中的 NRMSE 数据
    # 这里假设每组数据是多个实验的分布，实际应用中需提供完整数据集
    labels = [
        "DDPM+DDIM", "DDPM+DPM-Solver",
        "LDM-oVAE+DDIM","LDM-oVAE+DPM-Solver",
        "LDM-sVAE+DDIM", "LDM-sVAE+DPM-Solver",
        "DiT-oVAE+DDIM",
        # "DiT-oVAE+DPM-Solver",
        "DiT-sVAE+DDIM","DiT-sVAE+DPM-Solver",
        "DiT-nVAE+DDIM","DiT-nVAE+DPM-Solver"
    ]
    exp_name = [
        ("94_ds_diff", "ddim"),
        ("94_ds_diff", "dpm"),
        ("13_ldm", "ddim"),
        ("13_ldm", "dpm"),
        ("19_ldm", "ddim"),
        ("19_ldm", "dpm"),
        ("29_dit", "ddim"),
        # ("29_dit", "dpm"),
        ("18_dit", "ddim"),
        ("18_dit", "dpm"),
        ("108_ds_diff", "ddim"),
        ("108_ds_diff", "dpm"),
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
    for mcn in metric_name:
        data = [pd.read_excel(excel_dir)[mcn.split("-")[-1].lower()].values[1:] for excel_dir in excel_dirs]
        # 调用函数生成箱线图
        create_box_plot(
            data,
            labels,
            mcn,
            plot_save_dir,
            file_name=f"boxplot_{mcn}_prostate.png"
        )
