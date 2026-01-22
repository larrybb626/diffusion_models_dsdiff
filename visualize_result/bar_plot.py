# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：scatter_and_line.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/2/28 22:48 
"""
# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：scatter_and_line.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/2/28 22:48 
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
# 数据准备
# 数据准备

data_prostate = {
    'DDPM': {'NRMSE': 0.0825, 'PSNR': 22.18, 'MS-SSIM': 0.8084},
    'w/ MS-UNet': {'NRMSE': 0.0798, 'PSNR': 22.50, 'MS-SSIM': 0.8142},
    'w/ SADM': {'NRMSE': 0.0794, 'PSNR': 22.52, 'MS-SSIM': 0.8170},
}

data_brats = {
    'DDPM': {'NRMSE': 0.0450, 'PSNR': 27.39, 'MS-SSIM': 0.8914},
    'w/ MS-UNet': {'NRMSE': 0.0449, 'PSNR': 27.38, 'MS-SSIM': 0.8936},
    'w/ SADM': {'NRMSE': 0.0438, 'PSNR': 27.49, 'MS-SSIM': 0.8951},
}

data_prostate = {
    r'DFA': {'NRMSE': 0.0787, 'PSNR': 22.63, 'MS-SSIM': 0.8249},
    'Cross-Attention': {'NRMSE': 0.0792, 'PSNR': 22.56, 'MS-SSIM': 0.8222},
    'H-Cross-Attention': {'NRMSE': 0.0789, 'PSNR': 22.57, 'MS-SSIM': 0.8197}
}

data_brats = {
    r'DFA': {'NRMSE': 0.0429, 'PSNR': 27.70, 'MS-SSIM': 0.9008},
    'Cross-Attention': {'NRMSE': 0.0452, 'PSNR': 27.22, 'MS-SSIM': 0.8900},
    'H-Cross-Attention': {'NRMSE': 0.0448, 'PSNR': 27.31, 'MS-SSIM': 0.8936}
}

# #
data_prostate = {
    'cGAN': {'NRMSE': 0.1090, 'PSNR': 20.63, 'MS-SSIM': 0.8055},
    'ResViT': {'NRMSE': 0.0782, 'PSNR': 22.64, 'MS-SSIM': 0.8223},
    'DisC-Diff': {'NRMSE': 0.0781, 'PSNR': 22.68, 'MS-SSIM': 0.8237},
    'SD3': {'NRMSE': 0.0849, 'PSNR': 21.88, 'MS-SSIM': 0.7681},
    'DS-Diff': {'NRMSE': 0.0775, 'PSNR': 22.74, 'MS-SSIM': 0.8303}
}

data_brats = {
    'cGAN': {'NRMSE': 0.0565, 'PSNR': 25.85, 'MS-SSIM': 0.8726},
    'ResViT': {'NRMSE': 0.0453, 'PSNR': 27.22, 'MS-SSIM': 0.8955},
    'DisC-Diff': {'NRMSE': 0.0424, 'PSNR': 27.85, 'MS-SSIM': 0.9012},
    'SD3': {'NRMSE': 0.0489, 'PSNR': 26.45, 'MS-SSIM': 0.8670},
    'DS-Diff': {'NRMSE': 0.0422, 'PSNR': 27.99, 'MS-SSIM': 0.9090}
}
# # # #
# data_prostate = {
#     'w/o SG': {'NRMSE': 0.0787, 'PSNR': 22.63, 'MS-SSIM': 0.8249},
#     'Sobel': {'NRMSE': 0.0785, 'PSNR': 22.64, 'MS-SSIM': 0.8243},
#     'Canny': {'NRMSE': 0.0797, 'PSNR': 22.50, 'MS-SSIM': 0.8181},
#     'Laplacian': {'NRMSE': 0.0805, 'PSNR': 22.44, 'MS-SSIM': 0.8109},
#     'SFG': {'NRMSE': 0.0783, 'PSNR': 22.65, 'MS-SSIM': 0.8254},
#     'Sobel+SFG': {'NRMSE': 0.0775, 'PSNR': 22.74, 'MS-SSIM': 0.8303}
# }
#
# data_brats = {
#     'w/o SG': {'NRMSE': 0.0429, 'PSNR': 27.70, 'MS-SSIM': 0.9008},
#     'Sobel': {'NRMSE': 0.0428, 'PSNR': 27.70, 'MS-SSIM': 0.9011},
#     'Canny': {'NRMSE': 0.0468, 'PSNR': 26.91, 'MS-SSIM': 0.8934},
#     'Laplacian': {'NRMSE': 0.0457, 'PSNR': 27.16, 'MS-SSIM': 0.8913},
#     'SFG': {'NRMSE': 0.0436, 'PSNR': 27.74, 'MS-SSIM': 0.9048},
#     'Sobel+SFG': {'NRMSE': 0.0422, 'PSNR': 27.99, 'MS-SSIM': 0.9090}
# }
# 设置绘图样式
plt.style.use('seaborn-whitegrid')
colors = sns.color_palette("Set2", 6)  # 为三个模型选择柔和颜色
nature_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
colors = sns.color_palette(nature_colors)
# models = [r'DFA', 'Cross-Attention', 'H-Cross-Attention']
# models = ['DDPM', 'w/ MS-UNet', 'w/ SADM']
models = ['cGAN', 'ResViT', 'DisC-Diff', 'SD3', 'DS-Diff']
# models = ['w/o SG', 'Sobel', 'Canny', 'Laplacian', 'SFG', 'Sobel+SFG']
metrics = ['NRMSE', 'PSNR', 'MS-SSIM']
bar_width = 0.6

# 辅助函数：标注柱状图数值
def autolabel(bars, ax,metric):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}' if metric == 'PSNR' else f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 数值标签向上偏移5个单位
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16)

# 辅助函数：自动调整Y轴范围
def set_ylim(ax, values):
    min_val, max_val = min(values), max(values)
    padding = (max_val - min_val) * 0.1  # 添加10%的缓冲区
    ax.set_ylim(min_val - padding, max_val + padding)

# 绘制单个数据集的图表
def plot_dataset(data, title):
    fig = plt.figure(figsize=(20, 6), dpi=120)  # 设置画布大小
    gs = GridSpec(1, 3, wspace=0.32)  # 1行3列布局，子图间距
    x = np.arange(len(models))*1.1  # 横轴位置
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 使用确认的名称
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        values = [data[model][metric] for model in models]

        # 绘制柱状图
        bars = ax.bar(x, values, bar_width, color=colors)

        # 标注数值
        autolabel(bars, ax, metric)

        # 设置坐标轴
        ax.set_xlabel('Models', fontsize=18, labelpad=14)
        ax.set_ylabel(metric + (' ↓' if metric == 'NRMSE' else ' ↑'),
                      fontsize=18, labelpad=8)
        ax.set_xticks(x)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_xticklabels(models, fontsize=18, rotation=20)
        ax.set_title(metric, fontsize=18, pad=10)

        # 自动设置Y轴范围
        set_ylim(ax, values)

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.7)

    # 添加全局标题
    # plt.suptitle(title, fontsize=18, y=1.05)

    # 保存和显示
    plt.savefig(f'/data/newnas_1/MJY_file/visualize_all/bar_chart/4_3_2_{title.lower().replace(" ", "_")}.png', bbox_inches='tight', dpi=600)
    plt.show()

# 绘制 Prostate 数据集图表
plot_dataset(data_prostate, 'Performance Comparison on Prostate Dataset')

# 绘制 BraTs 数据集图表
plot_dataset(data_brats, 'Performance Comparison on BraSyn Dataset')