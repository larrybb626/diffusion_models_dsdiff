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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ================== 数据准备 ==================
# 使用与示例代码完全一致的数据结构
data_pro = {
    "L_dis": {
        'NRMSE': [0.0796, 0.0787, 0.0803],
        'PSNR': [22.51, 22.63, 22.43],
        'MS-SSIM': [0.8160, 0.8249, 0.8133]
    },
    "L_dis_contrast": {
        'NRMSE': [0.0797, 0.0819, 0.0827],
        'PSNR': [22.51, 22.25, 22.19],
        'MS-SSIM': [0.8180, 0.8110, 0.8087]
    }
}

data_bra = {
    "L_dis": {
        'NRMSE': [0.0447, 0.0429, 0.0443],
        'PSNR': [27.35, 27.70, 27.40],
        'MS-SSIM': [0.8965, 0.9008, 0.8966]
    },
    "L_dis_contrast": {
        'NRMSE': [0.0454, 0.0465, 0.0480],
        'PSNR': [27.25, 27.04, 26.68],
        'MS-SSIM': [0.8931, 0.8889, 0.8836]
    }
}
datas = [data_pro, data_bra]
for data in datas:
    # 可视化设计
    plt.style.use('seaborn-whitegrid')
    nature_colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
    colors = sns.color_palette(nature_colors)
    dark_colors = [sns.dark_palette(color, n_colors=1, input="rgb")[0] for color in colors]
    labels = [r'$L_{\mathrm{dis}}$', r'$L_{\mathrm{dis}}^{\mathrm{contrast}}$']
    x_labels = [r'$\lambda=0.1$', r'$\lambda=0.5$', r'$\lambda=1.0$']
    bar_width = 0.5

    # 创建画布
    fig = plt.figure(figsize=(18, 7), dpi=120)
    gs = GridSpec(1, 3, wspace=0.35)


    # 辅助函数
    def autolabel(bars, ax, metric):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}' if metric == 'PSNR' else f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)


    def set_ylim(ax, values):
        min_val, max_val = min(values), max(values)
        padding = (max_val - min_val) * 0.1
        ax.set_ylim(min_val - padding, max_val + padding)


    def plot_bars_and_lines(ax, x, values1, values2, metric):
        # 绘制柱状图
        bars1 = ax.bar(x - bar_width / 2, values1, bar_width,
                       color=colors[0], label=labels[0])
        bars2 = ax.bar(x + bar_width / 2, values2, bar_width,
                       color=colors[1], label=labels[1])
        autolabel(bars1, ax, metric)
        autolabel(bars2, ax, metric)

        # 绘制折线图
        ax.plot(x - bar_width / 2, values1, color=dark_colors[0], marker='o',markersize=4, linestyle='-', linewidth=1.5)
        ax.plot(x + bar_width / 2, values2, color=dark_colors[1], marker='o',markersize=4, linestyle='-', linewidth=1.5)


    # 对每个指标绘制子图
    metrics = ['NRMSE', 'PSNR', 'MS-SSIM']
    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        x = np.arange(len(x_labels)) * 1.3

        # 绘制柱状图和折线图
        plot_bars_and_lines(ax, x, data['L_dis'][metric], data['L_dis_contrast'][metric], metric)

        # 设置坐标轴
        ax.set_xlabel(r'$\lambda$ Value', fontsize=18, labelpad=10)
        ax.set_ylabel(metric + (' ↓' if metric == 'NRMSE' else ' ↑'),
                      fontsize=18, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=18)
        ax.set_title(metric, fontsize=18, pad=10)
        ax.tick_params(axis='y', labelsize=18)
        # 自动设置Y轴范围
        all_values = data['L_dis'][metric] + data['L_dis_contrast'][metric]
        set_ylim(ax, all_values)

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.7)

    # 添加全局图例和标题
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.04), fontsize=20, frameon=True)

    # 保存和显示
    plt.savefig('/data/newnas_1/MJY_file/visualize_all/bar_chart/{}.png'.format('BraSyn' if data == data_bra else 'Prostate'),
                bbox_inches='tight', dpi=600)
    plt.show()
