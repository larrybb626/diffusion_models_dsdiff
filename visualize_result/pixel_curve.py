import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec  # 更灵活的布局控制
import SimpleITK as sitk
# 强制使用独立窗口显示
import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')  # 或 'Qt5Agg'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

# 强制使用独立窗口显示
import matplotlib

matplotlib.use('TkAgg')


def plot_multi_model_curves_v(real_image, generated_images, model_names, initial_col=50):
    """
    支持三维图像切片切换的最终版
    """
    # 校验输入
    assert len(generated_images) == len(model_names), "模型数量与名称数量不匹配"
    assert real_image.ndim == 3, "真实图像必须是三维数组 (n_slices, height, width)"
    for img in generated_images:
        assert img.ndim == 3, "生成图像必须是三维数组 (n_slices, height, width)"

    # --- 创建画布 ---
    fig = plt.figure(figsize=(15, 6), dpi=120)
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=[1.0, 1.3],  # 右侧增加左边距
                  height_ratios=[0.8, 0.1, 0.1],  # 新增切片滑块行
                  hspace=0.1, wspace=0.12)  # 调整间距

    # --- 子图定义 ---
    ax_image = fig.add_subplot(gs[0, 0])  # 真实图像区域
    ax_curves = fig.add_subplot(gs[0, 1])  # 曲线区域
    ax_slice_slider = fig.add_subplot(gs[1, 0])  # 切片滑块区域
    ax_col_slider = fig.add_subplot(gs[2, 0])  # 列滑块区域

    # --- 初始化切片索引 ---
    initial_slice = 0  # 初始切片索引
    n_slices = real_image.shape[0]  # 切片总数

    # --- 左侧真实图像（无坐标轴）---
    def update_image(slice_idx):
        """更新真实图像和曲线"""
        ax_image.clear()
        ax_image.imshow(real_image[slice_idx], cmap='gray', aspect='auto')
        ax_image.set_title(f"Slice {slice_idx} - Click to Select Column", fontsize=9, pad=8)
        ax_image.axis('off')
        return ax_image.axvline(x=initial_col, color='r', linestyle='-', linewidth=0.8)

    vertical_line = update_image(initial_slice)

    # --- 右侧曲线图 ---
    colors = [
        '#0000FF',  # 纯蓝
        '#00FF00',  # 荧光绿
        '#FFD700',  # 亮金色
        '#FF00FF',  # 品红
        '#00FFFF',  # 青蓝
        '#FFA500',  # 橙色
        '#9400D3',  # 深紫罗兰
        '#32CD32',  # 酸橙绿
        '#FF1493',  # 深粉色
        '#7FFF00'  # 查特绿
    ]

    # 动态纵轴范围计算
    def get_global_ylim(slice_idx, col):
        all_values = [real_image[slice_idx, :, col]] + [img[slice_idx, :, col] for img in generated_images]
        global_min = np.min([np.min(arr) for arr in all_values])
        global_max = np.max([np.max(arr) for arr in all_values])
        padding = (global_max - global_min) * 0.05
        return (global_min - padding, global_max + padding)

    # 初始化曲线
    ymin, ymax = get_global_ylim(initial_slice, initial_col)
    line_real, = ax_curves.plot(real_image[initial_slice, :, initial_col], 'r-', lw=0.8, label='Real')
    lines_gen = [
        ax_curves.plot(img[initial_slice, :, initial_col], color=color, ls=':', lw=0.8, label=name)[0]
        for img, name, color in zip(generated_images, model_names, colors)
    ]

    # 曲线图样式
    ax_curves.set_title("Intensity Profiles", fontsize=10)
    ax_curves.set_xlabel("Row Index", fontsize=8, labelpad=2)
    ax_curves.set_ylabel("Intensity", fontsize=8, labelpad=5)
    ax_curves.yaxis.set_label_coords(-0.06, 0.5)
    ax_curves.legend(fontsize=7, framealpha=0.95, loc='upper right')
    ax_curves.grid(True, alpha=0.15)
    ax_curves.set_ylim(ymin, ymax)
    ax_curves.tick_params(labelsize=7, pad=2)

    # --- 切片滑块 ---
    slice_slider = Slider(
        ax=ax_slice_slider,
        label='Slice Index',
        valmin=0,
        valmax=n_slices - 1,
        valinit=initial_slice,
        valstep=1,
        color='#e0e0e0',
        track_color='#808080',
        handle_style={'facecolor': '#ff4444', 'size': 4}
    )
    slice_slider.label.set_fontsize(8)
    slice_slider.valtext.set_fontsize(7)

    # --- 列滑块 ---
    col_slider = Slider(
        ax=ax_col_slider,
        label='Column Index',
        valmin=0,
        valmax=real_image.shape[2] - 1,
        valinit=initial_col,
        valstep=1,
        color='#e0e0e0',
        track_color='#808080',
        handle_style={'facecolor': '#ff4444', 'size': 4}
    )
    col_slider.label.set_fontsize(8)
    col_slider.valtext.set_fontsize(7)

    # --- 交互更新函数 ---
    def update(val):
        slice_idx = int(round(slice_slider.val))
        col = int(round(col_slider.val))

        # 更新图像
        vertical_line.set_xdata([col, col])
        ax_image.clear()
        ax_image.imshow(real_image[slice_idx], cmap='gray', aspect='auto')
        ax_image.set_title(f"Slice {slice_idx} - Click to Select Column", fontsize=9, pad=8)
        ax_image.axis('off')
        ax_image.axvline(x=col, color='r', linestyle='-', linewidth=0.8)

        # 更新曲线
        line_real.set_ydata(real_image[slice_idx, :, col])
        for line, img in zip(lines_gen, generated_images):
            line.set_ydata(img[slice_idx, :, col])

        # 更新纵轴范围
        ymin, ymax = get_global_ylim(slice_idx, col)
        ax_curves.set_ylim(ymin, ymax)

        fig.canvas.draw_idle()

    # 事件绑定
    def onclick(event):
        if event.inaxes == ax_image:
            col = int(round(event.xdata))
            col_slider.set_val(col)
            update(None)

    fig.canvas.mpl_connect('button_press_event', onclick)
    slice_slider.on_changed(update)
    col_slider.on_changed(update)

    # --- 保存按钮 ---
    saveax = fig.add_axes([0.88, 0.02, 0.1, 0.04])
    save_btn = Button(saveax, 'SAVE', color='#ffffff00', hovercolor='#f0f0f030')
    save_btn.label.set_fontsize(6)

    def save_fig(event):
        plt.savefig('intensity_profiles.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    save_btn.on_clicked(save_fig)

    plt.tight_layout(pad=0.5)
    plt.show()


def plot_multi_model_curves_h(real_image, generated_images, model_names, initial_row=50):
    """
    横线版：支持三维图像切片切换
    """
    # 校验输入
    assert len(generated_images) == len(model_names), "模型数量与名称数量不匹配"
    assert real_image.ndim == 3, "真实图像必须是三维数组 (n_slices, height, width)"
    for img in generated_images:
        assert img.ndim == 3, "生成图像必须是三维数组 (n_slices, height, width)"

    # --- 创建画布 ---
    fig = plt.figure(figsize=(15, 6), dpi=120)
    gs = GridSpec(3, 2, figure=fig,
                  width_ratios=[1.0, 1.3],  # 右侧增加左边距
                  height_ratios=[0.8, 0.1, 0.1],  # 新增切片滑块行
                  hspace=0.1, wspace=0.12)  # 调整间距

    # --- 子图定义 ---
    ax_image = fig.add_subplot(gs[0, 0])  # 真实图像区域
    ax_curves = fig.add_subplot(gs[0, 1])  # 曲线区域
    ax_slice_slider = fig.add_subplot(gs[1, 0])  # 切片滑块区域
    ax_row_slider = fig.add_subplot(gs[2, 0])  # 行滑块区域

    # --- 初始化切片索引 ---
    initial_slice = 0  # 初始切片索引
    n_slices = real_image.shape[0]  # 切片总数

    # --- 左侧真实图像（无坐标轴）---
    def update_image(slice_idx):
        """更新真实图像和曲线"""
        ax_image.clear()
        ax_image.imshow(real_image[slice_idx], cmap='gray', aspect='auto')
        ax_image.set_title(f"Slice {slice_idx} - Click to Select Row", fontsize=9, pad=8)
        ax_image.axis('off')
        return ax_image.axhline(y=initial_row, color='r', linestyle='-', linewidth=0.8)

    horizontal_line = update_image(initial_slice)

    # --- 右侧曲线图 ---
    colors = [
        '#FF00FF',  # 品红
        '#FFD700',  # 亮金色
        '#00FF00',  # 荧光绿
        '#00FFFF',  # 青蓝
        '#FFA500',  # 橙色
        '#0000FF',  # 纯蓝
        '#9400D3',  # 深紫罗兰
        '#32CD32',  # 酸橙绿
        '#FF1493',  # 深粉色
        '#7FFF00'  # 查特绿
    ]
    # colors = [
    #     '#1F77B4',  # 深蓝 - 鲜明且沉稳
    #     '#FF4500',  # 橙红 - 高亮度，对比强烈
    #     '#228B22',  # 森林绿 - 深绿，易区分
    #     '#DC143C',  # 猩红 - 鲜艳红，与绿对比强
    #     '#8A2BE2',  # 紫罗兰 - 独特紫色
    #     '#FFD700',  # 金黄 - 明亮且醒目
    #     '#00CED1',  # 暗青 - 清新，与其他色区分
    #     '#FF69B4',  # 热粉 - 高饱和度粉色
    #     '#4B0082',  # 靛紫 - 深色，与浅色对比强
    #     '#ADFF2F'  # 酸橙绿 - 高亮绿，与深色对比
    # ]
    # 动态纵轴范围计算
    def get_global_ylim(slice_idx, row):
        all_values = [real_image[slice_idx, row, :]] + [img[slice_idx, row, :] for img in generated_images]
        global_min = np.min([np.min(arr) for arr in all_values])
        global_max = np.max([np.max(arr) for arr in all_values])
        padding = (global_max - global_min) * 0.05
        return (global_min - padding, global_max + padding)

    # 初始化曲线
    ymin, ymax = get_global_ylim(initial_slice, initial_row)
    line_real, = ax_curves.plot(real_image[initial_slice, initial_row, :], 'r-', lw=1.0, label='Real')
    lines_gen = [
        ax_curves.plot(img[initial_slice, initial_row, :], color=color, ls=':', lw=1.0, label=name)[0]
        for img, name, color in zip(generated_images, model_names, colors)
    ]

    # 曲线图样式
    ax_curves.set_title("Intensity Profiles", fontsize=10)
    ax_curves.set_xlabel("Column Index", fontsize=8, labelpad=2)
    ax_curves.set_ylabel("Intensity", fontsize=8, labelpad=5)
    ax_curves.yaxis.set_label_coords(-0.06, 0.5)
    ax_curves.legend(fontsize=9, framealpha=0.95, loc='upper right')
    ax_curves.grid(True, alpha=0.15)
    ax_curves.set_ylim(ymin, ymax)
    ax_curves.tick_params(labelsize=7, pad=2)

    # --- 切片滑块 ---
    slice_slider = Slider(
        ax=ax_slice_slider,
        label='Slice Index',
        valmin=0,
        valmax=n_slices - 1,
        valinit=initial_slice,
        valstep=1,
        color='#e0e0e0',
        track_color='#808080',
        handle_style={'facecolor': '#ff4444', 'size': 4}
    )
    slice_slider.label.set_fontsize(8)
    slice_slider.valtext.set_fontsize(7)

    # --- 行滑块 ---
    row_slider = Slider(
        ax=ax_row_slider,
        label='Row Index',
        valmin=0,
        valmax=real_image.shape[1] - 1,
        valinit=initial_row,
        valstep=1,
        color='#e0e0e0',
        track_color='#808080',
        handle_style={'facecolor': '#ff4444', 'size': 4}
    )
    row_slider.label.set_fontsize(8)
    row_slider.valtext.set_fontsize(7)

    # --- 交互更新函数 ---
    def update(val):
        slice_idx = int(round(slice_slider.val))
        row = int(round(row_slider.val))

        # 更新图像
        horizontal_line.set_ydata([row, row])
        ax_image.clear()
        ax_image.imshow(real_image[slice_idx], cmap='gray', aspect='auto')
        ax_image.set_title(f"Slice {slice_idx} - Click to Select Row", fontsize=9, pad=8)
        ax_image.axis('off')
        ax_image.axhline(y=row, color='r', linestyle='-', linewidth=0.8)

        # 更新曲线
        line_real.set_ydata(real_image[slice_idx, row, :])
        for line, img in zip(lines_gen, generated_images):
            line.set_ydata(img[slice_idx, row, :])

        # 更新纵轴范围
        ymin, ymax = get_global_ylim(slice_idx, row)
        ax_curves.set_ylim(ymin, ymax)

        fig.canvas.draw_idle()

    # 事件绑定
    def onclick(event):
        if event.inaxes == ax_image:
            row = int(round(event.ydata))
            row_slider.set_val(row)
            update(None)

    fig.canvas.mpl_connect('button_press_event', onclick)
    slice_slider.on_changed(update)
    row_slider.on_changed(update)

    # --- 保存按钮 ---
    saveax = fig.add_axes([0.88, 0.02, 0.1, 0.04])
    save_btn = Button(saveax, 'SAVE', color='#ffffff00', hovercolor='#f0f0f030')
    save_btn.label.set_fontsize(6)

    def save_fig(event):
        plt.savefig('intensity_profiles.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    save_btn.on_clicked(save_fig)

    plt.tight_layout(pad=0.5)
    plt.show()
# 使用示例 -------------------------------------------------
if __name__ == "__main__":
    plot_save_dir = "S:/home/user4/sharedata/newnas_1/MJY_file/visualize_all/box_plot"
    gt_dir = "S:/home/user4/sharedata/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts"
    whole_dir = "S:/home/user4/sharedata/newnas_1/MJY_file/diffusion/train_result"
    id_num = "0000303571"
    slice_index = 8

    # 示例：使用附件表格中的 NRMSE 数据
    # 这里假设每组数据是多个实验的分布，实际应用中需提供完整数据集
    labels = [
        "DDPM+DDIM",
        # "DDPM+DPM-Solver",
        "LDM-oVAE+DDIM",
        # "LDM-oVAE+DPM-Solver",
        "LDM-sVAE+DDIM",
        # "LDM-sVAE+DPM-Solver",
        "DiT-oVAE+DDIM",
        # "DiT-oVAE+DPM-Solver",
        "DiT-sVAE+DDIM",
        # "DiT-sVAE+DPM-Solver",
        "DiT-nVAE+DDIM",
        # "DiT-nVAE+DPM-Solver"
    ]
    model_names = [
        ("94_ds_diff", "ddim"),
        # ("94_ds_diff", "dpm"),
        ("13_ldm", "ddim"),
        # ("13_ldm", "dpm"),
        ("19_ldm", "ddim"),
        # ("19_ldm", "dpm"),
        ("29_dit", "ddim"),
        # ("29_dit", "dpm"),
        ("18_dit", "ddim"),
        # ("18_dit", "dpm"),
        ("108_ds_diff", "ddim"),
        # ("108_ds_diff", "dpm"),
    ]

    labels = [
        "DDPM",
        "w/ MS-UNet",
        "w/ SADM",
        "w/ $L_{dis}$ λ=0.5",
        # "w/ L_{dis}^{contrast} λ=0.1",
        # "w/ Cross-Attn2",
    ]
    model_names = [
        ("94_ds_diff", "ddim"),
        ("55_ds_diff", "ddim"),
        ("49_ds_diff", "ddim"),
        ("139_ds_diff", "ddim"),
        # ("142_ds_diff", "ddim"),
        # ("95_ds_diff", "ddim"),
    ]
    labels = [
        "w/o SG",
        "Sobel",
        "Canny",
        "Laplacian",
        "SFG",
        "Sobel+SFG",
    ]
    model_names = [
        ("139_ds_diff", "ddim"),
        ("144_ds_diff", "ddim"),
        ("105_ds_diff", "ddim"),
        ("145_ds_diff", "ddim"),
        ("134_ds_diff", "ddim"),
        ("156_ds_diff", "ddim"),
    ]
    exp_dirs = []
    for i, sample in model_names:
        if "dit" not in i:
            exp_dirs.append(
                whole_dir + "/CE_MRI_synthesis_{}_fold5-1/pred_nii_{}_20_eta0_checkpoint".format(i, sample)
            )
        else:
            exp_dirs.append(
                os.path.dirname(os.path.dirname(
                    whole_dir)) + "/SOTA_models/DiT_test/0{}-DiT-XL-2/{}20/itk_result/".format(
                    i.split("_")[0], sample)
            )
        if not os.path.exists(exp_dirs[-1]):
            raise FileNotFoundError("File not found: {}".format(exp_dirs[-1]))

    # 生成示例数据
    real_img = os.path.join(gt_dir, id_num , "T1CE.nii.gz")
    real_img = sitk.GetArrayFromImage(sitk.ReadImage(real_img))
    # slice_index = real_img.shape[0] - slice_index
    real_img = real_img
    real_img = np.flip(real_img, axis=1)
    img_list = [os.path.join(img_path,id_num+"_pred.nii.gz") for img_path in exp_dirs]
    # 生成多个模型的模拟数据

    model_imgs = [np.flip(sitk.GetArrayFromImage(sitk.ReadImage(img_path)), axis=1) for img_path in img_list]

    # 调用可视化函数
    plot_multi_model_curves_h(
        real_image=real_img,
        generated_images=model_imgs,
        model_names=labels,
        initial_row=125
    )