import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec  # 更灵活的布局控制
import SimpleITK as sitk
# 强制使用独立窗口显示
import matplotlib

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

    # 动态纵轴范围计算
    def get_global_ylim(slice_idx, col):
        all_values = [real_image[slice_idx, :, col]] + [img[slice_idx, :, col] for img in generated_images]
        global_min = np.min([np.min(arr) for arr in all_values])
        global_max = np.max([np.max(arr) for arr in all_values])
        padding = (global_max - global_min) * 0.05
        return (global_min - padding, global_max + padding)

    # 初始化曲线
    ymin, ymax = get_global_ylim(initial_slice, initial_col)
    line_real, = ax_curves.plot(real_image[initial_slice, :, initial_col], 'r-', lw=1.0, label='Real')
    lines_gen = [
        ax_curves.plot(img[initial_slice, :, initial_col], color=color, ls=':', lw=1.0, label=name)[0]
        for img, name, color in zip(generated_images, model_names, colors)
    ]

    # 曲线图样式
    ax_curves.set_title("Intensity Profiles", fontsize=10)
    ax_curves.set_xlabel("Row Index", fontsize=8, labelpad=2)
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
    横线版：支持选择行并绘制列强度曲线

    参数：
    - real_image: 2D numpy数组 (真实图像)
    - generated_images: 多个生成模型的图像列表 [img1, img2, ...]
    - model_names: 对应的模型名称列表
    - initial_row: 初始行位置
    """
    # 校验输入
    assert len(generated_images) == len(model_names), "模型数量与名称数量不匹配"

    # --- 创建画布 ---
    fig = plt.figure(figsize=(15, 6), dpi=120)
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1.0, 1.3],  # 右侧增加左边距
                  height_ratios=[0.95, 0.05],
                  hspace=0.02, wspace=0.12)  # 增加水平间距
    # --- 子图定义 ---
    ax_image = fig.add_subplot(gs[0, 0])
    ax_curves = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, 0])

    # --- 左侧真实图像（无坐标轴）---
    ax_image.imshow(real_image, cmap='gray', aspect='auto')
    ax_image.set_title("Click to Select Row", fontsize=9, pad=8)
    horizontal_line = ax_image.axhline(y=initial_row, color='r', linestyle='--', linewidth=0.8)
    ax_image.axis('off')  # 隐藏所有坐标轴元素

    # --- 右侧曲线图 ---
    # colors = plt.cm.tab10(np.linspace(0, 1, len(generated_images)))
    colors = ['#00FF00'] + ['#808080']*(len(generated_images)-1)
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
    def get_global_ylim(row):
        all_values = [real_image[row, :]] + [img[row, :] for img in generated_images]
        global_min = np.min([np.min(arr) for arr in all_values])
        global_max = np.max([np.max(arr) for arr in all_values])
        padding = (global_max - global_min) * 0.05
        return (global_min - padding, global_max + padding)

    # 初始化曲线
    ymin, ymax = get_global_ylim(initial_row)
    line_real, = ax_curves.plot(real_image[initial_row, :], 'r-', lw=1.2, label='Real')
    lines_gen = [
        ax_curves.plot(img[initial_row, :], color=color, ls=':', lw=1.2, label=name)[0]
        for img, name, color in zip(generated_images, model_names, colors)
    ]

    # 曲线图样式
    ax_curves.set_title("Intensity Profiles", fontsize=10)
    ax_curves.set_xlabel("Column Index", fontsize=8, labelpad=2)  # 调整标签间距
    ax_curves.set_ylabel("Intensity", fontsize=8, labelpad=5)  # 增加纵轴标签右边距
    ax_curves.yaxis.set_label_coords(-0.08, 0.5)  # 向左微调纵轴标签位置
    ax_curves.legend(fontsize=7, framealpha=0.95, loc='upper right')
    ax_curves.grid(True, alpha=0.15)
    ax_curves.set_ylim(ymin, ymax)
    ax_curves.tick_params(labelsize=7, pad=2)  # 减少刻度标签与轴的距离

    # --- 紧凑滑块 ---
    slider = Slider(
        ax=ax_slider,
        label='',
        valmin=0,
        valmax=real_image.shape[0] - 1,
        valinit=initial_row,
        valstep=1,
        color='#e0e0e0',
        track_color='#808080',
        handle_style={'facecolor': '#ff4444', 'size': 3}
    )
    slider.valtext.set_fontsize(7)
    slider.valtext.set_position((0.5, -1.5))
    ax_slider.axis('off')

    # --- 交互更新函数 ---
    def update(val):
        row = int(round(slider.val))
        horizontal_line.set_ydata([row, row])
        line_real.set_ydata(real_image[row, :])
        for line, img in zip(lines_gen, generated_images):
            line.set_ydata(img[row, :])
        ax_curves.set_ylim(*get_global_ylim(row))
        fig.canvas.draw_idle()

    # 事件绑定
    def onclick(event):
        if event.inaxes == ax_image:
            row = int(round(event.ydata))
            slider.set_val(row)
            update(row)

    fig.canvas.mpl_connect('button_press_event', onclick)
    slider.on_changed(update)

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
    gt_dir = "S:/home/user4/sharedata/newnas_1/MJY_file/BraTS_dataset/data_pre_original/images_ts"
    whole_dir = "S:/home/user4/sharedata/newnas_1/MJY_file/diffusion/train_result_BraTs"
    id_num = "BraTS-GLI-00525-000"
    slice_index = 15

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
        ("4_ds_diff", "ddim"),
        # ("4_ds_diff", "dpm"),
        ("46_ldm", "ddim"),
        # ("46_ldm", "dpm"),
        ("33_ds_diff", "ddim"),
        # ("33_ds_diff", "dpm"),
        ("09_dit", "ddim"),
        # ("09_dit", "dpm"),
        ("11_dit", "ddim"),
        # ("11_dit", "dpm"),
        ("49_ds_diff", "ddim"),
        # ("49_ds_diff", "dpm"),
    ]
    labels = [
        "DDPM",
        "w/ MS-UNet",
        "w/ Disentanglor",
        "w/ $L_{dis}$ λ=0.5",
        # "w/ L_{dis}^{contrast} λ=0.1",
        # "w/ Cross-Attn2",
    ]
    model_names = [
        ("4_ds_diff", "ddim"),
        ("8_ds_diff", "ddim"),
        ("6_ds_diff", "ddim"),
        ("63_ds_diff", "ddim"),
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
        ("64_ds_diff", "ddim"),
        ("22_ds_diff", "ddim"),
        ("52_ds_diff", "ddim"),
        ("51_ds_diff", "ddim"),
        ("60_ds_diff", "ddim"),
        ("74_ds_diff", "ddim"),
    ]
    exp_dirs = []
    for i, sample in model_names:
        if "dit" not in i:
            exp_dirs.append(
                whole_dir + "/BraTs_synthesis_{}_fold5-1/pred_nii_{}_20_eta0_checkpoint".format(i, sample)
            )
        else:
            exp_dirs.append(
                os.path.dirname(os.path.dirname(
                    whole_dir)) + "/SOTA_models/DiT_BraTs_test/0{}-DiT-XL-2/{}20/itk_result/".format(
                    i.split("_")[0], sample)
            )
        if not os.path.exists(exp_dirs[-1]):
            raise FileNotFoundError("File not found: {}".format(exp_dirs[-1]))

    # 生成示例数据
    real_img = os.path.join(gt_dir, id_num , "ce.nii.gz")
    real_img = sitk.GetArrayFromImage(sitk.ReadImage(real_img))
    # slice_index = real_img.shape[0] - slice_index
    real_img = real_img
    real_img = np.flip(real_img, axis=1)
    img_list = [os.path.join(img_path,id_num+"_pred.nii.gz") for img_path in exp_dirs]
    # 生成多个模型的模拟数据

    model_imgs = [np.flip(sitk.GetArrayFromImage(sitk.ReadImage(img_path)), axis=1) for img_path in img_list]

    # 调用可视化函数
    plot_multi_model_curves_v(
        real_image=real_img,
        generated_images=model_imgs,
        model_names=labels,
        initial_col=125
    )