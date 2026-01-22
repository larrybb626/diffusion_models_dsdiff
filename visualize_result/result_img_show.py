# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：result_img_show.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/3/15 16:31 
"""
import os

import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # 使用交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SimpleITK as sitk

all_pred = [
    'S:/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/mulD_RegGAN/braCE_MRI_simulate_PCa_64_pix2pix_mulD_fold5-1/pred_nii',
    'S:/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/SOTA_GAN/ResVit_brats/result',
    'S:/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/DiscDiff_test/BraTs_v_predict_7e4_uknow/itk_save_dir',
    'S:/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/controlnet-model-brats/controlnet_result/checkpoint-50000/itk_result',
    'S:/home/user4/sharedata/newnas_1/MJY_file/diffusion/train_result_BraTs/BraTs_synthesis_74_ds_diff_fold5-1/pred_nii_ddim_20_eta0_checkpoint',
    'S:\\home\\user4\\sharedata\\newnas_1\\MJY_file\\BraTS_dataset\\data_pre_original\\images_ts'
]

all_pred = [
    'S:/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/mulD_RegGAN/CE_MRI_simulate_PCa_63_pix2pix_mulD_fold5-1/pred_nii',
    'S:/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/SOTA_GAN/ResVit_all_Seq_noval_new_pretrain/result',
    'S:/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/DiscDiff_test/Prostate_v_predict_7e4/itk_save_dir',
    'S:/home/user4/sharedata/newnas_1/MJY_file/SOTA_models/controlnet-model/controlnet_result/checkpoint-50000/itk_result',
    'S:/home/user4/sharedata/newnas_1/MJY_file/diffusion/train_result/CE_MRI_synthesis_154_ds_diff_fold5-1/pred_nii_ddim_20_eta0_checkpoint',
    'S:\\home\\user4\\sharedata\\newnas\\MJY_file\\CE-MRI\\PCa_new\\CE-MRI-PCa-new-pre-320320-01norm\\images_ts'
]
labels = ['cGAN', 'ResViT', 'DisC-Diff', 'SD3', 'DS-Diff', 'Real']
id_name = 'BraTS-GLI-00285-000'
id_name = '0000811681'
all_pred_img = []
for i in all_pred:
    if os.path.basename(i) == 'images_ts':
        pred_img = os.path.join(i, id_name, "T1CE.nii.gz")
    else:
        pred_img = os.path.join(i, f"{id_name}_pred.nii.gz")
        if not os.path.isfile(pred_img):
            pred_img = os.path.join(i, f"{id_name}.nii.gz")
    all_pred_img.append(pred_img)
# 1. 准备示例数据
# 假设有6个三维图像，每个图像形状为 (50, 256, 256)
images = [np.flip(sitk.GetArrayFromImage(sitk.ReadImage(i))) for i in all_pred_img]
image_depth = images[0].shape[0]  # 切片数量
height, width = images[0].shape[1], images[0].shape[2]  # 图像高度和宽度

# 2. 创建图形和子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # 展平为1D数组
plt.subplots_adjust(bottom=0.2)  # 调整布局，为滑动条留空间
# 设置子图标题
for ax, title in zip(axes, labels):
    ax.set_title(title)

# 3. 添加用于显示ROI放大的inset axes
inset_axes_list = []
for ax in axes:
    inset_ax = inset_axes(ax, width="30%", height="30%", loc='upper right')
    inset_ax.axis('off')
    inset_axes_list.append(inset_ax)

# 4. 创建单一滑动条
ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
slider = plt.Slider(ax_slider, 'Slice', 0, image_depth - 1, valinit=0, valstep=1)

# 5. 定义ROI框大小
roi_width = 50
roi_height = 50

# 6. 为每个子图创建Rectangle patch表示ROI
rect_patches = []
for ax in axes:
    rect = Rectangle((0, 0), 0, 0, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(rect)
    rect_patches.append(rect)


# 7. 定义更新inset axes的函数
def update_insets():
    slice_index = int(slider.val)
    for ax, rect, inset_ax, image in zip(axes, rect_patches, inset_axes_list, images):
        if rect.get_width() > 0 and rect.get_height() > 0:  # 确保ROI有效
            x_left, y_bottom = rect.get_xy()  # 获取ROI位置
            width, height = rect.get_width(), rect.get_height()
            x_right, y_top = x_left + width, y_bottom + height
            # 提取ROI区域
            roi = image[slice_index, int(y_bottom):int(y_top), int(x_left):int(x_right)]
            # 使用原图的像素范围
            vmin, vmax = image[slice_index].min(), image[slice_index].max()
            inset_ax.clear()  # 清除小框内容
            inset_ax.imshow(roi, cmap='gray', vmin=vmin, vmax=vmax)  # 显示ROI
            inset_ax.axis('off')  # 关闭坐标轴


# 8. 定义滑动条更新函数
def update(val):
    slice_index = int(slider.val)
    for ax, image in zip(axes, images):
        img_data = image[slice_index]  # 获取当前切片
        vmin, vmax = img_data.min(), img_data.max()  # 记录像素范围
        ax.imshow(img_data, cmap='gray', vmin=vmin, vmax=vmax)  # 显示原图
        ax.axis('off')
    update_insets()  # 更新小框
    fig.canvas.draw_idle()


# 9. 绑定滑动条事件
slider.on_changed(update)

# 10. 初始显示
update(0)


# 11. 定义鼠标单击事件处理函数
def on_click(event):
    if event.inaxes in axes:
        x_c = event.xdata
        y_c = event.ydata
        # 计算ROI边界
        x_left = max(0, x_c - roi_width / 2)
        x_right = min(width, x_c + roi_width / 2)
        y_bottom = max(0, y_c - roi_height / 2)
        y_top = min(height, y_c + roi_height / 2)
        # 更新所有ROI框
        for rect in rect_patches:
            rect.set_xy((x_left, y_bottom))
            rect.set_width(x_right - x_left)
            rect.set_height(y_top - y_bottom)
        update_insets()
        fig.canvas.draw_idle()


# 12. 连接鼠标事件
fig.canvas.mpl_connect('button_press_event', on_click)

# 13. 显示
plt.show()
