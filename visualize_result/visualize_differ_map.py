# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：visualize_differ_map.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/2/15 14:57 
"""
import argparse
import os.path
from concurrent.futures import ProcessPoolExecutor, as_completed
import SimpleITK as sitk
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Manager
import multiprocessing as mp

def load_nii_image(nii_path):
    """加载 NIfTI 图像并转换为 numpy 数组"""
    img = sitk.ReadImage(nii_path)
    img_array = sitk.GetArrayFromImage(img)  # (depth, height, width)
    return img_array, img


def save_difference_image(diff_array, output_path, cmap='seismic'):
    """保存差值图像"""
    plt.figure(figsize=(6, 6))
    plt.imshow(diff_array, cmap=cmap, origin='lower')
    plt.colorbar()  # 颜色图例
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_multiple_differences(diff_images, labels, output_path, method, slice_idx):
    """Plot multiple difference images in a grid layout"""
    n = len(diff_images)
    ncols = 3
    nrows = math.ceil(n / ncols)

    # Get image shape
    img_height, img_width = diff_images[0].shape

    # Calculate figure size to maintain aspect ratio
    height_per_subplot = 4  # inches
    width_per_subplot = (img_width / img_height) * height_per_subplot
    figsize = (ncols * width_per_subplot, nrows * height_per_subplot)

    # Find overall vmin and vmax
    all_values = np.array(diff_images).flatten()
    vmin = all_values.min()
    vmax = all_values.max()

    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)

    # Flatten axes for easy indexing
    axes = axes.ravel()
    # gist_heat
    cmap_str = 'seismic' if method == 'abs' else 'seismic'
    #[seismic,gist_heat,virids,magma]
    # Plot each difference image
    for i, (diff_img, label) in enumerate(zip(diff_images, labels)):
        ax = axes[i]
        im = ax.imshow(diff_img, cmap=cmap_str, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(label)
        ax.axis('off')

    for i in range(n, nrows * ncols):
        axes[i].set_visible(False)
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, location='right', fraction=0.02, pad=0.02)
    cbar.ax.set_ylabel("Difference Intensity")

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def compute_difference(A, B, method='raw'):
    """计算两张图像的差值（支持3种方法）"""
    if method == 'raw':  # A - B
        return A - B
    elif method == 'normalized':  # 归一化后相减
        A_norm = (A - np.mean(A)) / np.std(A)
        B_norm = (B - np.mean(B)) / np.std(B)
        return A_norm - B_norm
    elif method == 'abs':  # 绝对值 |A - B|
        return np.abs(A - B)
    else:
        raise ValueError("Unsupported method: choose from ['raw', 'normalized', 'abs']")

def process_id(id_num, real_dir, whole_dir, output_dir, exp_name, labels):

    # 读取两张 NIfTI 图像
    nii_path_gt = os.path.join(real_dir, id_num, "T1CE.nii.gz")
    nii_path_list = [os.path.join(e_dir, id_num + "_pred.nii.gz") for e_dir in excel_dirs]

    nparray_list = [load_nii_image(nii_path)[0] for nii_path in nii_path_list]
    A, _ = load_nii_image(nii_path_gt)
    # 计算差值图像并保存
    methods = ['abs']

    for method in methods:
        output_dir_final = os.path.join(output_dir,method, id_num)
        os.makedirs(output_dir_final, exist_ok=True)
        diff_images_list = [compute_difference(model_img, A, method) for model_img in nparray_list]
        for slice_idx in range(A.shape[0]):
            diff_images = [diff_img[slice_idx] for diff_img in diff_images_list]
            output_path = os.path.join(output_dir_final, f"diff_{slice_idx}_{method}.png")
            plot_multiple_differences(diff_images, labels, output_path, method, slice_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="可视化医学图像差值图")
    parser.add_argument('--real_dir', type=str,
                        default="/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts")
    parser.add_argument('--fake_dir', type=str,
                        default="/data/newnas_1/MJY_file/diffusion/train_result")
    parser.add_argument('--output_dir', type=str, default="/data/newnas_1/MJY_file/visualize_all/differmap_P4")
    parser.add_argument('--id_num', type=str, default="0000894408")

    args = parser.parse_args()
    real_dir = args.real_dir
    whole_dir = args.fake_dir
    id_list = sorted(os.listdir(real_dir))
    # labels = [
    #     "DDPM+DDIM", "DDPM+DPM-Solver",
    #     "LDM-oVAE+DDIM", "LDM-oVAE+DPM-Solver",
    #     "LDM-sVAE+DDIM", "LDM-sVAE+DPM-Solver",
    #     "DiT-oVAE+DDIM",
    #     # "DiT-oVAE+DPM-Solver",
    #     "DiT-sVAE+DDIM", "DiT-sVAE+DPM-Solver",
    #     "DiT-nVAE+DDIM", "DiT-nVAE+DPM-Solver"
    # ]
    # exp_name = [
    #     ("94_ds_diff", "ddim"),
    #     ("94_ds_diff", "dpm"),
    #     ("13_ldm", "ddim"),
    #     ("13_ldm", "dpm"),
    #     ("19_ldm", "ddim"),
    #     ("19_ldm", "dpm"),
    #     ("29_dit", "ddim"),
    #     # ("29_dit", "dpm"),
    #     ("18_dit", "ddim"),
    #     ("18_dit", "dpm"),
    #     ("108_ds_diff", "ddim"),
    #     ("108_ds_diff", "dpm"),
    # ]
    # labels = [
    #     "DDPM",
    #     "w/ MS-UNet",
    #     "w/ SADM",
    #     "w/ $L_{dis}$ λ=0.5",
    #     "w/ $L_{dis}^{contrast}$ λ=0.1",
    #     "w/ H-Cross-Attn",
    # ]
    # exp_name = [
    #     ("94_ds_diff", "ddim"),
    #     ("55_ds_diff", "ddim"),
    #     ("49_ds_diff", "ddim"),
    #     ("139_ds_diff", "ddim"),
    #     ("142_ds_diff", "ddim"),
    #     ("95_ds_diff", "ddim"),
    # ]
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
        ("154_ds_diff", "ddim"),
    ]


    excel_dirs = []
    for i, sample in exp_name:
        if "dit" not in i:
            excel_dirs.append(
                whole_dir + "/CE_MRI_synthesis_{}_fold5-1/pred_nii_{}_20_eta0_checkpoint".format(i, sample)
            )
        else:
            excel_dirs.append(
                os.path.dirname(os.path.dirname(
                    whole_dir)) + "/SOTA_models/DiT_test/0{}-DiT-XL-2/{}20/itk_result/".format(
                    i.split("_")[0], sample)
            )
        if not os.path.exists(excel_dirs[-1]):
            raise FileNotFoundError("File not found: {}".format(excel_dirs[-1]))

    num_workers = min(32, len(id_list))  # 限制进程数
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交任务并创建 future 对象
        futures = {executor.submit(process_id, id_num, real_dir, whole_dir, args.output_dir, excel_dirs, labels): id_num
                   for id_num in id_list}
        # 使用 tqdm 跟踪进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            id_num = futures[future]
            try:
                future.result()  # 获取结果，检查是否有异常
            except Exception as e:
                print(f"Error processing {id_num}: {e}")