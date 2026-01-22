# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：preprocess_nii.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/10/23 17:08
Get_lesion_slice, normalize, to_niigz
"""

import os.path

import SimpleITK as sitk
import numpy as np
import tqdm
def clip_mri_intensity(image, low_perc=2, high_perc=98):
    """按百分位数截断 MRI 图像的像素值"""
    low, high = np.percentile(image, [low_perc, high_perc])
    return np.clip(image, low, high)

def clip_zscore(image, threshold=3.0):
    """裁剪 z-score 归一化后的 MRI 图像，防止过度亮化"""
    mean, std = np.mean(image), np.std(image)
    low, high = mean - threshold * std, mean + threshold * std
    return np.clip(image, low, high)
def resample_image(image, new_spacing, interpolator=sitk.sitkLinear):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # 计算新的 size，使物理尺寸不变
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())

    return resample.Execute(image)

if __name__ == "__main__":
    phase = ['test']
    # crop_size = 192  #裁剪224
    dir = "/home/user15/sharedata/newnas/MJY_file/CE-MRI/PCa_new/output_normalization_renming/"
    save_dir = "/home/user15/sharedata/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts_rm"
    all_id = os.listdir(dir)
    for id_name in tqdm.tqdm(all_id, desc="Processing"):
        ce_path, seg_path, t1_path, t2_path, dwi_path = None, None, None, None, None
        path = os.path.join(dir, id_name)
        ce_path = os.path.join(path, 'T1CE.nii.gz')
        seg_path = os.path.join(path, 'T1CE_body_mask.nii.gz')
        t1_path = os.path.join(path, 'T1.nii.gz')
        t2_path = os.path.join(path, 'T2.nii.gz')
        dwi_path = os.path.join(path, 'B1400.nii.gz')
        # 读取数据
        assert ce_path and seg_path and t1_path and t2_path and dwi_path
        t1 = sitk.ReadImage(t1_path, sitk.sitkFloat32)
        t2 = sitk.ReadImage(t2_path, sitk.sitkFloat32)
        ce = sitk.ReadImage(ce_path, sitk.sitkFloat32)
        flair = sitk.ReadImage(dwi_path, sitk.sitkFloat32)
        seg = sitk.ReadImage(seg_path)
        t1_array = sitk.GetArrayFromImage(t1)
        t2_array = sitk.GetArrayFromImage(t2)
        ce_array = sitk.GetArrayFromImage(ce)
        flair_array = sitk.GetArrayFromImage(flair)
        seg_array = sitk.GetArrayFromImage(seg)
        assert t1_array.shape[-1] == 320
        # 截断
        t1_array = clip_mri_intensity(t1_array, 0.1, 99.99)
        t2_array = clip_mri_intensity(t2_array, 0.1, 99.99)
        ce_array = clip_mri_intensity(ce_array, 0.1, 99.99)
        flair_array = clip_mri_intensity(flair_array, 0.1, 99.99)
        # 缩放到-1~1
        t1_array = (t1_array - t1_array.min()) / (t1_array.max() - t1_array.min()) * 2 - 1
        t2_array = (t2_array - t2_array.min()) / (t2_array.max() - t2_array.min()) * 2 - 1
        ce_array = (ce_array - ce_array.min()) / (ce_array.max() - ce_array.min()) * 2 - 1
        flair_array = (flair_array - flair_array.min()) / (flair_array.max() - flair_array.min()) * 2 - 1

        t1_new = sitk.GetImageFromArray(t1_array)
        t2_new = sitk.GetImageFromArray(t2_array)
        ce_new = sitk.GetImageFromArray(ce_array)
        flair_new = sitk.GetImageFromArray(flair_array)
        seg_new = sitk.GetImageFromArray(seg_array)

        t1_spacing = t1.GetSpacing()
        t2_spacing = t2.GetSpacing()
        ce_spacing = ce.GetSpacing()
        flair_spacing = flair.GetSpacing()
        seg_spacing = seg.GetSpacing()

        t1_new.SetSpacing((t1_spacing[0], t1_spacing[1], t1_spacing[2]))
        t2_new.SetSpacing((t2_spacing[0], t2_spacing[1], t2_spacing[2]))
        ce_new.SetSpacing((ce_spacing[0], ce_spacing[1], ce_spacing[2]))
        flair_new.SetSpacing((flair_spacing[0], flair_spacing[1], flair_spacing[2]))
        seg_new.SetSpacing((seg_spacing[0], seg_spacing[1], seg_spacing[2]))

        t1_new.SetOrigin(t1.GetOrigin())
        t2_new.SetOrigin(t2.GetOrigin())
        ce_new.SetOrigin(ce.GetOrigin())
        flair_new.SetOrigin(flair.GetOrigin())
        seg_new.SetOrigin(seg.GetOrigin())

        t1_new.SetDirection(t1.GetDirection())
        t2_new.SetDirection(t2.GetDirection())
        ce_new.SetDirection(ce.GetDirection())
        flair_new.SetDirection(flair.GetDirection())
        seg_new.SetDirection(seg.GetDirection())

        seg_new = sitk.Cast(seg_new, sitk.sitkUInt8)
        os.makedirs(os.path.join(save_dir, id_name), exist_ok=True)
        sitk.WriteImage(t1_new, os.path.join(save_dir, id_name, "T1.nii.gz"))
        sitk.WriteImage(t2_new, os.path.join(save_dir, id_name, "T2.nii.gz"))
        sitk.WriteImage(ce_new, os.path.join(save_dir, id_name, "T1CE.nii.gz"))
        sitk.WriteImage(flair_new, os.path.join(save_dir, id_name, "B1400.nii.gz"))
        sitk.WriteImage(seg_new, os.path.join(save_dir, id_name, "T1CE_body_mask.nii.gz"),)
        # print(id_name, "finished", end='')
