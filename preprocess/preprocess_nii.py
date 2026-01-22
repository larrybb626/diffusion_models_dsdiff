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
    phase = ['train', 'val']
    crop_size = 192  #裁剪224
    for p in phase:
        print('go ', p)
        dir = "/data/newnas_1/MJY_file/BraTS_dataset/BraSyn_2024" + "/{}_data".format(p)
        postfix = 'val' if p == 'val' else ('tr' if p == 'train' else 'ts')
        save_dir = "/data/newnas_1/MJY_file/BraTS_dataset/data_pre_original/images_{}".format(postfix)
        all_id = os.listdir(dir)
        for id_name in tqdm.tqdm(all_id, desc="Processing {}".format(p)):
            ce_path, seg_path, t1_path, t2_path, flair_path = None, None, None, None, None
            path = os.path.join(dir, id_name)
            for file in os.listdir(path):
                if 't1c' in file:
                    ce_path = os.path.join(path, file)
                elif 'seg' in file:
                    seg_path = os.path.join(path, file)
                elif 't1' in file:
                    t1_path = os.path.join(path, file)
                elif 'flair' in file or 't2f' in file:
                    flair_path = os.path.join(path, file)
                elif 't2w' in file or 't2' in file:
                    t2_path = os.path.join(path, file)
                else:
                    raise ValueError("Unknown file")
            # 读取数据
            assert ce_path and seg_path and t1_path and t2_path and flair_path
            t1 = sitk.ReadImage(t1_path, sitk.sitkFloat32)
            t2 = sitk.ReadImage(t2_path, sitk.sitkFloat32)
            ce = sitk.ReadImage(ce_path, sitk.sitkFloat32)
            flair = sitk.ReadImage(flair_path, sitk.sitkFloat32)
            seg = sitk.ReadImage(seg_path)
            t1_array = sitk.GetArrayFromImage(t1)
            t2_array = sitk.GetArrayFromImage(t2)
            ce_array = sitk.GetArrayFromImage(ce)
            flair_array = sitk.GetArrayFromImage(flair)
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array_tmp = seg_array.sum(axis=1).sum(axis=1)
            non_zero = np.nonzero(seg_array_tmp)[0]
            # seg_array 第0维有数值的地方bigmask为1
            # if p != 'train':
            min_idx, max_idx = int(min(non_zero)), int(max(non_zero)) + 1
            start_h = (t1_array.shape[-2] - crop_size) // 2
            start_w = (t1_array.shape[-1] - crop_size) // 2
            t1_array = t1_array[min_idx:max_idx, start_h:start_h + crop_size, start_w:start_w + crop_size]
            t2_array = t2_array[int(min(non_zero)):max_idx, start_h:start_h + crop_size, start_w:start_w + crop_size]
            ce_array = ce_array[int(min(non_zero)):max_idx, start_h:start_h + crop_size, start_w:start_w + crop_size]
            flair_array = flair_array[int(min(non_zero)):max_idx, start_h:start_h + crop_size, start_w:start_w + crop_size]
            seg_array = seg_array[int(min(non_zero)):max_idx, start_h:start_h + crop_size, start_w:start_w + crop_size]
            # non_zero_image = np.nonzero(t1_array.sum(1).sum(1))
            # t1_array = t1_array[non_zero_image[0][0]:non_zero_image[0][-1]+1]
            # t2_array = t2_array[non_zero_image[0][0]:non_zero_image[0][-1]+1]
            # ce_array = ce_array[non_zero_image[0][0]:non_zero_image[0][-1]+1]
            # flair_array = flair_array[non_zero_image[0][0]:non_zero_image[0][-1]+1]
            # seg_array = seg_array[non_zero_image[0][0]:non_zero_image[0][-1]+1]
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
            # 调spacing
            # t1_new = resample_image(t1_new, (t1_spacing[0], t1_spacing[1], t1_spacing[2]*3))
            # t2_new = resample_image(t2_new, (t2_spacing[0], t2_spacing[1], t2_spacing[2]*3))
            # ce_new = resample_image(ce_new, (ce_spacing[0], ce_spacing[1], ce_spacing[2]*3))
            # flair_new = resample_image(flair_new, (flair_spacing[0], flair_spacing[1], flair_spacing[2]*3))
            # seg_new = resample_image(seg_new, (seg_spacing[0], seg_spacing[1], seg_spacing[2]*3), sitk.sitkNearestNeighbor)
            seg_new = sitk.Cast(seg_new, sitk.sitkUInt8)
            os.makedirs(os.path.join(save_dir, id_name), exist_ok=True)
            sitk.WriteImage(t1_new, os.path.join(save_dir, id_name, "t1.nii.gz"))
            sitk.WriteImage(t2_new, os.path.join(save_dir, id_name, "t2.nii.gz"))
            sitk.WriteImage(ce_new, os.path.join(save_dir, id_name, "ce.nii.gz"))
            sitk.WriteImage(flair_new, os.path.join(save_dir, id_name, "flair.nii.gz"))
            sitk.WriteImage(seg_new, os.path.join(save_dir, id_name, "seg.nii.gz"),)
            # print(id_name, "finished", end='')
