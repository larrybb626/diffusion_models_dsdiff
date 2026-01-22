# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：for_nnUNet.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/11/5 15:58 
"""
import os
import shutil

import SimpleITK as sitk
import cv2
import nibabel as nib
import numpy as np


def transfer_file():
    path = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts/'
    save_dir = '/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset506_prostate_ce/imagesTs'
    for i in os.listdir(path):
        p = os.path.join(path, i, 'T1CE.nii.gz')
        target = os.path.join(save_dir, i + '_0000.nii.gz')
        print(target)
        # 把p复制到save_dir
        shutil.copy(p, target)


    # nnUNetv2_predict -i /data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/CE_tr -o /data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/prostate_mask -d 506 -c 2d


def transfer_mask_file():
    path = '/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset506_prostate_ce/labelsTs'
    save_dir = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts/'
    for i in os.listdir(path):
        if not i.endswith('nii.gz'):
            continue
        p = os.path.join(path, i)
        target = os.path.join(save_dir, i.split('.')[0], 'prostate.nii.gz')
        print(target)
        # p复制到save_dir
        shutil.copy(p, target)


def fill_hole_with_cv2():
    path = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts/'
    for i in os.listdir(path):
        mask_temp = sitk.ReadImage(os.path.join(path, i, 'CE_mask.nii.gz'))
        mask = fill_inter_3D(mask_temp)
        mask = sitk.GetImageFromArray(mask)
        mask.CopyInformation(mask_temp)
        sitk.WriteImage(mask, os.path.join(path, i, 'CE_mask.nii.gz'))
        print(i)


def fill_inter_bone(mask):
    # 对一张图像做孔洞填充，读入的是一层
    mask = mask_fill = mask.astype(np.uint8)
    if np.sum(mask[:]) != 0:  # 即读入图层有值
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)
        mask_fill = sum(contour_list)
        mask_fill[mask_fill >= 1] = 1
    return mask_fill.astype(np.uint8)


def fill_inter_3D(mask, other_axis=True):
    # 对3D图像做孔洞填充，即三个维度的fill_inter_bone
    if not isinstance(mask, np.ndarray):
        mask = sitk.GetArrayFromImage(mask)
    mask_final = mask.copy()
    for i in range(mask.shape[0]):
        if np.max(mask[i, :, :]) > 0:
            mask_final[i, :, :] = fill_inter_bone(mask_final[i, :, :])
    if other_axis:
        for i in range(mask.shape[1]):
            if np.max(mask[:, i, :]) > 0:
                mask_final[:, i, :] = fill_inter_bone(mask_final[:, i, :])
        for i in range(mask.shape[2]):
            if np.max(mask[:, :, i]) > 0:
                mask_final[:, :, i] = fill_inter_bone(mask_final[:, :, i])
    return mask_final.astype(np.uint8)


def TotalSegment():
    from totalsegmentator.python_api import totalsegmentator
    dir = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts/'
    for i in os.listdir(dir)[:1]:
        input_path = os.path.join(dir, i, 'T1CE.nii.gz')
        input_image = nib.load(input_path)
        # input_image = sitk.ReadImage(input_path)
        output_path = os.path.join(dir, i, 'whole_body_mask')
        # option 1: provide input and output as file paths
        a = totalsegmentator(input_image, output_path, task='total_mr', device='gpu:1')
        nib.save(a, os.path.dirname(output_path) + '/total_mask_1.nii.gz')
        print('done')


def TotalSegment_new():
    from mrsegmentator import inference
    import os

    # dir = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/'
    # for i in os.listdir(os.path.join(dir,'images_tr')):
    #     input_path = os.path.join(dir, 'images_tr', i, 'T1CE.nii.gz')
    #     if not os.path.isfile(input_path):
    #         continue
    #     target_path = os.path.join(dir, 'CE_tr', i+'_0000.nii.gz')
    #     os.makedirs(os.path.join(dir, 'CE_tr'), exist_ok=True)
    #     shutil.copy(input_path, target_path)


    # for i in os.listdir(dir)[:20]:
    input_path = [os.path.join(dir, 'CE_tr', i) for i in os.listdir(os.path.join(dir, 'CE_tr'))[:]]
    output_path = os.path.join(dir, 'mr_mask_tr')
    # option 1: provide input and output as file paths
    inference.infer(input_path, output_path,batchsize=8)


def mask_move():
    dir = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/mr_mask_tr'
    mask_save_dir = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_tr'
    for i in os.listdir(dir):
        input_path = os.path.join(dir, i)
        if not os.path.isfile(input_path):
            continue
        image = sitk.ReadImage(input_path)
        mask = sitk.GetArrayFromImage(image)
        mask[mask != 24] = 0
        mask[mask == 24] = 1
        mask = sitk.GetImageFromArray(mask)
        mask.CopyInformation(image)
        target = os.path.join(mask_save_dir, i.split('_')[0], 'bladder.nii.gz')
        sitk.WriteImage(mask, target)
        print(input_path)
        # 把p复制到save_dir


def move_bladder_to_nnunet():
    dir = '/home/user4/sharedata/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_tr/'
    save_dir = '/home/user4/sharedata/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset505_bladder/imagesTr'
    mask_save_dir = '/home/user4/sharedata/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset505_bladder/labelsTr'
    for i in os.listdir(dir)[:24]:
        input_path = os.path.join(dir, i, 'bladder.nii.gz')
        if not os.path.isfile(input_path):
            continue
        target = os.path.join(mask_save_dir, i + '.nii.gz')
        print(target)
        t1_path = os.path.join(dir, i, 'T1CE.nii.gz')
        target_T1ce = os.path.join(save_dir, i + '_0000.nii.gz')
        # 把p复制到save_dir
        os.makedirs(mask_save_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy(input_path, target)
        shutil.copy(t1_path, target_T1ce)

def move_prostate_to_nnunet():
    mask_save_dir = '/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset506_prostate_ce/labelsTr'
    ce = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/CE_tr'
    dir = '/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset506_prostate_ce/imagesTr'
    for i in os.listdir(mask_save_dir):
        input_path = os.path.join(ce, i.split('.')[0] + '_0000.nii.gz')
        if not os.path.isfile(input_path):
            continue
        target = os.path.join(dir, i.split('.')[0] + '_0000.nii.gz')
        print(target)
        # 把p复制到save_dir
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(input_path, target)

def move_img_to_nnunet_according_to_label():
    maskdir = '/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset506_prostate_ce/labelsTr'
    ce = '/data/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/CE_tr'
    dir = '/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset506_prostate_ce/imagesTr'
    for i in os.listdir(maskdir):
        input_path = os.path.join(ce, i.split('.')[0] + '_0000.nii.gz')
        if not os.path.isfile(input_path):
            continue
        target = os.path.join(dir, i.split('.')[0] + '_0000.nii.gz')
        print(target)
        # 把p复制到save_dir
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(input_path, target)

def move_brats():
    # 原始数据目录
    source_dir = "/data/newnas_1/MJY_file/BraTS_dataset/data_pre_original/images_tr"
    # 目标 imagesTr 目录
    target_images_dir = "/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset508_brats_ce/imagesTr"
    # 目标 labelsTr 目录
    target_labels_dir = "/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset508_brats_ce/labelsTr"

    # 确保目标目录存在
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)

    # 遍历 source_dir 下的所有子文件夹
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        # 确保是文件夹
        if not os.path.isdir(folder_path):
            continue

        # 定义文件路径（按顺序）
        t1_path = os.path.join(folder_path, "t1.nii.gz")
        t2_path = os.path.join(folder_path, "t2.nii.gz")
        flair_path = os.path.join(folder_path, "flair.nii.gz")
        ce_path = os.path.join(folder_path, "ce.nii.gz")
        seg_path = os.path.join(folder_path, "seg.nii.gz")  # 分割标签

        # 目标文件名
        t1_target = os.path.join(target_images_dir, f"{folder_name}_0000.nii.gz")
        t2_target = os.path.join(target_images_dir, f"{folder_name}_0001.nii.gz")
        flair_target = os.path.join(target_images_dir, f"{folder_name}_0002.nii.gz")
        ce_target = os.path.join(target_images_dir, f"{folder_name}_0003.nii.gz")
        seg_target = os.path.join(target_labels_dir, f"{folder_name}.nii.gz")

        # 复制影像数据
        for src, dst in [(t1_path, t1_target), (t2_path, t2_target), (flair_path, flair_target), (ce_path, ce_target)]:
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {src} -> {dst}")
            else:
                print(f"Warning: {src} not found!")

        # 复制标签数据（如果存在）
        if os.path.exists(seg_path):
            seg_img = sitk.ReadImage(seg_path)
            seg_array = sitk.GetArrayFromImage(seg_img)

            # 处理标签：
            # - 1 和 3 合并为 1
            # - 2 置为 0
            seg_array[seg_array == 3] = 1
            seg_array[seg_array == 2] = 0

            # 重新存回 SimpleITK 格式
            new_seg_img = sitk.GetImageFromArray(seg_array)
            new_seg_img.CopyInformation(seg_img)  # 保持空间信息
            sitk.WriteImage(new_seg_img, seg_target)

            print(f"Processed & Saved {seg_target}")
            # print(f"Copied {seg_path} -> {seg_target}")
        else:
            print(f"Warning: {seg_path} not found!")

def process_lots_of_brats():
    all_pred = ['/data/newnas_1/MJY_file/SOTA_models/mulD_RegGAN/braCE_MRI_simulate_PCa_64_pix2pix_mulD_fold5-1/pred_nii',
                '/data/newnas_1/MJY_file/SOTA_models/SOTA_GAN/ResVit_brats/result',
                '/data/newnas_1/MJY_file/SOTA_models/DiscDiff_test/BraTs_v_predict_7e4_uknow/itk_save_dir',
                '/data/newnas_1/MJY_file/SOTA_models/controlnet-model-brats/controlnet_result/checkpoint-50000/itk_result',
                '/data/newnas_1/MJY_file/diffusion/train_result_BraTs/BraTs_synthesis_74_ds_diff_fold5-1/pred_nii_ddim_20_eta0_checkpoint',
                ]
    models = ['cGAN', 'ResViT', 'DisC-Diff', 'SD3', 'DS-Diff', 'Real']
    # 原始数据目录
    source_dir = "/data/newnas_1/MJY_file/BraTS_dataset/data_pre_original/images_ts"
    # 目标 imagesTr 目录
    target_images_dir = "/data/newnas_1/MJY_file/BraTS_dataset_Seg/"
     # 确保目标目录存在
    for model in models:
        # 遍历 source_dir 下的所有子文件夹
        target_dir = os.path.join(target_images_dir, model)
        os.makedirs(target_dir, exist_ok=True)
        for folder_name in os.listdir(source_dir):
            folder_path = os.path.join(source_dir, folder_name)
            # 确保是文件夹
            if not os.path.isdir(folder_path):
                continue
            # 定义文件路径（按顺序）
            t1_path = os.path.join(folder_path, "t1.nii.gz")
            t2_path = os.path.join(folder_path, "t2.nii.gz")
            flair_path = os.path.join(folder_path, "flair.nii.gz")
            if model == 'Real':
                ce_path = os.path.join(folder_path, "ce.nii.gz")
            else:
                ce_path = os.path.join(all_pred[models.index(model)], folder_name + '_pred.nii.gz')
                if not os.path.isfile(ce_path):
                    ce_path = os.path.join(all_pred[models.index(model)], folder_name + '.nii.gz')
            # 目标文件名
            t1_target = os.path.join(target_dir, f"{folder_name}_0000.nii.gz")
            t2_target = os.path.join(target_dir, f"{folder_name}_0001.nii.gz")
            flair_target = os.path.join(target_dir, f"{folder_name}_0002.nii.gz")
            ce_target = os.path.join(target_dir, f"{folder_name}_0003.nii.gz")
            print(f"Copied {ce_path} -> {ce_target}")
            # 复制影像数据
            for src, dst in [(t1_path, t1_target), (t2_path, t2_target), (flair_path, flair_target), (ce_path, ce_target)]:
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    # print(f"Copied {src} -> {dst}")

                else:
                    print(f"Warning: {src} not found!")

def itk_resample(moving, target, resamplemethod=sitk.sitkLinear):
    #初始化一个列表
    target_Size = [0, 0, 0]
    target_Size = target.GetSize()
    # 读取原始图像的spacing和size
    ori_size = moving.GetSize()
    ori_spacing = moving.GetSpacing()
    # 读取重采样的参数
    target_Spacing = target.GetSpacing()
    # 方向和origin不必变动
    target_direction = moving.GetDirection()
    target_origin = moving.GetOrigin()
    # 获取重采样的图像大小
    # target_Size[0] = round(ori_size[0] * ori_spacing[0] / target_Spacing[0])
    # target_Size[1] = round(ori_size[1] * ori_spacing[1] / target_Spacing[1])
    # target_Size[2] = round(ori_size[2] * ori_spacing[2] / target_Spacing[2])
    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(target)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    resampler.SetOutputPixelType(sitk.sitkUInt8)  # 线性插值是用于PET/CT/MRI之类的，所以保存float32格式
    resampler.SetTransform(sitk.Transform(3, sitk.sitkAffine))  # 3, sitk.sitkIdentity这个参数的用处还不确定
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(moving)  # 得到重新采样后的图像
    return itk_img_resampled

def resample():
    dir = '/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset506_prostate_ce/labelsTr'
    target_dir = '/data/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset506_prostate_ce/imagesTr'
    for i in os.listdir(dir):
        input_path = os.path.join(dir, i)
        if not os.path.isfile(input_path):
            continue
        image = sitk.ReadImage(input_path)
        target = os.path.join(target_dir, i.split('.')[0] + '_0000.nii.gz')
        target = sitk.ReadImage(target)
        resample = itk_resample(image, target)
        save = os.path.join(dir, i)
        sitk.WriteImage(resample, save)
        print(input_path)

if __name__ == '__main__':
    move_brats()