import os
import SimpleITK as sitk

# 路径配置
src_folder = r'/nas_3/LaiRuiBin/Changhai/data/normalization/SSA/images_ts'
dst_folder = r'/nas_3/LaiRuiBin/Changhai/data/normalization/SSA/images_ts_256'
TARGET_SIZE_2D = [256, 256]


def get_256_reference(original_ref_img):
    """
    创建一个基于原图物理坐标、但像素尺寸强制为 256x256 的模板
    """
    old_size = original_ref_img.GetSize()
    old_spacing = original_ref_img.GetSpacing()

    # 强制 XY 为 256，Z 保持原样
    new_size = [TARGET_SIZE_2D[0], TARGET_SIZE_2D[1], old_size[2]]

    # 计算新的 XY Spacing 保证物理尺寸不变
    new_spacing = [
        old_size[0] * old_spacing[0] / new_size[0],
        old_size[1] * old_spacing[1] / new_size[1],
        old_spacing[2]
    ]

    # 创建一个空的 SimpleITK 图像作为模板
    ref_256 = sitk.Image(new_size, original_ref_img.GetPixelIDValue())
    ref_256.SetOrigin(original_ref_img.GetOrigin())
    ref_256.SetDirection(original_ref_img.GetDirection())
    ref_256.SetSpacing(new_spacing)
    return ref_256


def resample_to_ref(moving_img, ref_img, is_mask=False):
    """
    将图像重采样到参考模板的网格上
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)  # 核心：强制对齐到模板的 Size, Spacing, Origin, Direction

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # 归一化后的数据建议用线性插值，BSpline 在层数跨度太大时可能会产生伪影
        resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    return resampler.Execute(moving_img)


if __name__ == '__main__':
    os.makedirs(dst_folder, exist_ok=True)
    patient_ids = sorted(os.listdir(src_folder))

    for i, pid in enumerate(patient_ids, 1):
        p_src_path = os.path.join(src_folder, pid)
        if not os.path.isdir(p_src_path): continue

        # 1. 尝试读取 F_Data1 作为该病人的物理基准
        ref_path = os.path.join(p_src_path, 'F_Data1.nii.gz')
        if not os.path.exists(ref_path):
            print(f"\n跳过 {pid}：缺失基准文件 F_Data1")
            continue

        original_f1 = sitk.ReadImage(ref_path)

        # 2. 生成该病人的 256 物理模板
        target_ref_256 = get_256_reference(original_f1)

        # 3. 处理该病人的所有文件
        p_dst_path = os.path.join(dst_folder, pid)
        os.makedirs(p_dst_path, exist_ok=True)

        for filename in ['F_Data1.nii.gz', 'F_Data2.nii.gz', 'S_Data1.nii.gz', 'S_Data2.nii.gz']:
            img_path = os.path.join(p_src_path, filename)
            if os.path.exists(img_path):
                img = sitk.ReadImage(img_path)
                # 将所有序列都对齐到这个病人的 256 模板上
                res_img = resample_to_ref(img, target_ref_256)
                sitk.WriteImage(res_img, os.path.join(p_dst_path, filename))

        print(f"\r进度: {i}/{len(patient_ids)} - {pid} 已物理对齐并统一为 256", end='')

    print("\n✅ 完成！现在所有病人的 4 个序列不仅长宽是 256，且解剖结构已经物理对齐。")