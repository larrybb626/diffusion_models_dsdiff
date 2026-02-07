"""
对nii_data_pre图像做归一化和删除缺块的切片保存到新的文件夹

"""
import numpy as np
import SimpleITK as sitk
import os
import glob
# import h5py
import pandas as pd


# def find_valid_slice(array, mask):
#     """
#     non-zero area > 50%
#     :return:
#     """
#     non_zero_array = array > 0
#     valid_list = []
#     for i in range(array.shape[0]):
#         ratio = non_zero_array[i].sum() / (mask[i].sum() + 1)
#         if ratio > 0.5:
#             valid_list.append(i)
#     return np.array(valid_list)


if __name__ == "__main__":
    nii_pre_path = r'/nas_3/LaiRuiBin/Changhai/data/resampled/SSA/images_ts'
    nii_list = [item for item in glob.glob(nii_pre_path + "/*") if os.path.isdir(item)]

    mode = "MinMax"
    new_pre_path = r'/nas_3/LaiRuiBin/Changhai/data/normalization/SSA/images_ts'
    # if mode == "stdnorm":
    #     new_pre_path = r'/nas_3/LaiRuiBin/Changhai/data/normalization/SSA/images_tr'
    # else:
    #     new_pre_path = r'/nas_3/LaiRuiBin/Changhai/data/normalization/SSA/images_tr'
    print(new_pre_path)

    if not os.path.exists(new_pre_path):
        os.makedirs(new_pre_path)

    for idx, nii_folder in enumerate(nii_list):
        id_num = os.path.basename(nii_folder)

        try:
            # 读取图像
            F_Data1 = os.path.join(nii_folder, "F_Data1.nii.gz")
            F_Data2 = os.path.join(nii_folder, "F_Data2.nii.gz")
            S_Data1 = os.path.join(nii_folder, "S_Data1.nii.gz")
            S_Data2 = os.path.join(nii_folder, "S_Data2.nii.gz")

            file_list = [F_Data1, F_Data2, S_Data1, S_Data2]

            # --- 提前创建好该病人的保存目录，不用在循环里反复判断 ---
            save_folder = os.path.join(new_pre_path, id_num)
            os.makedirs(save_folder, exist_ok=True)

            for nii in file_list:
                img_o = sitk.ReadImage(nii)
                img = sitk.GetArrayFromImage(img_o)

                # 数据处理逻辑...
                if os.path.basename(nii) != "":
                    if mode == "stdnorm":
                        mean = img[img != 0].mean()
                        std = img[img != 0].std()
                        img = (img - mean) / std
                    else:
                        upper_1onk = img.max() * 0.75
                        img[img > upper_1onk] = upper_1onk
                        img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1
                    # img = img * body_mask
                # img = img[start_point:end_point + 1]
                img = sitk.GetImageFromArray(img)
                img.SetOrigin(img_o.GetOrigin())
                img.SetSpacing(img_o.GetSpacing())
                img.SetDirection(img_o.GetDirection())

                # --- 修正后的路径逻辑 ---
                # 无论文件夹是否存在，都要定义 new_nii
                new_nii = os.path.join(save_folder, os.path.basename(nii))

                sitk.WriteImage(img, new_nii)
            print('\r' + str(idx + 1) + '/' + str(len(nii_list)), id_num, end='', flush=True)
        except Exception as e:
            print("\n error in {}, ".format(id_num), e)
