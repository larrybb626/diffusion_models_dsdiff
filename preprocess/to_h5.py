# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：to_h5.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/10/24 20:18 
"""
import os

import SimpleITK as sitk
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    phase = ['train', 'test']
    for p in phase:
        print('go ', p)
        postfix = 'val' if p == 'val' else ('tr' if p == 'train' else 'ts')
        save_dir = "/nas_3/LaiRuiBin/Changhai/data/normalization/SSA/images_{}".format(postfix)
        save_dir_h5 = "/nas_3/LaiRuiBin/Changhai/data/H5_new/SSA/images_{}".format(postfix)
        all_id = os.listdir(save_dir)
        for id_name in tqdm(all_id, desc="Processing {}".format(p)):

            t1_path = os.path.join(save_dir, id_name, 'F_Data1.nii.gz')
            t2_path = os.path.join(save_dir, id_name, 'F_Data2.nii.gz')
            S_Data1 = os.path.join(save_dir, id_name, 'S_Data1.nii.gz')
            S_Data2 = os.path.join(save_dir, id_name, 'S_Data2.nii.gz')

            F_Data1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
            F_Data2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
            S_Data1 = sitk.GetArrayFromImage(sitk.ReadImage(S_Data1))
            S_Data2 = sitk.GetArrayFromImage(sitk.ReadImage(S_Data2))

            os.makedirs(os.path.join(save_dir_h5, id_name), exist_ok=True)

            for layer in range(len(F_Data1)):
                with h5py.File(os.path.join(save_dir_h5, id_name, 'layer_{}.h5'.format(layer)), 'w') as f:
                    f['F_Data1'] = F_Data1[layer]
                    # # ========= debug =========
                    # a = F_Data1[5, :, :]
                    # plt.imshow(a)
                    # plt.show()
                    # ========= debug =========
                    f['F_Data2'] = F_Data2[layer]
                    f['S_Data1'] = S_Data1[layer]
                    f['S_Data2'] = S_Data2[layer]
                    # f['mask'] = mask[layer]
                    f.close()
                # 改用npz来存储
                # np.savez(os.path.join(save_dir_h5, id_name, 'layer{}.npz'.format(layer)), F_Data1=F_Data1[layer], F_Data2=F_Data2[layer], S_Data1=S_Data1[layer]
                # , S_Data2=S_Data2[layer])
