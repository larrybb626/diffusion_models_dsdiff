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

if __name__ == "__main__":
    save_dir = "/home/user15/sharedata/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm/images_ts_lh"
    save_dir_h5 = "/home/user15/sharedata/newnas/MJY_file/CE-MRI/PCa_new/h5_data_2d_pre_320320_01norm/images_ts_lh"
    all_id = os.listdir(save_dir)
    for id_name in tqdm(all_id, desc="Processing"):
        t1_path = os.path.join(save_dir, id_name, 'T1.nii.gz')
        t2_path = os.path.join(save_dir, id_name, 'T2.nii.gz')
        flair_path = os.path.join(save_dir, id_name, 'B1400.nii.gz')
        ce_path = os.path.join(save_dir, id_name, 'T1CE.nii.gz')
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
        flair = sitk.GetArrayFromImage(sitk.ReadImage(flair_path))
        ce = sitk.GetArrayFromImage(sitk.ReadImage(ce_path))
        mask = np.zeros_like(t1)
        os.makedirs(os.path.join(save_dir_h5, id_name), exist_ok=True)
        for layer in range(len(t1)):
            with h5py.File(os.path.join(save_dir_h5, id_name, 'layer_{}.h5'.format(layer)), 'w') as f:
                f['t1'] = t1[layer]
                f['t2'] = t2[layer]
                f['b1500'] = flair[layer]
                f['t1ce'] = ce[layer]
                f['mask'] = mask[layer]
                f.close()
            # 改用npz来存储
            # np.savez(os.path.join(save_dir_h5, id_name, 'layer{}.npz'.format(layer)), t1=t1[layer], t2=t2[layer], flair=flair[layer], t1ce=ce[layer], mask=mask[layer])
