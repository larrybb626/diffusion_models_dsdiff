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
    phase = ['train', 'val']
    for p in phase:
        print('go ', p)
        postfix = 'val' if p == 'val' else ('tr' if p == 'train' else 'ts')
        save_dir = "/data/newnas_1/MJY_file/BraTS_dataset/data_pre_original/images_{}".format(postfix)
        save_dir_h5 = "/data/newnas_1/MJY_file/BraTS_dataset/data_pre_h5_original/images_{}".format(postfix)
        all_id = os.listdir(save_dir)
        for id_name in tqdm(all_id, desc="Processing {}".format(p)):
            t1_path = os.path.join(save_dir, id_name, 't1.nii.gz')
            t2_path = os.path.join(save_dir, id_name, 't2.nii.gz')
            flair_path = os.path.join(save_dir, id_name, 'flair.nii.gz')
            ce_path = os.path.join(save_dir, id_name, 'ce.nii.gz')
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
                    f['flair'] = flair[layer]
                    f['t1ce'] = ce[layer]
                    f['mask'] = mask[layer]
                    f.close()
                # 改用npz来存储
                # np.savez(os.path.join(save_dir_h5, id_name, 'layer{}.npz'.format(layer)), t1=t1[layer], t2=t2[layer], flair=flair[layer], t1ce=ce[layer], mask=mask[layer])
