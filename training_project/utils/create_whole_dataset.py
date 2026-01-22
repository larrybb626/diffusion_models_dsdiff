# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：create_whole_dataset.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/5/27 20:52 
"""
import os

import h5py
import numpy as np
import tqdm
from sklearn.model_selection import KFold

from Disc_diff.guided_diffusion.image_datasets import dataset_config


def do_split(K, fold, train_dir):
    """
    :reg_param K: 分几折
    :reg_param fold: 第几折，从1开始
    :return:分折的病人id列表[train,val]
    """
    fold_train = []
    fold_test = []

    kf = KFold(n_splits=K, random_state=2024, shuffle=True)
    id_list = sorted(os.listdir(train_dir))
    for train_index, test_index in kf.split(id_list):
        fold_train.append(np.array(id_list)[train_index])
        fold_test.append(np.array(id_list)[test_index])

    train_id = fold_train[fold - 1]
    test_id = fold_test[fold - 1]
    print(f'train_id:{len(train_id)}||valid_id:{len(test_id)}')
    return [train_id, test_id]


def process_data(id_list, train_dir, save_dir, phase):
    ce_data, t1_data, t2_data, dwi_data = [], [], [], []
    for i in tqdm.tqdm(id_list):
        p = os.path.join(train_dir, str(i))
        for j in os.listdir(p):
            with h5py.File(os.path.join(p, j), 'r') as f:
                ce_data.append(f['t1ce'][()])
                t1_data.append(f['t1'][()])
                t2_data.append(f['t2'][()])
                dwi_data.append(f['flair'][()])
    ce_data, t1_data, t2_data, dwi_data = (np.stack(ce_data, axis=0),
                                           np.stack(t1_data, axis=0),
                                           np.stack(t2_data, axis=0),
                                           np.stack(dwi_data, axis=0)
                                           )
    # save as npy
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{phase}_ce_data.npy"), ce_data)
    np.save(os.path.join(save_dir, f"{phase}_t1_data.npy"), t1_data)
    np.save(os.path.join(save_dir, f"{phase}_t2_data.npy"), t2_data)
    np.save(os.path.join(save_dir, f"{phase}_dwi_data.npy"), dwi_data)


if __name__ == "__main__":
    print("hello world")
    save_dir = "/data/newnas_1/MJY_file/BraTs_dataset_npy/"
    data_config = {
        "K": 5,
        "random_state": 2024,
        "train_dir": "/data/newnas_1/MJY_file/BraTS_dataset/data_pre_h5/images_tr",
        "fold": 1
    }
    data_config = dataset_config(**data_config)
    phase = ['train']
    id_list = sorted(os.listdir(data_config.train_dir)[:])
    for p in phase:
        # id_list = do_split(data_config.K, data_config.fold, data_config.train_dir)[0 if p == 'train' else 1]
        process_data(id_list, data_config.train_dir, save_dir, p)
