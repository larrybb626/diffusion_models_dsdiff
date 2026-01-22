# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：mri_dataset.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/1/10 17:41 
"""
import os
import sys

import numpy as np
from monai.data import Dataset, CacheDataset, decollate_batch, DataLoader, pad_list_data_collate
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

from training_project.training_transform import *


class MriBraTSData:
    def __init__(self,config):
        self.dataset_type = config.dataset_type
        self.conclude_test = True
        self.random_state = config.seed
        self.random_prob = config.augmentation_prob
        self.num_workers = config.num_workers
        self.keys = config.train_keys
        self.data_dir = config.h5_2d_img_dir
        self.fold_K = config.fold_K
        self.fold_idx = config.fold_idx
        self.train_batch_size = config.train_batch_size
        self.val_batch_size = config.val_batch_size
        self.test_batch_size = None
        self.train_dir = os.path.join(self.data_dir, "images_tr")
        self.val_dir = os.path.join(self.data_dir, "images_val")
        self.test_dir = os.path.join(self.data_dir, "images_ts")
        self.record_file = os.path.join(config.root_dir, "log_txt.txt")
        self.init_all_loader()
    def get_dataset(self, data_list, transform, mode="train", dataset_type="normal"):
        """
        :param data_list:
        :param transform:
        :param mode: "train" or "val"
        :param dataset_type: "normal" or "cache"
        :return:
        """
        if mode == "train":
            if dataset_type == "normal":
                self.train_ds = Dataset(
                    data=data_list,
                    transform=transform,
                )
            elif dataset_type == "cache":
                self.train_ds = CacheDataset(
                    data=data_list,
                    transform=transform,
                    # cache_num=300,
                    cache_rate=0,
                    num_workers=self.num_workers,
                )
        elif mode == "val":
            if dataset_type == "normal":
                self.val_ds = Dataset(
                    data=data_list,
                    transform=transform
                )
            elif dataset_type == "cache":
                self.val_ds = CacheDataset(
                    data=data_list,
                    transform=transform,
                    # cache_num=100,
                    cache_rate=0,
                    num_workers=self.num_workers,
                )
        elif mode == "test":
            # if dataset_type == "normal":
            self.test_ds = Dataset(
                data=data_list,
                transform=transform
            )
            # elif dataset_type == "cache":
            #     self.test_ds = CacheDataset(
            #         data=data_list,
            #         transform=transform,
            #         # cache_num=100,
            #         cache_rate=1,
            #         num_workers=self.num_workers,
            #     )

    def do_split(self, K, fold):
        """
        :reg_param K: 分几折
        :reg_param fold: 第几折，从1开始
        :return:分折的病人id列表[train,val]
        """
        fold_train = []
        fold_test = []

        kf = KFold(n_splits=K, random_state=self.random_state, shuffle=True)
        id_list = sorted(os.listdir(self.train_dir))
        for train_index, test_index in kf.split(id_list):
            fold_train.append(np.array(id_list)[train_index])
            fold_test.append(np.array(id_list)[test_index])

        train_id = fold_train[fold - 1]
        test_id = fold_test[fold - 1]
        self.print_to_txt(f'train_id:{len(train_id)}||valid_id:{len(test_id)}')
        if self.conclude_test:
            train_id = np.stack([train_id, test_id], axis=0)
        return [train_id, test_id]

    def get_data_dict(self, id_list):
        # 输入id的list获取数据字典
        data_dict = []
        for id_num in id_list:
            layer_list = sorted(os.listdir(os.path.join(self.train_dir, id_num)))
            # layer_list = [layer_list] * 4
            for layer in layer_list:  # 头尾不要?
                new_data_dict = {"path": os.path.join(self.train_dir, id_num, layer), "txt": ""}
                data_dict.append(new_data_dict)
        return data_dict
    def init_all_loader(self):
        train_transforms = get_2d_train_transform_diff(keys=self.keys, random_prob=self.random_prob)
        val_transforms = get_2d_val_transform_diff(keys=self.keys)
        test_transforms = get_2d_test_transform(keys=self.keys)
        datasets = self.do_split(self.fold_K, self.fold_idx)
        self.get_dataset(self.get_data_dict(datasets[0]), train_transforms, mode="train", dataset_type=self.dataset_type)
        self.get_dataset(self.get_data_dict(sorted(os.listdir(self.val_dir))), val_transforms, mode="val", dataset_type=self.dataset_type)
        self.get_dataset(self.get_data_dict(sorted(os.listdir(self.test_dir))), test_transforms, mode="test", dataset_type=self.dataset_type)
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers if sys.gettrace() is None else 1,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return val_loader

    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate,
        )
        return pred_loader

    def print_to_txt(self, args):
        print(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()






