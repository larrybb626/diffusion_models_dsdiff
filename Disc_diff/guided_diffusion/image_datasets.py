import os

import SimpleITK as sitk
import numpy as np
import torch
import tqdm
from skimage.measure import shannon_entropy
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


def load_data(
        *,
        hr_data_dir,
        lr_data_dir,
        other_data_dir,
        deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param lr_data_dir:
    :param other_data_dir:
    :param hr_data_dir:
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    return BraTSMRI(hr_data_dir, lr_data_dir, other_data_dir)


def _load_prostate_data(
        *,
        dataset_config
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param lr_data_dir:
    :param other_data_dir:
    :param hr_data_dir:
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    return ProstateMRI_with_shannon_entropy(dataset_config)


class BraTSMRI(Dataset):
    def __init__(self, hr_data_name, lr_data_name, other_data_name):
        self.hr_data, self.lr_data, self.other_data = np.load(hr_data_name, mmap_mode="r")[:, 40:60], \
            np.load(lr_data_name, mmap_mode="r")[:, 40:60], \
            np.load(other_data_name, mmap_mode="r")[:, 40:60]
        # T2H T2L T1H
        num_subject, num_slice, h, w = self.hr_data.shape
        self.hr_data = self.hr_data.reshape(num_subject * num_slice, h, w)
        self.lr_data = self.lr_data.reshape(num_subject * num_slice, h, w)
        self.other_data = self.other_data.reshape(num_subject * num_slice, h, w)

        data_dict = {}
        for s in range(len(self.hr_data)):
            entropy = np.round(shannon_entropy(self.hr_data[s]))
            if entropy in data_dict:
                data_dict[entropy].append(s)
            else:
                data_dict[entropy] = [s]

        self.hr_data = torch.from_numpy(self.hr_data).float()
        self.lr_data = torch.from_numpy(self.lr_data).float()
        self.other_data = torch.from_numpy(self.other_data).float()

        self.hr_data = torch.unsqueeze(self.hr_data, 1)
        self.lr_data = torch.unsqueeze(self.lr_data, 1)
        self.other_data = torch.unsqueeze(self.other_data, 1)
        self.data_dict = data_dict
        print(self.hr_data.shape, self.lr_data.shape, self.other_data.shape)

    def __len__(self):
        return self.hr_data.shape[0]

    def __getitem__(self, index):
        return self.hr_data[index], self.lr_data[index], self.other_data[index]


def get_prostate_dataset(dataset_config):
    fold_train = []
    fold_test = []

    kf = KFold(n_splits=dataset_config.K, random_state=dataset_config.random_state, shuffle=True)
    id_list = sorted(os.listdir(dataset_config.train_dir))
    for train_index, test_index in kf.split(id_list):
        fold_train.append(np.array(id_list)[train_index])
        fold_test.append(np.array(id_list)[test_index])

    train_id = fold_train[dataset_config.fold - 1]
    test_id = fold_test[dataset_config.fold - 1]
    print(f'train_id:{len(train_id)}||valid_id:{len(test_id)}')
    return train_id, test_id


class ProstateMRI_with_shannon_entropy(Dataset):
    def __init__(self, dataset_config):
        self.ce_data, self.t1_data, self.t2_data, self.dwi_data = (np.load(dataset_config.train_ce_dir, mmap_mode="r"),
                                                                   np.load(dataset_config.train_t1_dir, mmap_mode="r"),
                                                                   np.load(dataset_config.train_t2_dir, mmap_mode="r"),
                                                                   np.load(dataset_config.train_dwi_dir, mmap_mode="r"))

        data_dict = {}
        for s in tqdm.tqdm(range(len(self.ce_data)), desc="Calculating shannon entropy"):
            entropy = np.round(shannon_entropy(self.ce_data[s]))
            if entropy in data_dict:
                data_dict[entropy].append(s)
            else:
                data_dict[entropy] = [s]

        self.ce_data = torch.from_numpy(self.ce_data).float()
        self.t1_data = torch.from_numpy(self.t1_data).float()
        self.t2_data = torch.from_numpy(self.t2_data).float()
        self.dwi_data = torch.from_numpy(self.dwi_data).float()

        self.ce_data = torch.unsqueeze(self.ce_data, 1)
        self.t1_data = torch.unsqueeze(self.t1_data, 1)
        self.t2_data = torch.unsqueeze(self.t2_data, 1)
        self.dwi_data = torch.unsqueeze(self.dwi_data, 1)

        self.data_dict = data_dict
        print(self.ce_data.shape, self.t1_data.shape, self.t2_data.shape, self.dwi_data.shape)

    def __len__(self):
        return self.ce_data.shape[0]

    def __getitem__(self, index):
        return self.ce_data[index], self.t1_data[index], self.t2_data[index], self.dwi_data[index]


class BraTSMRI(Dataset):
    def __init__(self, dataset_config):
        self.ce_data, self.t1_data, self.t2_data = [], [], []
        train_dir = dataset_config.train_dir
        self.all_train_data = []
        for ids in sorted(os.listdir(train_dir)[:]):
            # ce_dir = os.path.join(train_dir, str(ids), "T1CE.nii.gz")
            # t1_dir = os.path.join(train_dir, str(ids), "T1.nii.gz")
            # t2_dir = os.path.join(train_dir, str(ids), "T2.nii.gz")
            # dwi_dir = os.path.join(train_dir, str(ids), "B1400.nii.gz")

            ce_dir = os.path.join(train_dir, str(ids), "ce.nii.gz")
            t1_dir = os.path.join(train_dir, str(ids), "t1.nii.gz")
            t2_dir = os.path.join(train_dir, str(ids), "t2.nii.gz")
            dwi_dir = os.path.join(train_dir, str(ids), "flair.nii.gz")
            self.all_train_data.append({"ce": ce_dir, "t1": t1_dir, "t2": t2_dir, "dwi": dwi_dir})
        print(len(self.all_train_data), "test_data")

    def __len__(self):
        return len(self.all_train_data)

    def __getitem__(self, index):
        ce_data = self.all_train_data[index]["ce"]
        t1_data = self.all_train_data[index]["t1"]
        t2_data = self.all_train_data[index]["t2"]
        dwi_data = self.all_train_data[index]["dwi"]
        t1_data = sitk.GetArrayFromImage(sitk.ReadImage(t1_data))
        t2_data = sitk.GetArrayFromImage(sitk.ReadImage(t2_data))
        dwi_data = sitk.GetArrayFromImage(sitk.ReadImage(dwi_data))

        t1_data = torch.from_numpy(t1_data.copy()).float()
        t2_data = torch.from_numpy(t2_data.copy()).float()
        dwi_data = torch.from_numpy(dwi_data.copy()).float()
        # t1_data = torch.unsqueeze(t1_data, 1)
        # t2_data = torch.unsqueeze(t2_data, 1)
        return ce_data, t1_data, t2_data, dwi_data

class ProstateMRI(Dataset):
    def __init__(self, dataset_config):
        self.ce_data, self.t1_data, self.t2_data = [], [], []
        train_dir = dataset_config.train_dir
        self.all_train_data = []
        for ids in sorted(os.listdir(train_dir)[:]):
            # ce_dir = os.path.join(train_dir, str(ids), "T1CE.nii.gz")
            # t1_dir = os.path.join(train_dir, str(ids), "T1.nii.gz")
            # t2_dir = os.path.join(train_dir, str(ids), "T2.nii.gz")
            # dwi_dir = os.path.join(train_dir, str(ids), "B1400.nii.gz")

            ce_dir = os.path.join(train_dir, str(ids), "T1CE.nii.gz")
            t1_dir = os.path.join(train_dir, str(ids), "T1.nii.gz")
            t2_dir = os.path.join(train_dir, str(ids), "T2.nii.gz")
            dwi_dir = os.path.join(train_dir, str(ids), "B1400.nii.gz")
            self.all_train_data.append({"ce": ce_dir, "t1": t1_dir, "t2": t2_dir, "dwi": dwi_dir})
        print(len(self.all_train_data), "test_data")

    def __len__(self):
        return len(self.all_train_data)

    def __getitem__(self, index):
        ce_data = self.all_train_data[index]["ce"]
        t1_data = self.all_train_data[index]["t1"]
        t2_data = self.all_train_data[index]["t2"]
        dwi_data = self.all_train_data[index]["dwi"]
        t1_data = sitk.GetArrayFromImage(sitk.ReadImage(t1_data))
        t2_data = sitk.GetArrayFromImage(sitk.ReadImage(t2_data))
        dwi_data = sitk.GetArrayFromImage(sitk.ReadImage(dwi_data))

        t1_data = torch.from_numpy(t1_data.copy()).float()
        t2_data = torch.from_numpy(t2_data.copy()).float()
        dwi_data = torch.from_numpy(dwi_data.copy()).float()
        # t1_data = torch.unsqueeze(t1_data, 1)
        # t2_data = torch.unsqueeze(t2_data, 1)
        return ce_data, t1_data, t2_data, dwi_data


class dataset_config:
    def __init__(self, K, random_state, train_dir, fold, mode="train"):
        self.K = K
        self.random_state = random_state
        self.train_dir = train_dir
        self.fold = fold
        self.train_ce_dir = os.path.join(train_dir, mode + "_ce_data.npy")
        self.train_t1_dir = os.path.join(train_dir, mode + "_t1_data.npy")
        self.train_t2_dir = os.path.join(train_dir, mode + "_t2_data.npy")
        self.train_dwi_dir = os.path.join(train_dir, mode + "_dwi_data.npy")

if __name__ == "__main__":
    data_config = {
        "K": 5,
        "random_state": 2024,
        "train_dir": "/home/user4/sharedata/newnas_1/MJY_file/Prostate_dataset/",
        "fold": 1
    }
    data_config = dataset_config(**data_config)
    # train_id, test_id = get_prostate_dataset(data_config)
    a = ProstateMRI_with_shannon_entropy(data_config)
