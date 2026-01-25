import random
from typing import Any, Optional

import SimpleITK as sitk
import cv2
import h5py
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable
import matplotlib.pyplot as plt
from training_project.utils.util import canny_edge_detector

def plot_images(images, titles=None):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))  # 一行 num_images 列

    if titles is None:
        titles = [f"Image {i+1}" for i in range(num_images)]

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap="gray")  # 显示灰度图
        axes[i].set_title(title)
        axes[i].axis("off")  # 关闭坐标轴

    plt.tight_layout()
    plt.show()

class GetEdgeMap(MapTransform):
    def __init__(self, keys: KeysCollection, type='sobel'):
        super().__init__(keys)
        self.type = type

    def __call__(self, data):
        d = dict(data)
        # index_list = [0,1,2]  # 0 T1  1 T2  2 DWI
        # all_to_edge = d[self.keys[0]][index_list]
        index_list = [0]  # 0 T1  1 T2  2 DWI
        all_to_edge = d[self.keys[0]][index_list]
        if isinstance(all_to_edge, torch.Tensor):
            all_to_edge = all_to_edge.numpy()
        all_edge = []
        if self.type == 'sobel':
            # for key in self.keys:
            threshold_random = random.randint(10, 20)
            bilateralFilter_random_c = random.randint(40, 50)
            bilateralFilter_random_s = bilateralFilter_random_c
            for inp in all_to_edge:
                nplabs = inp
                if isinstance(nplabs, torch.Tensor):
                    nplabs = nplabs.numpy()
                # 从-1~1回退到0-255
                nplabs = (nplabs + 1) * 255 / 2
                nplabs = cv2.bilateralFilter(np.uint8(nplabs), 10, bilateralFilter_random_c, bilateralFilter_random_s)
                nplabs = np.uint8(nplabs)

                x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
                y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
                absX = cv2.convertScaleAbs(x)
                absY = cv2.convertScaleAbs(y)
                nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

                nplabs[nplabs < threshold_random] = 0

                nplabs = (nplabs - nplabs.min() + 1e-12) / (nplabs.max() - nplabs.min() + 1e-8)
                all_edge.append(nplabs)
                #保存边缘图
                # cv2.imwrite(f"/home/user15/sharedata/newnas_1/MJY_file/visualize_all/edge/{}_edge.jpg", nplabs)
            # plot_images(all_edge, ["T1", "T2", "DWI"])
            # d['edge'] = np.stack(all_edge, axis=0).astype(np.float32)
            nplabs = np.max(all_edge, axis=0)
            d['edge'] = nplabs[None, :].astype(np.float32)
        elif self.type == 'canny':
            for inp in all_to_edge:
                inp = np.uint8((inp + 1) * 255 / 2)
                nplabs = canny_edge_detector(inp)
                nplabs = (nplabs - nplabs.min() + 1e-12) / (nplabs.max() - nplabs.min() + 1e-8)
                all_edge.append(nplabs)
            nplabs = np.max(all_edge, axis=0)
            d['edge'] = nplabs[None, :].astype(np.float32)
        elif self.type == 'laplacian':
            threshold_random = random.randint(10, 20)
            bilateralFilter_random_c = random.randint(40, 50)
            bilateralFilter_random_s = bilateralFilter_random_c
            for inp in all_to_edge:
                nplabs = inp
                if isinstance(nplabs, torch.Tensor):
                    nplabs = nplabs.numpy()
                # 从-1~1回退到0-255
                nplabs = (nplabs + 1) * 255 / 2
                nplabs = cv2.bilateralFilter(nplabs, 10, bilateralFilter_random_c, bilateralFilter_random_s)
                nplabs = np.uint8(nplabs)

                laplacian = cv2.Laplacian(nplabs, cv2.CV_16S, ksize=3)
                nplabs = cv2.convertScaleAbs(laplacian)
                nplabs[nplabs < threshold_random] = 0

                nplabs = (nplabs - nplabs.min() + 1e-12) / (nplabs.max() - nplabs.min() + 1e-8)
                all_edge.append(nplabs)
            # plot_images(all_edge, ["T1", "T2", "DWI"])
            # d['edge'] = np.stack(all_edge, axis=0).astype(np.float32)
            nplabs = np.max(all_edge, axis=0)
            d['edge'] = nplabs[None, :].astype(np.float32)
        elif self.type == 'sobel&laplacian':
            threshold_random = random.randint(10, 20)
            bilateralFilter_random_c = random.randint(40, 50)
            bilateralFilter_random_s = bilateralFilter_random_c
            for inp in all_to_edge:
                nplabs = inp
                if isinstance(nplabs, torch.Tensor):
                    nplabs = nplabs.numpy()
                # 从-1~1回退到0-255
                nplabs = (nplabs + 1) * 255 / 2
                nplabs = cv2.bilateralFilter(nplabs, 10, bilateralFilter_random_c, bilateralFilter_random_s)
                nplabs = np.uint8(nplabs)

                x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
                y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
                absX = cv2.convertScaleAbs(x)
                absY = cv2.convertScaleAbs(y)
                nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

                laplacian = cv2.Laplacian(nplabs, cv2.CV_16S, ksize=3)
                laplacian_result = cv2.convertScaleAbs(laplacian)
                laplacian_result [nplabs < threshold_random] = 0

                nplabs = cv2.addWeighted(nplabs, 0.7, laplacian_result, 0.3, 0)
                # nplabs = np.maximum(nplabs, laplacian_result)
                nplabs[nplabs < threshold_random] = 0

                nplabs = (nplabs - nplabs.min() + 1e-12) / (nplabs.max() - nplabs.min() + 1e-8)
                all_edge.append(nplabs)
            # plot_images(all_edge, ["T1", "T2", "DWI"])
            # d['edge'] = np.stack(all_edge, axis=0).astype(np.float32)
            nplabs = np.max(all_edge, axis=0)
            d['edge'] = nplabs[None, :].astype(np.float32)
        else:
            raise ValueError(f"Invalid type: {self.type}")
        return d


class LoadH5(MapTransform):
    def __init__(self, path_key, keys: KeysCollection):
        super().__init__(keys)
        self.path_key = path_key

    def __call__(self, data):
        d = dict(data)
        h5_file = h5py.File(d[self.path_key])
        for key in self.keys:
            d[key] = h5_file[key][()]
        # d.pop(self.path_key)
        return d


class LoadImageITKd(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = sitk.GetArrayFromImage(sitk.ReadImage(data[key], outputPixelType=sitk.sitkFloat32))
        # d.pop(self.path_key)
        return d


class RandCropOrPad(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection):
        super(Randomizable, self).__init__(keys)

    def set_random_state(
            self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandCropOrPad":
        pass

    def randomize(self, data: Any) -> None:
        if self.R.random() < self.prob:
            pass

    def __call__(self, data):
        self.randomize(data)


class CropOrPad(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        h5_file = h5py.File(d[self.path_key])
        for key in self.keys:
            d[key] = h5_file[key]
        d.pop(self.path_key)
        return d
