# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：get_edge_visual.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/3/3 20:35 
"""
import numpy as np
from PIL import Image

from training_project.utils.my_transform import GetEdgeMap

if __name__ == "__main__":
    path = "/home/user15/sharedata/newnas_1/MJY_file/visualize_all/edge/"
    # 从这个路径读取png为ndarray
    a = Image.open(path + "1.png").convert('L').resize((320, 320))
    a = np.array(a, dtype=np.float32)
    b = Image.open(path + "2.png").convert('L').resize((320, 320))
    b = np.array(b, dtype=np.float32)
    c = Image.open(path + "3.png").convert('L').resize((320, 320))
    c = np.array(c, dtype=np.float32)
    a = a / 255 * 2 - 1
    b = b / 255 * 2 - 1
    c = c / 255 * 2 - 1
    d = np.stack([a, b, c], axis=0)
    d = {"1": d}
    edge = GetEdgeMap("1")
    edge(d)
    print("Done")
