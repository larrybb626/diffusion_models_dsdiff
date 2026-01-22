# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：my_trash_file.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/9/9 15:01 
"""

#
import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32).reshape(1, 1, 2, 5)
print(a)