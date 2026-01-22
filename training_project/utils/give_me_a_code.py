# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：give_me_a_code.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2024/3/22 19:09 
"""


#write me a code that can copy the file into another directory
import shutil
import os
def copy_file(src:str, dst:str):
    shutil.copy(src,dst)
    print(f"copy {src} to {dst} successfully")
if __name__ == '__main__':
    src = "/home/user4/sharedata/newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-new-PCa-resample/"
    dst = "/home/user4/sharedata/newnas/MJY_file/nnUNetFile/nnUNet_raw/Dataset503_bodymask"
    for file in os.listdir(src)[61:]:
        if os.path.isdir(os.path.join(src,file)):
            # s = os.path.join(src,file,"T1.nii.gz")
            # d = os.path.join(dst,"imagesTs",file+"_0000.nii.gz")
            # os.makedirs(os.path.dirname(d),exist_ok=True)
            # copy_file(s,d)
            s = os.path.join(src, file, "body_mask.nii.gz")
            d = os.path.join(dst,"labelsTs", file + ".nii.gz")
            os.makedirs(os.path.dirname(d), exist_ok=True)
            copy_file(s, d)


