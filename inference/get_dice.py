# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：get_dice.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/2/17 15:17 
"""
import os
import SimpleITK as sitk
import numpy as np
# from torchmetrics.segmentation import DiceScore
import pandas as pd
def dice_coefficient(pred, gt):
    intersection = np.sum((pred > 0) & (gt > 0))
    union = np.sum(pred > 0) + np.sum(gt > 0)
    if union == 0:  # 避免除零错误
        return 1.0 if np.all(pred == gt) else 0.0
    return 2.0 * intersection / union
if __name__ == "__main__":
    # 真实标签 (Ground Truth) 目录
    gt_dir = "/data/newnas_1/MJY_file/BraTS_dataset/data_pre_original/images_ts"
    models = ['cGAN', 'DisC-Diff', 'DS-Diff', 'Real', 'ResViT', 'SD3']
    # 预测标签 (Predicted Labels) 目录
    for model in models:
        pred_dir = "/data/newnas_1/MJY_file/BraTS_dataset_Seg_result_508"
        pred_dir = os.path.join(pred_dir, model)
        # 存储每个 case 的 Dice
        dice_scores = {}
        # dice_score = DiceScore(num_classes=5, average="micro")
        # 遍历真实标签目录
        for case in os.listdir(gt_dir):
            gt_path = os.path.join(gt_dir, case, "seg.nii.gz")
            pred_path = os.path.join(pred_dir, f"{case}.nii.gz")  # 预测标签命名应与真实标签一致

            if not os.path.exists(gt_path):
                print(f"Warning: Ground Truth {gt_path} not found, skipping.")
                continue
            if not os.path.exists(pred_path):
                print(f"Warning: Predicted Label {pred_path} not found, skipping.")
                continue

            # 读取标签
            gt_img = sitk.ReadImage(gt_path)
            pred_img = sitk.ReadImage(pred_path)

            # 转换为 numpy 数组
            gt_array = sitk.GetArrayFromImage(gt_img)
            gt_array[gt_array == 3] = 1
            gt_array[gt_array == 2] = 0

            pred_array = sitk.GetArrayFromImage(pred_img)

            # 确保大小匹配
            if gt_array.shape != pred_array.shape:
                print(f"Shape Mismatch: {case} (GT: {gt_array.shape}, Pred: {pred_array.shape})")
                continue

            # 计算 Dice
            dice = dice_coefficient(pred_array, gt_array)
            dice_scores[case] = dice
            # 把键和值保存为excel 在pred_dir中保存
            # print(f"Dice for {case}: {dice:.4f}")
        mean_dice = np.mean(list(dice_scores.values()))
        dice_scores["Mean"] = mean_dice
        print(f"Mean Dice for {model}: {mean_dice:.4f}")
        if dice_scores:
            dice_df = pd.DataFrame(list(dice_scores.items()), columns=["Case", "Dice Score"])
            # 保存 Excel 到预测标签目录
            excel_path = os.path.join(pred_dir, f"{model}_dice_scores.xlsx")
            dice_df.to_excel(excel_path, index=False)


