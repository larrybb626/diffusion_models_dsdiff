import os
import shutil
from configs.train_config import config
import numpy as np
import random
import pandas as pd

seed = config.seed
np.random.seed(seed)
random.seed(seed)


def clean_filenames_in_folder(folder_path):
    """
    执行数据清理：去除文件名前的人名，只保留 F_Data... 或 S_Data...
    例如: BAO CHENG WANG_F_Data1.nii.gz -> F_Data1.nii.gz
    """
    if not os.path.exists(folder_path):
        return

    files = os.listdir(folder_path)
    for file in files:
        old_path = os.path.join(folder_path, file)
        if not os.path.isfile(old_path):
            continue

        new_name = None
        # 匹配 _F_ 或 _S_ 进行截取
        if "_F_" in file:
            parts = file.split('_F_')
            if len(parts) > 1:
                new_name = "F_" + parts[-1]  # 拼接回 F_Data...
        elif "_S_" in file:
            parts = file.split('_S_')
            if len(parts) > 1:
                new_name = "S_" + parts[-1]  # 拼接回 S_Data...

        # 执行重命名
        if new_name and new_name != file:
            new_path = os.path.join(folder_path, new_name)
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {file} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {file}: {e}")


if __name__ == "__main__":
    nii_folder = r'/nas_3/LaiRuiBin/Changhai/data/resampled/SSA'
    ts_folder = os.path.join(nii_folder, "images_ts")
    tr_folder = os.path.join(nii_folder, "images_tr")
    train_test_excel = r'/nas_3/LaiRuiBin/Changhai/data/train_test.xlsx'

    if not os.path.exists(ts_folder):
        os.makedirs(ts_folder)
    if not os.path.exists(tr_folder):
        os.makedirs(tr_folder)

    # 1. 获取所有ID列表并清理系统文件夹
    all_id_list = sorted(os.listdir(nii_folder))
    for ignore_item in ["images_tr", "images_ts", "train_test.xlsx"]:
        if ignore_item in all_id_list:
            all_id_list.remove(ignore_item)

    print(f"Total folders detected: {len(all_id_list)}")

    # ---------------------------------------------------------
    # 步骤 A: 先对所有文件夹进行数据文件名清洗 (新增的操作)
    # ---------------------------------------------------------
    print("Starting filename cleaning...")
    for id_num in all_id_list:
        current_person_folder = os.path.join(nii_folder, id_num)
        clean_filenames_in_folder(current_person_folder)
    print("Filename cleaning finished.")

    # ---------------------------------------------------------
    # 步骤 B: 划分训练集和测试集 (删除了is_patient，改为随机划分)
    # ---------------------------------------------------------
    train_num = int(0.7 * len(all_id_list))

    if os.path.isfile(train_test_excel):
        print("Loading existing split from Excel...")
        train_list = list(pd.read_excel(train_test_excel, sheet_name="train")["id"].dropna().astype(str))
        test_list = list(pd.read_excel(train_test_excel, sheet_name="test")["id"].dropna().astype(str))
    else:
        print("Creating new random split (70% Train, 30% Test)...")
        # 删除了原有的 is_patient 和 StratifiedShuffleSplit 逻辑
        # 使用随机打乱进行划分
        shuffled_list = all_id_list.copy()
        random.shuffle(shuffled_list)

        train_list = sorted(shuffled_list[:train_num])
        test_list = sorted(shuffled_list[train_num:])

        # 保存划分结果到Excel
        excel_writer = pd.ExcelWriter(train_test_excel)
        train_dataframe = pd.DataFrame(train_list, columns=["id"])
        test_dataframe = pd.DataFrame(test_list, columns=["id"])
        train_dataframe.to_excel(excel_writer, sheet_name="train", index=False)
        test_dataframe.to_excel(excel_writer, sheet_name="test", index=False)
        excel_writer.close()

    print("train set size:", len(train_list))
    print("test set size:", len(test_list))

    # ---------------------------------------------------------
    # 步骤 C: 移动文件到对应的 images_tr 或 images_ts 文件夹
    # ---------------------------------------------------------
    # 注意：old_folder 的路径逻辑我根据你的原始代码保持不变
    # 但请确保 id_num 文件夹此时还在 nii_folder 根目录下

    print("Moving test files...")
    for id_num in test_list:
        # 原始路径: /SSA/id_num
        old_folder = os.path.join(nii_folder, id_num)
        # 目标路径: /SSA/images_ts/id_num
        new_folder = os.path.join(ts_folder, id_num)

        if os.path.exists(old_folder):
            shutil.move(old_folder, new_folder)
        else:
            print(f"Warning: {old_folder} not found (maybe already moved?)")

    print("Moving train files...")
    for id_num in train_list:
        # 原始路径: /SSA/id_num
        old_folder = os.path.join(nii_folder, id_num)
        # 目标路径: /SSA/images_tr/id_num
        new_folder = os.path.join(tr_folder, id_num)

        if os.path.exists(old_folder):
            shutil.move(old_folder, new_folder)
        else:
            print(f"Warning: {old_folder} not found (maybe already moved?)")

    print("Done.")