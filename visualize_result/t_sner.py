# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：t_sner.py
@IDE     ：PyCharm 
@Author  ：MJY
@Date    ：2025/3/11 11:28 
"""
import os.path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

np.random.seed(42)


# put in model

# file_path = "/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_49_small.h5"
# if max(timesteps) == 0:
#     print("saving feature")
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_a_style, 'h_a_style')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_al_style, 'h_al_style')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_l_style, 'h_l_style')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_a_content, 'h_a_content')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_al_content, 'h_al_content')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_l_content, 'h_l_content')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_a_anatomy, 'h_a_anatomy')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_al_anatomy, 'h_al_anatomy')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_al_lesion, 'h_al_lesion')
#     save_tensor_hdf5("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", h_l_lesion, 'h_l_lesion')
#     # save_tensor_hdf5("/data/newnas_1/MJY_file/t-SNE.h5", h_lesion, 'h_lesion')
#     a = h5py.File("/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE.h5", 'r')['h_a_style'][()]
#     if a.shape[0] > 1000:
#         print(a.shape)
#         sys.exit()

# if max(timesteps) == 0:
#     print("saving feature")
#     save_tensor_hdf5("/data/newnas_1/MJY_file/t-SNE.h5", h_style, 'h_style')
#     save_tensor_hdf5("/data/newnas_1/MJY_file/t-SNE.h5", h_share_content, 'h_share_content')
#     save_tensor_hdf5("/data/newnas_1/MJY_file/t-SNE.h5", h_anatomy, 'h_anatomy')
#     save_tensor_hdf5("/data/newnas_1/MJY_file/t-SNE.h5", h_lesion, 'h_lesion')
#     a = h5py.File("/data/newnas_1/MJY_file/t-SNE.h5", 'r')['h_style'][()]
#     if a.shape[0] > 500:
#         print(a.shape)
#         sys.exit()

# file_path = "/home/user4/sharedata/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_55.h5"
# if max(timesteps) == 0 or max(timesteps) == 1:
#     print("saving feature")
#     save_tensor_hdf5(file_path, h_a, 'h_a')
#     save_tensor_hdf5(file_path, h_al, 'h_al')
#     save_tensor_hdf5(file_path, h_l, 'h_l')
#     a = h5py.File(file_path, 'r')['h_a'][()]
#     if a.shape[0] > 1000:
#         print(a.shape)
#         sys.exit()
# 第一步：定义函数，将特征张量展平
def flatten_features(feature):
    """
    将形状为 [B, 144, 10, 10] 的张量展平为 [B, 144*10*10]
    """
    B = feature.shape[0]
    return feature.reshape(B, -1)


def main_10(p):
    with h5py.File(p, "r") as f:
        h_a_style = f['h_a_style'][:]
        h_al_style = f['h_al_style'][:]
        h_l_style = f['h_l_style'][:]
        h_a_content = f['h_a_content'][:]
        h_al_content = f['h_al_content'][:]
        h_l_content = f['h_l_content'][:]
        h_a_anatomy = f['h_a_anatomy'][:]
        h_al_anatomy = f['h_al_anatomy'][:]
        h_al_lesion = f['h_al_lesion'][:]
        h_l_lesion = f['h_l_lesion'][:]

    # 假设你的四个特征张量已经定义好，形状均为 [B, 144, 10, 10]
    # 例如：style, content, anatomy, lesion
    B = h_a_style.shape[0]  # 假设批次大小为100，你可以替换为实际的B值

    # 展平所有特征张量
    h_a_style = flatten_features(h_a_style)  # 形状：[B, 14400]
    h_al_style = flatten_features(h_al_style)  # 形状：[B, 14400]
    h_l_style = flatten_features(h_l_style)  # 形状：[B, 14400]
    h_a_content = flatten_features(h_a_content)  # 形状：[B, 14400]
    h_al_content = flatten_features(h_al_content)  # 形状：[B, 14400]
    h_l_content = flatten_features(h_l_content)  # 形状：[B, 14400]
    h_a_anatomy = flatten_features(h_a_anatomy)  # 形状：[B, 14400]
    h_al_anatomy = flatten_features(h_al_anatomy)  # 形状：[B, 14400]
    h_al_lesion = flatten_features(h_al_lesion)  # 形状：[B, 14400]
    h_l_lesion = flatten_features(h_l_lesion)  # 形状：[B, 14400]

    # 第二步：合并所有特征类型的数据
    all_features = np.concatenate(
        [
            h_a_style,
            h_al_style,
            h_l_style,
            h_a_content,
            h_al_content,
            h_l_content,
            h_a_anatomy,
            h_al_anatomy,
            h_al_lesion,
            h_l_lesion
        ], axis=0)
    # 形状：[4*B, 14400]

    # 第三步：使用PCA进行预降维（可选，但推荐）
    pca = PCA(n_components=50, random_state=42)  # 降到50维，可以根据需要调整
    all_features = pca.fit_transform(all_features)  # 形状：[4*B, 50]

    # 第四步：应用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)  # random_state固定种子以保证可重复性
    all_features_tsne = tsne.fit_transform(all_features)  # 形状：[4*B, 2]

    # 第五步：准备标签，用于区分不同的特征类型
    labels = np.array(
        ['$S_1$'] * B +
        ['$S_2$'] * B +
        ['$S_3$'] * B +
        ['$C_1$'] * B +
        ['$C_2$'] * B +
        ['$C_3$'] * B +
        ['$A_1$'] * B +
        ['$A_2$'] * B +
        ['$L_2$'] * B +
        ['$L_3$'] * B
    )

    markers = {
        'S': 'o',  # 圆圈
        'C': 's',  # 方块
        'A': '^',  # 三角形
        'L': 'x'  # 叉号
    }

    colors = {
        '$S_1$': 'blue',
        '$S_2$': 'lightblue',
        '$S_3$': 'darkblue',
        '$C_1$': 'green',
        '$C_2$': 'lightgreen',
        '$C_3$': 'darkgreen',
        '$A_1$': 'red',
        '$A_2$': 'pink',
        '$L_2$': 'purple',
        '$L_3$': 'magenta'
    }

    # 提取大类函数
    def get_major_category(label):
        return label[1]  # 例如 '$S_1$' -> 'S'

    # 第六步：可视化
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        indices = labels == label
        major_cat = get_major_category(label)
        plt.scatter(all_features_tsne[indices, 0],
                    all_features_tsne[indices, 1],
                    label=label,
                    marker=markers[major_cat],  # 使用大类的标记符号
                    c=colors[label],  # 使用具体类别的颜色
                    alpha=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16,markerscale=1.5,handletextpad=0.5)
    # plt.title('t-SNE Visualization of Features')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.savefig(
        '/data/newnas_1/MJY_file/visualize_all/t-SNE/{}.png'.format(os.path.basename(p).split(".")[0]),
        bbox_inches='tight', dpi=600)
    plt.show()


def main_4(p):
    with h5py.File(p, "r") as f:
        style = f['h_style'][:]
        content = f['h_share_content'][:]
        anatomy = f['h_anatomy'][:]
        lesion = f['h_lesion'][:]
    # with h5py.File("/data/newnas_1/MJY_file/t-SNE.h5", "r") as f:
    #     style = f['h_a'][:]
    #     content = f['h_al'][:]
    #     anatomy = f['h_l'][:]
    # 假设你的四个特征张量已经定义好，形状均为 [B, 144, 10, 10]
    # 例如：style, content, anatomy, lesion
    B = style.shape[0]  # 假设批次大小为100，你可以替换为实际的B值

    # 展平所有特征张量
    style_flat = flatten_features(style)  # 形状：[B, 14400]
    content_flat = flatten_features(content)  # 形状：[B, 14400]
    anatomy_flat = flatten_features(anatomy)  # 形状：[B, 14400]
    lesion_flat = flatten_features(lesion)  # 形状：[B, 14400]

    # 第二步：合并所有特征类型的数据
    all_features = np.concatenate(
        [
            style_flat,
            content_flat,
            anatomy_flat,
            lesion_flat
        ], axis=0)
    # 形状：[4*B, 14400]

    # 第三步：使用PCA进行预降维（可选，但推荐）
    pca = PCA(n_components=50, random_state=42)  # 降到50维，可以根据需要调整
    all_features = pca.fit_transform(all_features)  # 形状：[4*B, 50]

    # 第四步：应用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)  # random_state固定种子以保证可重复性
    all_features_tsne = tsne.fit_transform(all_features)  # 形状：[4*B, 2]

    # 第五步：准备标签，用于区分不同的特征类型
    labels = np.array(['style'] * B + ['content'] * B + ['anatomy'] * B
                      + ['lesion'] * B
                      )

    markers = {
        'style': 'o',  # 圆圈
        'content': 's',  # 方块
        'anatomy': '^',  # 三角形
        'lesion': 'x'  # 叉号
    }
    # markers = {
    #     '1': 'o',  # 圆圈
    #     '2': 's',  # 方块
    #     '3': '^',  # 三角形
    #     # 'lesion': 'x'  # 叉号
    # }

    # 第六步：可视化
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(all_features_tsne[indices, 0],
                    all_features_tsne[indices, 1],
                    label=label,
                    marker=markers[label],
                    alpha=0.5)  # alpha控制透明度，便于观察重叠区域
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16,markerscale=1.5,handletextpad=0.5)
    # plt.title('t-SNE Visualization of Features')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.savefig(
        '/data/newnas_1/MJY_file/visualize_all/t-SNE/{}.png'.format(os.path.basename(p).split(".")[0]),
        bbox_inches='tight', dpi=600)
    plt.show()


def main_3(p):
    with h5py.File(p, "r") as f:
        style = f['h_a'][:]
        content = f['h_al'][:]
        anatomy = f['h_l'][:]
    # 假设你的四个特征张量已经定义好，形状均为 [B, 144, 10, 10]
    # 例如：style, content, anatomy, lesion
    B = style.shape[0]  # 假设批次大小为100，你可以替换为实际的B值

    # 展平所有特征张量
    style_flat = flatten_features(style)  # 形状：[B, 14400]
    content_flat = flatten_features(content)  # 形状：[B, 14400]
    anatomy_flat = flatten_features(anatomy)  # 形状：[B, 14400]

    # 第二步：合并所有特征类型的数据
    all_features = np.concatenate(
        [
            style_flat,
            content_flat,
            anatomy_flat,
        ], axis=0)
    # 形状：[4*B, 14400]

    # 第三步：使用PCA进行预降维（可选，但推荐）
    pca = PCA(n_components=50, random_state=42)  # 降到50维，可以根据需要调整
    all_features = pca.fit_transform(all_features)  # 形状：[4*B, 50]

    # 第四步：应用t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)  # random_state固定种子以保证可重复性
    all_features_tsne = tsne.fit_transform(all_features)  # 形状：[4*B, 2]

    # 第五步：准备标签，用于区分不同的特征类型
    labels = np.array(['H1'] * B + ['H2'] * B + ['H3'] * B)

    markers = {
        'H1': 'o',  # 圆圈
        'H2': 's',  # 方块
        'H3': '^',  # 三角形
    }

    # 第六步：可视化
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(all_features_tsne[indices, 0],
                    all_features_tsne[indices, 1],
                    label=label,
                    marker=markers[label],
                    alpha=0.5)  # alpha控制透明度，便于观察重叠区域
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16,markerscale=1.5,handletextpad=0.5)
    # plt.title('t-SNE Visualization of Features')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    plt.savefig(
        '/data/newnas_1/MJY_file/visualize_all/t-SNE/{}.png'.format(os.path.basename(p).split(".")[0]),
        bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    main_3("/data/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_55.h5")
    # main_4("/data/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_brats_8.h5")
    # main_4("/data/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_brats_63.h5")
    main_10("/data/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_49_small.h5")
    main_10("/data/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_139_small.h5")

    main_3("/data/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_brats_6.h5")
    main_10("/data/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_brats_8_small.h5")
    main_10("/data/newnas_1/MJY_file/visualize_all/t-SNE/feature/t-SNE_brats_63_small.h5")
