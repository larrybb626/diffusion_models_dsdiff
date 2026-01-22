import argparse

parser = argparse.ArgumentParser()
# =============================常改的参数=============================
parser.add_argument('--Task_id', type=str, default='6')
parser.add_argument('--cuda_idx', type=int, default=3)  # 用几号卡的显存
parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')  # 交叉验证的折数
parser.add_argument('--fold_idx', type=int, default=1)  # 跑第几折的数据 1开始

parser.add_argument('--train_keys', type=list, default=["t1"])  # 使用那些序列进行训练,ce必有的，随便填
parser.add_argument('--train_batch_size', type=int, default=8)  # 训练的batch_size
parser.add_argument('--val_batch_size', type=int, default=8)  # 测试batch_size
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--checkpoint_epoch', type=int, default=5)  # 多久保存一个断点
# =============================偶尔改的参数=============================
# dataset_type
parser.add_argument('--dataset_type', type=str, default="normal")  # "normal"
# result&save
parser.add_argument('--dir_prefix', type=str, default=r'/home/user15/sharedata/')
parser.add_argument('--result_path', type=str, default=r'newnas/MJY_file/CE-MRI/train_result/')  # 结果保存地址
parser.add_argument('--filepath_img', type=str, default=r'newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm')
parser.add_argument('--h5_3d_img_dir', type=str, default=r'newnas/MJY_file/CE-MRI/PCa_new/h5_data_3d_patch_pre')
parser.add_argument('--h5_2d_img_dir', type=str, default=r'newnas/MJY_file/CE-MRI/PCa_new/h5_data_2d_pre_320320_01norm')
parser.add_argument('--filepath_mask', type=str, default=r'newnas/MJY_file/CE-MRI/PCa_new/CE-MRI-PCa-new-pre-320320-01norm')
# model hyper-parameters
parser.add_argument('--image_size', type=int, default=320)
# 设置学习率(chrome有问题,额外记录在一个txt里面)
# 注意,学习率记录部分代码也要更改
parser.add_argument('--lr', type=float, default=1e-3)  # 初始or最大学习率
parser.add_argument('--lr_low', type=float, default=1e-8)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)
parser.add_argument('--num_epochs', type=int, default=300)  # 总epoch
parser.add_argument('--num_steps', type=int, default=120000)  # 10w
parser.add_argument('--lr_cos_epoch', type=int, default=300)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用
parser.add_argument('--lr_warm_epoch', type=int, default=0)  # warm_up的epoch数,一般就是10~20,为0或False则不使用

# parser.add_argument('--save_model_step', type=int, default=200)  # 多少epoch保存一次模型
parser.add_argument('--val_step', type=int, default=1)  # 多少epoch测试一次

# =============================一般不改的参数=============================
parser.add_argument('--mode', type=str, default='train', help='train/test')  # 训练还是测试
parser.add_argument('--num_epochs_decay', type=int, default=10)  # decay开始的最小epoch数
parser.add_argument('--decay_ratio', type=float, default=0.1)  # 0~1,每次decay到1*ratio
parser.add_argument('--decay_step', type=int, default=80)  # epoch

# optimizer reg_param
parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
parser.add_argument('--augmentation_prob', type=float, default=0.4)  # 数据扩增的概率

# training hyper-parameters
parser.add_argument('--img_ch', type=int, default=4)
parser.add_argument('--output_ch', type=int, default=1)
parser.add_argument('--DataParallel', type=bool, default=False)  # 数据并行,开了可以用多张卡的显存,不推荐使用
parser.add_argument('--train_flag', type=bool, default=False)  # 训练过程中是否测试训练集,不测试会节省很多时间
parser.add_argument('--seed', type=int, default=2024)  # 随机数的种子点，一般不变
parser.add_argument('--TTA', type=bool, default=False)

config = parser.parse_args()
