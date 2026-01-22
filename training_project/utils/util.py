import cv2
import numpy as np
import torch
from monai.transforms import SobelGradients, AsDiscrete, Compose
import torch.nn as nn
from PIL import Image
import h5py

def normalize_percentile_to_255(data, lower_percentile=0, upper_percentile=100):
    """
    Normalize data based on the specified lower and upper percentiles and scale to [0, 255].

    Parameters:
    data (torch.Tensor): The image data to normalize (either 2D or 3D).
    lower_percentile (int): The lower percentile for clipping.
    upper_percentile (int): The upper percentile for clipping.

    Returns:
    torch.Tensor: Normalized image data scaled to [0, 255].
    """
    # Convert MRI data to a NumPy array if it's a torch Tensor
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Calculate the percentile values
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)

    # Clip the data
    data_clipped = np.clip(data, lower_bound, upper_bound)

    # Normalize the data to [0, 1] then scale to [0, 255]
    if upper_bound - lower_bound > 0:
        data_normalized = (data_clipped - lower_bound) / (upper_bound - lower_bound)
    else:
        data_normalized = data_clipped
    data_scaled = data_normalized * 255

    # Convert to integer type suitable for image data
    data_scaled = np.round(data_scaled).astype(np.uint8)

    return data_scaled
def get_duration_time_str(s_time, e_time):
    h, remainder = divmod((e_time - s_time), 3600)  # 小时和余数
    m, s = divmod(remainder, 60)  # 分钟和秒
    time_str = "%02d h:%02d m:%02d s" % (h, m, s)
    return time_str


def canny_edge_detector(image, low_threshold=50, high_threshold=100, kernel_size=7):
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Apply the Canny edge detector
    magnitude = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # magnitude = normalize_percentile_to_255(magnitude)

    return magnitude

def get_edge(tensor):
    # sobel算子和0.4卡阈值
    # tensor.squeeze(0)
    transforms_x = Compose([
        SobelGradients(kernel_size=3, spatial_axes=[-1],
                       padding_mode="zeros"
                       ),
        # AsDiscrete(threshold=0.4)
    ]
    )
    transforms_y = Compose([
        SobelGradients(kernel_size=3, spatial_axes=[-2],
                       padding_mode="zeros"
                       ),
        # AsDiscrete(threshold=0.4)
    ]
    )
    edge_map_x = transforms_x(tensor)
    edge_map_y = transforms_y(tensor)
    # edge_map = torch.pow((torch.pow(edge_map_x, 2) + torch.pow(edge_map_y, 2)), 0.5)
    edge_map = edge_map_y+edge_map_x
    # edge_map.unsqueeze(1)
    return edge_map


def init_weights(net, init_type='kaiming', gain=0.02):
    """
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    """

    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    # propagate to children
    for m in net.children():
        if hasattr(m, 'init_weights'):
            m.init_weights(init_type, gain)
    return net


def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(opt.items()):
        # comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    # print(message)
    return message

def get_heatmap(cosine_matrix):
    #将cosine_matrix转换为[0,1]中
    cosine_sim_matrix_normalized = (cosine_matrix - cosine_matrix.min()) / (cosine_matrix.max() - cosine_matrix.min())
    cosine_sim_np = cosine_sim_matrix_normalized.cpu().numpy()
    from matplotlib import cm

    colormap = cm.get_cmap('viridis')
    heatmap_colored = colormap(cosine_sim_np)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_colored = Image.fromarray(heatmap_colored)
    image_tensor = torch.tensor(np.array(heatmap_colored))
    return image_tensor

def save_tensor_hdf5(file_path, new_data, label):
    """
    追加写入新的张量数据到 HDF5 文件中
    :param file_path: HDF5 文件路径
    :param new_data: 要存储的张量 (torch.Tensor)
    :param label: 类别标签 (str)
    """
    with h5py.File(file_path, "a") as f:  # "a" 模式表示追加
        if label in f:
            # 如果数据集已存在，扩展它
            dset = f[label]
            dset.resize((dset.shape[0] + new_data.shape[0]), axis=0)  # 扩展第一维
            dset[-new_data.shape[0]:] = new_data.cpu().numpy()  # 追加数据
        else:
            # 如果数据集不存在，创建新的
            maxshape = (None, *new_data.shape[1:])  # 第一维不固定，其他维度固定
            f.create_dataset(label, data=new_data.cpu().numpy(), maxshape=maxshape)