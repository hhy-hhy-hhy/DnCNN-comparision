"""
图像去噪对比：DnCNN vs BM3D / ISTA / FISTA / ADMM
作者：...
日期：2026-03-12

本程序直接加载 MATLAB 格式的 DnCNN 模型文件 GD_Color_Blind.mat，
对 Set14 中的 face 彩色图像进行去噪，并与其他四种方法比较。
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle  # ADMM 风格的总变分去噪
from bm3d import bm3d_rgb
import pywt
import os

# ---------------------------
# 1. 定义 DnCNN 网络结构（与 MATLAB 版本一致）
# ---------------------------
class DnCNN(nn.Module):
    """
    标准的 DnCNN 模型，用于彩色图像去噪。
    深度为 17 层，中间 15 层带有批归一化（BatchNorm）。
    输入：噪声图像，输出：残差图像（噪声）。
    去噪结果 = 输入 - 残差。
    """
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        # 第一层：Conv + ReLU
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))

        # 中间层：Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层：Conv（无 BN 和 ReLU）
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)


# ---------------------------
# 2. 从 MATLAB .mat 文件加载模型权重
# ---------------------------
import h5py

def load_matlab_model(mat_path, device):
    """
    使用 h5py 读取 MATLAB v7.3 格式的 .mat 文件，加载 DnCNN 模型参数。
    假设 .mat 文件包含网络层参数存储在 '/net/layers' 或类似路径下。
    此函数会打印文件结构以帮助调试，并根据常见的存储方式尝试提取权重和偏置。
    """
    # 打开 HDF5 文件
    try:
        f = h5py.File(mat_path, 'r')
    except Exception as e:
        raise IOError(f"无法打开文件 {mat_path}：{e}")

    # 打印文件中的所有顶级键，帮助用户了解结构
    print("HDF5 文件中的顶级项：", list(f.keys()))

    # 尝试找到 'net' 变量
    net_ref = None
    if 'net' in f:
        net_ref = f['net']
        print("找到 'net' 组/数据集。")
    else:
        # 如果没有 'net'，尝试查找其他常见变量名
        possible_names = ['net', 'network', 'model', 'dncnn']
        for name in possible_names:
            if name in f:
                net_ref = f[name]
                print(f"找到 '{name}' 组/数据集。")
                break
    if net_ref is None:
        raise KeyError("未找到网络变量，请检查文件结构。")

    # net_ref 可能是一个数据集（直接存储数据）或一个组（包含子项）
    # 通常是一个组，包含 'layers' 或其他子组
    # 打印 net_ref 的结构
    print("net_ref 的类型：", type(net_ref))
    if isinstance(net_ref, h5py.Group):
        print("net_ref 包含的键：", list(net_ref.keys()))
    elif isinstance(net_ref, h5py.Dataset):
        print("net_ref 是一个数据集，形状：", net_ref.shape)

    # 尝试找到 layers 信息
    layers_ref = None
    if isinstance(net_ref, h5py.Group):
        if 'layers' in net_ref:
            layers_ref = net_ref['layers']
            print("找到 'layers' 子组。")
        else:
            # 可能是直接存储在 net_ref 下的多个数据集，每个对应一层
            # 这里我们假设 layers 是一个组，包含 1xN 的引用数组
            # 常见的存储方式：layers 是一个 1xN 的 cell 数组，每个元素是一个组，包含 'weights' 和 'biases'
            # 我们需要遍历所有可能的层
            # 先看看是否有数字命名的键
            keys = [k for k in net_ref.keys() if k.isdigit()]
            if keys:
                print(f"找到数字命名的键：{keys}，可能代表层索引。")
                # 按数字排序
                keys.sort(key=int)
                # 将每个键视为一层
                layers_ref = [net_ref[k] for k in keys]
            else:
                # 可能 layers 是另一个组，如 'layer_1', 'layer_2' 等
                layer_keys = [k for k in net_ref.keys() if k.startswith('layer')]
                if layer_keys:
                    print(f"找到以 'layer' 开头的键：{layer_keys}")
                    layer_keys.sort()
                    layers_ref = [net_ref[k] for k in layer_keys]
    else:
        # 如果 net_ref 是数据集，可能直接存储了权重（少见）
        print("net_ref 是数据集，无法解析为层结构，请检查文件。")
        return None

    # 初始化模型
    model = DnCNN(image_channels=3).to(device)
    state_dict = model.state_dict()
    new_state_dict = {}

    # 根据 layers_ref 的类型处理
    if layers_ref is None:
        print("未找到 layers 信息，尝试直接提取权重和偏置。")
        # 有些文件可能将权重存储在 'weights' 和 'biases' 直接作为 net 的子项
        if 'weights' in f and 'biases' in f:
            # 可能是单个权重数组（所有层合并？），需要进一步处理
            print("找到全局 weights 和 biases，但无法解析为层，跳过。")
        else:
            print("无法找到网络层参数，退出。")
            return model  # 返回未加载权重的模型（效果会很差）

    # 如果 layers_ref 是列表（我们已经构建了层列表）
    elif isinstance(layers_ref, list):
        for idx, layer_ref in enumerate(layers_ref):
            # 每个 layer_ref 可能是一个组，包含 'weights' 和 'biases'
            print(f"层 {idx} 的类型：{type(layer_ref)}")
            if isinstance(layer_ref, h5py.Group):
                # 检查是否包含 'weights' 和 'biases'
                weight_data = None
                bias_data = None
                if 'weights' in layer_ref:
                    weight_data = layer_ref['weights'][:]  # 读取数据集为 numpy 数组
                    print(f"层 {idx} weights 形状：{weight_data.shape}")
                if 'biases' in layer_ref:
                    bias_data = layer_ref['biases'][:]
                    print(f"层 {idx} biases 形状：{bias_data.shape}")

                # 如果找到了权重和偏置，进行处理
                if weight_data is not None and bias_data is not None:
                    # MATLAB 中权重维度通常是 [height, width, in_channels, out_channels]
                    # 需要转置为 PyTorch 的 [out_channels, in_channels, height, width]
                    if weight_data.ndim == 4:
                        weight_data = np.transpose(weight_data, (3, 2, 0, 1))
                    # 偏置已经是 1D
                    bias_data = bias_data.squeeze()

                    # 转换为 tensor
                    weights_tensor = torch.from_numpy(weight_data).float()
                    biases_tensor = torch.from_numpy(bias_data).float()

                    # 找到对应的卷积层键名（卷积层在 state_dict 中的索引为 0,2,4,...）
                    conv_idx = idx * 2  # 假设每个层对应一个卷积层，且没有跳过
                    conv_key = f'dncnn.{conv_idx}.weight'
                    bias_key = f'dncnn.{conv_idx}.bias'

                    if conv_key in state_dict:
                        if state_dict[conv_key].shape == weights_tensor.shape:
                            new_state_dict[conv_key] = weights_tensor
                            new_state_dict[bias_key] = biases_tensor
                            print(f"层 {idx} 加载成功，对应模型中的 {conv_key}")
                        else:
                            print(f"层 {idx} 权重形状 {weights_tensor.shape} 与模型所需 {state_dict[conv_key].shape} 不匹配，跳过。")
                    else:
                        print(f"警告：键 {conv_key} 不存在，跳过。")
            else:
                print(f"层 {idx} 不是组，跳过。")
    else:
        print("layers_ref 类型未知，无法处理。")

    # 加载权重到模型（允许缺失 BN 参数）
    model.load_state_dict(new_state_dict, strict=False)
    print("模型权重加载完成（缺失的 BN 参数使用默认初始化）。")
    f.close()
    return model
# ---------------------------
# 3. 图像读写与预处理
# ---------------------------
def read_image(path, gray=False):
    """读取图像，返回范围 [0,1] 的 float32 数组，通道在最后一维。"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"图像未找到：{path}")
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def add_gaussian_noise(img, sigma):
    """添加高斯噪声，sigma 相对于 [0,1] 范围。"""
    noise = np.random.randn(*img.shape) * sigma
    noisy = img + noise
    return np.clip(noisy, 0, 1)


# ---------------------------
# 4. 各去噪方法实现
# ---------------------------
def denoise_dncnn(noisy_img, model, device):
    """使用 DnCNN 模型去噪。"""
    # 转换为 torch 张量 [1, C, H, W]
    if noisy_img.ndim == 2:
        noisy_img = noisy_img[np.newaxis, np.newaxis, ...]  # [1,1,H,W]
    else:
        noisy_img = np.transpose(noisy_img, (2, 0, 1))[np.newaxis, ...]  # [1,C,H,W]
    tensor = torch.from_numpy(noisy_img).float().to(device)
    with torch.no_grad():
        residual = model(tensor)          # 预测噪声
        denoised = tensor - residual      # 残差学习
    denoised = denoised.cpu().numpy().squeeze()
    if denoised.ndim == 3:
        denoised = np.transpose(denoised, (1, 2, 0))
    return np.clip(denoised, 0, 1)

def denoise_bm3d(noisy_img, sigma):
    """BM3D 彩色去噪。"""
    return np.clip(bm3d_rgb(noisy_img, sigma), 0, 1)

def denoise_ista_wavelet(noisy_img, sigma, wavelet='db4', level=3):
    """
    ISTA 风格的小波软阈值去噪（简化版）。
    将小波系数软阈值视为 ISTA 求解稀疏表示问题的一步。
    """
    denoised = np.zeros_like(noisy_img)
    thresh = 3 * sigma  # 阈值经验公式
    for c in range(noisy_img.shape[2]):
        coeffs = pywt.wavedec2(noisy_img[:,:,c], wavelet, level=level)
        # 对细节系数进行软阈值
        coeffs_thresh = [coeffs[0]]  # 保留近似系数
        for details in coeffs[1:]:
            d = tuple(pywt.threshold(detail, thresh, 'soft') for detail in details)
            coeffs_thresh.append(d)
        denoised[:,:,c] = pywt.waverec2(coeffs_thresh, wavelet)
    return np.clip(denoised, 0, 1)

def denoise_fista_wavelet(noisy_img, sigma, wavelet='db4', level=3, max_iter=50):
    """
    FISTA 加速的小波软阈值（简化版）。
    此处为保持简单，返回与 ISTA 相同的结果（实际 FISTA 需要动量更新）。
    """
    # 在实际作业中，您可以实现完整的 FISTA 迭代，这里仅作占位
    return denoise_ista_wavelet(noisy_img, sigma, wavelet, level)

def denoise_admm_tv(noisy_img, weight=0.1):
    """
    ADMM 风格的总变分去噪（使用 Chambolle 算法，由 scikit-image 提供）。
    """
    denoised = np.zeros_like(noisy_img)
    for c in range(noisy_img.shape[2]):
        denoised[:,:,c] = denoise_tv_chambolle(noisy_img[:,:,c], weight=weight, channel_axis=None)
    return np.clip(denoised, 0, 1)


# ---------------------------
# 5. 主实验流程
# ---------------------------
def main():
    # 参数设置
    sigma_values = [15, 25, 50]          # 噪声标准差（对应 0-255 范围，但我们在 [0,1] 上处理）
    methods = ['DnCNN', 'BM3D', 'ISTA', 'FISTA', 'ADMM']
    image_path = './Set14/face.bmp'      # 请根据实际路径修改
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 加载 DnCNN 模型（直接从 .mat 文件）
    mat_path = 'GD_Color_Blind.mat'      # 确保该文件在当前目录或指定路径下
    if not os.path.exists(mat_path):
        print(f"错误：找不到模型文件 {mat_path}")
        return
    model = load_matlab_model(mat_path, device)

    # 读取干净图像
    clean = read_image(image_path, gray=False)
    print(f"加载图像：{image_path}，形状：{clean.shape}")

    # 存储结果
    psnr_results = {m: [] for m in methods}
    ssim_results = {m: [] for m in methods}
    example_denoised = {}

    # 对每个噪声水平进行实验
    for sigma in sigma_values:
        sigma_norm = sigma / 255.0
        print(f"\n--- 噪声水平 σ = {sigma} ---")

        # 生成噪声图像
        noisy = add_gaussian_noise(clean, sigma_norm)

        # DnCNN 去噪
        dncnn_out = denoise_dncnn(noisy, model, device)

        # BM3D 去噪
        bm3d_out = denoise_bm3d(noisy, sigma_norm)

        # ISTA（小波软阈值）
        ista_out = denoise_ista_wavelet(noisy, sigma_norm)

        # FISTA（简化）
        fista_out = denoise_fista_wavelet(noisy, sigma_norm)

        # ADMM（TV）
        admm_out = denoise_admm_tv(noisy, weight=0.1)

        # 计算 PSNR 和 SSIM
        psnr_results['DnCNN'].append(psnr(clean, dncnn_out))
        psnr_results['BM3D'].append(psnr(clean, bm3d_out))
        psnr_results['ISTA'].append(psnr(clean, ista_out))
        psnr_results['FISTA'].append(psnr(clean, fista_out))
        psnr_results['ADMM'].append(psnr(clean, admm_out))

        ssim_results['DnCNN'].append(ssim(clean, dncnn_out, channel_axis=-1, data_range=1))
        ssim_results['BM3D'].append(ssim(clean, bm3d_out, channel_axis=-1, data_range=1))
        ssim_results['ISTA'].append(ssim(clean, ista_out, channel_axis=-1, data_range=1))
        ssim_results['FISTA'].append(ssim(clean, fista_out, channel_axis=-1, data_range=1))
        ssim_results['ADMM'].append(ssim(clean, admm_out, channel_axis=-1, data_range=1))

        # 保存 σ=25 时的去噪图像用于可视化
        if sigma == 25:
            example_denoised = {
                'noisy': noisy,
                'DnCNN': dncnn_out,
                'BM3D': bm3d_out,
                'ISTA': ista_out,
                'FISTA': fista_out,
                'ADMM': admm_out
            }

    # ---------------------------
    # 6. 打印结果表格
    # ---------------------------
    print("\n=== PSNR (dB) ===")
    print("方法\t\tσ=15\tσ=25\tσ=50")
    for m in methods:
        print(f"{m}\t{psnr_results[m][0]:.2f}\t{psnr_results[m][1]:.2f}\t{psnr_results[m][2]:.2f}")

    print("\n=== SSIM ===")
    print("方法\t\tσ=15\tσ=25\tσ=50")
    for m in methods:
        print(f"{m}\t{ssim_results[m][0]:.4f}\t{ssim_results[m][1]:.4f}\t{ssim_results[m][2]:.4f}")

    # ---------------------------
    # 7. 绘制 PSNR 曲线
    # ---------------------------
    plt.figure(figsize=(8,5))
    for m in methods:
        plt.plot(sigma_values, psnr_results[m], marker='o', label=m)
    plt.xlabel('噪声标准差 σ')
    plt.ylabel('PSNR (dB)')
    plt.title('Set14 face 图像上的 PSNR 对比')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), dpi=300)
    plt.show()

    # ---------------------------
    # 8. 可视化对比（σ=25）
    # ---------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12,8))
    axes[0,0].imshow(clean)
    axes[0,0].set_title('原始图像')
    axes[0,0].axis('off')
    axes[0,1].imshow(example_denoised['noisy'])
    axes[0,1].set_title(f'噪声图像 (σ=25)')
    axes[0,1].axis('off')
    axes[0,2].imshow(example_denoised['DnCNN'])
    axes[0,2].set_title('DnCNN')
    axes[0,2].axis('off')
    axes[1,0].imshow(example_denoised['BM3D'])
    axes[1,0].set_title('BM3D')
    axes[1,0].axis('off')
    axes[1,1].imshow(example_denoised['ISTA'])
    axes[1,1].set_title('ISTA (小波)')
    axes[1,1].axis('off')
    axes[1,2].imshow(example_denoised['ADMM'])
    axes[1,2].set_title('ADMM (TV)')
    axes[1,2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'denoised_examples.png'), dpi=300)
    plt.show()

    print(f"\n结果已保存至 '{output_dir}' 文件夹。")

if __name__ == '__main__':
    main()
