"""
CASA-Net 配置和工具函数
"""

import torch
import yaml
from pathlib import Path


VISDRONE_CLASSES = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor',
}


CASA_NET_CONFIGS = {
    'n': {
        'depth_multiple': 0.50,
        'width_multiple': 0.25,
        'max_channels': 1024,
        'param_count': '~8.5M',
        'gflops': '~24.3G',
    },
    's': {
        'depth_multiple': 0.50,
        'width_multiple': 0.50,
        'max_channels': 1024,
        'param_count': '~25.6M',
        'gflops': '~78.5G',
    },
    'm': {
        'depth_multiple': 0.50,
        'width_multiple': 1.00,
        'max_channels': 512,
        'param_count': '~21.9M',
        'gflops': '~75.4G',
    },
    'l': {
        'depth_multiple': 1.00,
        'width_multiple': 1.00,
        'max_channels': 512,
        'param_count': '~26.3M',
        'gflops': '~93.8G',
    },
    'x': {
        'depth_multiple': 1.00,
        'width_multiple': 1.50,
        'max_channels': 512,
        'param_count': '~59.0M',
        'gflops': '~209.5G',
    },
}


E_STAL_CONFIGS = {
    'min_assign': 6,
    'spatial_tolerance': 0.7,
    'threshold_scale_factor': 0.20,
    'base_threshold': 0.3,
    'small_object_threshold': 8,
}


MFEM_CONFIGS = {
    'scales': [1, 2, 4],
    'groups': 32,
    'expand_ratio': 0.5,
}


AUGMENTATION_CONFIGS = {
    'perspective_prob': 0.5,
    'scale_jitter_prob': 0.5,
    'motion_blur_prob': 0.3,
    'shadow_prob': 0.5,
}


TRAIN_CONFIGS = {
    'epochs': 300,
    'batch_size': 16,
    'img_size': 640,
    'optimizer': 'MuSGD',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'box_loss_gain': 7.5,
    'cls_loss_gain': 0.5,
    'dfl_loss_gain': 1.5,
}


def load_yaml_config(config_path: str) -> dict:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_yaml_config(config: dict, save_path: str):
    """
    保存配置到 YAML 文件

    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)


def check_gpu_memory():
    """检查 GPU 内存状态"""
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    total_mem = props.total_memory / (1024**3)
    allocated_mem = torch.cuda.memory_allocated(device) / (1024**3)
    reserved_mem = torch.cuda.memory_reserved(device) / (1024**3)
    free_mem = total_mem - allocated_mem

    return {
        'device_name': props.name,
        'total_memory_gb': total_mem,
        'allocated_memory_gb': allocated_mem,
        'reserved_memory_gb': reserved_mem,
        'free_memory_gb': free_mem,
    }


def estimate_batch_size(gpu_mem_gb: float, img_size: int = 640, model_size: str = 's') -> int:
    """
    根据 GPU 内存估算合适的批次大小

    Args:
        gpu_mem_gb: GPU 内存大小 (GB)
        img_size: 输入图像尺寸
        model_size: 模型大小 (n/s/m/l/x)
    Returns:
        推荐的批次大小
    """
    mem_per_sample = {
        'n': 0.5,
        's': 1.2,
        'm': 2.5,
        'l': 4.0,
        'x': 7.0,
    }

    base_mem = mem_per_sample.get(model_size, 1.2)

    scale_factor = (img_size / 640) ** 2

    recommended_batch = int((gpu_mem_gb * 0.7) / (base_mem * scale_factor))

    recommended_batch = max(1, min(recommended_batch, 64))

    return recommended_batch


def print_model_info(model_size: str = 's'):
    """打印模型配置信息"""
    if model_size not in CASA_NET_CONFIGS:
        model_size = 's'

    config = CASA_NET_CONFIGS[model_size]

    print("\n" + "=" * 50)
    print(f"CASA-Net ({model_size}) 模型配置")
    print("=" * 50)
    print(f"深度因子: {config['depth_multiple']}")
    print(f"宽度因子: {config['width_multiple']}")
    print(f"最大通道数: {config['max_channels']}")
    print(f"参数量: {config['param_count']}")
    print(f"计算量: {config['gflops']}")
    print("=" * 50)

    print("\nE-STAL 配置:")
    for k, v in E_STAL_CONFIGS.items():
        print(f"  {k}: {v}")

    print("\nMFEM 配置:")
    for k, v in MFEM_CONFIGS.items():
        print(f"  {k}: {v}")

    print("\n训练配置:")
    for k, v in TRAIN_CONFIGS.items():
        print(f"  {k}: {v}")

    print("=" * 50)
