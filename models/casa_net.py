"""
CASA-Net 模型定义

论文: CASA-Net: Context-Aware Small Object Detection Network for UAV Aerial Images
基于 YOLOv26s 架构，添加:
1. E-STAL (Enhanced STAL) 标签分配
2. MFEM (Multi-scale Feature Enhancement Module) 特征增强
3. 航空专用数据增强

注意: MFEM 和 E-STAL 的实际实现已集成到 ultralytics 框架中:
- MFEM: ultralytics/nn/modules/block.py
- E-STAL: ultralytics/utils/tal.py
- 航拍增强: ultralytics/data/augment.py
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from pathlib import Path


class CASAYOLOv26(nn.Module):
    """
    CASA-Net 模型类

    基于 YOLOv26s，集成:
    - MFEM 特征增强模块 (内嵌于 casa_net.yaml，利用跨 FPN 层特征)
    - E-STAL 标签分配 (集成于 ultralytics/utils/tal.py)
    - 航拍数据增强 (集成于 ultralytics/data/augment.py)
    """

    def __init__(
        self,
        cfg: str = "casa_net",
        ch: int = 3,
        nc: int = None,
        verbose: bool = True
    ):
        super().__init__()

        self.nc = nc or 10
        self.verbose = verbose

        cfg_path = Path(__file__).resolve().parents[2] / "ultralytics" / "cfg" / "models" / "26" / "casa_net.yaml"
        self.base_model = YOLO(str(cfg_path))
        self.model = self.base_model.model

    def forward(self, x):
        """前向传播 - 直接使用 YOLO 模型"""
        return self.model(x)

    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        if self.model is not None:
            self.model.train(mode)
        return self

    def eval(self):
        """设置评估模式"""
        return self.train(False)


def build_casa_net_model(
    model_size: str = "s",
    nc: int = 10,
    pretrained: bool = True,
    model_path: str = None,
    verbose: bool = True,
) -> CASAYOLOv26:
    """
    构建 CASA-Net 模型

    Args:
        model_size: 模型大小 (n/s/m/l/x)
        nc: 类别数
        pretrained: 是否使用预训练权重
        model_path: 自定义权重路径
        verbose: 是否打印信息

    Returns:
        CASAYOLOv26 模型实例
    """
    cfg = f"yolo26{model_size}"

    model = CASAYOLOv26(cfg=cfg, nc=nc, verbose=verbose)

    if pretrained and model_path:
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'model' in ckpt:
                state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
            else:
                state_dict = ckpt
            model.model.load_state_dict(state_dict, strict=False)
            if verbose:
                print(f"CASA-Net: Loaded pretrained weights from {model_path}")
        except Exception as e:
            if verbose:
                print(f"CASA-Net: Could not load pretrained weights: {e}")

    return model


def create_casa_trainer(
    model_size: str = "s",
    data_yaml: str = None,
    epochs: int = 300,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "casa_net",
    pretrained: bool = True,
    yolo26_weights: str = None,
):
    """
    创建 CASA-Net 训练器

    CASA-Net 的三个核心模块已集成到 ultralytics 框架:
    1. MFEM: 内嵌于 casa_net.yaml，利用跨 FPN 层特征
    2. E-STAL: 集成于 ultralytics/utils/tal.py
    3. 航拍增强: 集成于 ultralytics/data/augment.py

    Args:
        model_size: 模型大小
        data_yaml: 数据集配置文件
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图像大小
        device: 训练设备
        project: 项目保存路径
        name: 实验名称
        pretrained: 是否使用预训练权重
        yolo26_weights: YOLOv26 预训练权重路径

    Returns:
        训练结果
    """
    print(f"\n初始化 CASA-Net ({model_size})...")

    casa_model = build_casa_net_model(
        model_size=model_size,
        nc=10,
        pretrained=pretrained,
        model_path=yolo26_weights,
    )

    print(f"\n启动 YOLO 训练 (CASA-Net 配置)...")
    print(f"  数据集: {data_yaml}")
    print(f"  轮数: {epochs}")
    print(f"  批次: {batch_size}")
    print(f"  图像尺寸: {img_size}")

    results = casa_model.base_model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=pretrained,
        optimizer="MuSGD",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.4,
        crop_fraction=1.0,
        patience=100,
        save=True,
        save_period=-1,
        cache=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        multi_scale=False,
    )

    return results


if __name__ == "__main__":
    print("CASA-Net 模型构建测试")
    print("=" * 50)

    model = build_casa_net_model(model_size="s", nc=10, pretrained=False)

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model(x)
    print(f"\n前向传播测试: {x.shape} -> 成功")
