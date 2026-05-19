"""
CASA-Net 训练脚本

三个核心模块集成方式:
1. MFEM: 通过 casa_net.yaml 模型配置自动注入到 neck 层
2. E-STAL: 通过超参 estal=True 启用，在 TaskAlignedAssigner 中生效
3. 航拍增强: 通过超参 aerial_aug=True 启用，在 v8_transforms 中生效

用法:
    python train_casa.py
    python train_casa.py --epochs 100 --batch-size 8
    python train_casa.py --mode val --model-path runs/train/weights/best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

CASA_YAML = str(Path(__file__).parent / "ultralytics" / "cfg" / "models" / "26" / "casa_net.yaml")
DATA_YAML = str(Path(__file__).parent / "rsod.yaml")
PRETRAINED = str(Path(__file__).parent.parent / "weights" / "yolo26s.pt")


def train(
    data: str = DATA_YAML,
    epochs: int = 300,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = str(Path(__file__).parent / "runs"),
    name: str = "casa_net_train",
    pretrained: str = PRETRAINED,
):
    model = YOLO(CASA_YAML)

    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        pretrained=pretrained,
        optimizer="MuSGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        amp=True,
        cache=True,
        workers=8,
        patience=50,
        verbose=True,
        seed=0,
        cos_lr=True,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_period=-1,
        estal=True,
        estal_min_assign=6,
        estal_spatial_tolerance=0.7,
        estal_threshold_scale=0.20,
        estal_base_threshold=0.3,
        estal_small_threshold=8,
        aerial_aug=True,
        aerial_motion_blur=0.3,
        aerial_shadow=0.5,
        aerial_scale_jitter=0.5,
        aerial_perspective=0.5,
    )
    return results


def validate(
    model_path: str,
    data: str = DATA_YAML,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
):
    model = YOLO(model_path)
    results = model.val(
        data=data,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        split="val",
        save_json=True,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=True,
    )
    print(f"\nmAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CASA-Net Training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--pretrained", type=str, default=PRETRAINED)
    parser.add_argument("--name", type=str, default="casa_net_train")
    args = parser.parse_args()

    if args.mode == "train":
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            pretrained=args.pretrained,
            name=args.name,
        )
    elif args.mode == "val":
        validate(model_path=args.model_path)
