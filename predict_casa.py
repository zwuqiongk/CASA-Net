"""
CASA-Net 预测脚本

使用带 MFEM 模块的 CASA-Net 模型进行推理。
预测阶段不启用 E-STAL 和航拍增强（仅训练阶段使用）。

用法:
    python predict_casa.py --model-path runs/casa_net_train/weights/best.pt --source test.jpg
    python predict_casa.py --model-path best.pt --source images/ --name output
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO


def predict(
    model_path: str,
    source: str,
    img_size: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "0",
    save: bool = True,
    save_txt: bool = True,
    project: str = None,
    name: str = "predict",
):
    if project is None:
        project = str(Path(__file__).parent / "runs")

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        device=device,
        save=save,
        save_txt=save_txt,
        save_conf=True,
        project=project,
        name=name,
        max_det=300,
        line_thickness=2,
        embed=None,
    )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CASA-Net Prediction")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default="predict")
    args = parser.parse_args()

    source_path = Path(args.source)

    if source_path.is_file():
        print(f"Predicting single image: {args.source}")
        predict(
            model_path=args.model_path,
            source=args.source,
            img_size=args.img_size,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project=args.project,
            name=args.name,
        )
    elif source_path.is_dir():
        print(f"Predicting folder: {args.source}")
        predict(
            model_path=args.model_path,
            source=str(source_path),
            img_size=args.img_size,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project=args.project,
            name=args.name,
        )
    else:
        print(f"Source not found: {args.source}")
