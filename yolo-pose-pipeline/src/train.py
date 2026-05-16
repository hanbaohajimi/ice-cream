import os
from pathlib import Path

import torch
from ultralytics import YOLO


def _load_config():
    """向上查找 config.yaml，返回解析结果；找不到则返回空字典。"""
    import yaml
    here = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = here / "config.yaml"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        here = here.parent
    return {}


def train():
    cfg = _load_config().get("train", {})
    root = Path(__file__).resolve().parent
    # config.yaml 中路径相对于仓库根（config.yaml 所在目录）
    repo_root = root.parent

    pretrained   = cfg.get("pretrained_weights", "yolo11n-pose.pt")
    data_yaml    = str(repo_root / cfg.get("data_yaml", "data.yaml"))
    epochs       = cfg.get("epochs", 100)
    imgsz        = cfg.get("imgsz", 640)
    batch        = cfg.get("batch", 4)
    run_name     = cfg.get("run_name", "pose_training")
    runs_dir     = str(repo_root / cfg.get("runs_dir", "runs"))
    device       = 0 if torch.cuda.is_available() else "cpu"

    os.chdir(root)
    model = YOLO(pretrained)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=runs_dir,
        name=run_name,
        device=device,
    )

if __name__ == '__main__':
    train()
