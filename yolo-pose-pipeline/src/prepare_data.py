import argparse
import random
import shutil
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_config():
    here = SCRIPT_DIR
    for _ in range(4):
        candidate = here / "config.yaml"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        here = here.parent
    return {}


_CFG = _load_config()
_PCFG = _CFG.get("prepare_data", {})
_REPO_ROOT = SCRIPT_DIR.parent


def split_yolo_labels(
    dataset_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """原地分割：将 dataset_root/images/ 和 dataset_root/labels/ 下的平铺文件
    直接移动到 images/train|val|test 和 labels/train|val|test 子目录。
    """
    src_images = dataset_root / "images"
    src_labels = dataset_root / "labels"
    if not src_images.exists():
        raise FileNotFoundError(f"找不到 {src_images}")

    imgs = sorted(
        f for f in src_images.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not imgs:
        raise FileNotFoundError(f"{src_images} 下没有图片文件")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < -1e-9:
        raise ValueError("train_ratio + val_ratio 之和不能超过 1.0")

    random.seed(seed)
    random.shuffle(imgs)
    n = len(imgs)
    i1 = min(max(1, int(n * train_ratio)), n)
    i2 = min(max(i1, int(n * (train_ratio + val_ratio))), n)
    split_map = {"train": imgs[:i1], "val": imgs[i1:i2], "test": imgs[i2:]}

    for split in ("train", "val", "test"):
        (src_images / split).mkdir(exist_ok=True)
        if src_labels.exists():
            (src_labels / split).mkdir(exist_ok=True)

    missing_labels = []
    for split, files in split_map.items():
        for img in files:
            shutil.move(str(img), src_images / split / img.name)
            if src_labels.exists():
                lbl = src_labels / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.move(str(lbl), src_labels / split / lbl.name)
                else:
                    missing_labels.append(img.name)

    print(f"原地分割完成：{dataset_root}")
    print(f"  train: {len(split_map['train'])}  val: {len(split_map['val'])}  test: {len(split_map['test'])}")
    if missing_labels:
        print(f"Warning: {len(missing_labels)} 张图片缺少对应 txt 标注")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="原地分割 YOLO 数据集：将平铺的 images/ 和 labels/ 直接分到 train/val/test 子目录。"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_REPO_ROOT / _PCFG.get("src_dir", "dataset"),
        help="数据集根目录（含 images/ 和 labels/），例如 dataset/data516",
    )
    parser.add_argument("--train", type=float, default=_PCFG.get("train_ratio", 0.8))
    parser.add_argument("--val",   type=float, default=_PCFG.get("val_ratio",   0.1))
    parser.add_argument("--seed",  type=int,   default=_PCFG.get("seed",        42))
    args = parser.parse_args()

    split_yolo_labels(
        dataset_root=args.dataset,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
    )
