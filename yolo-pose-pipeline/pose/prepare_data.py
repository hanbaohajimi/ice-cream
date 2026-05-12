import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_config():
    """向上查找 config.yaml，返回解析结果；找不到则返回空字典。"""
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
_REPO_ROOT = SCRIPT_DIR.parent  # config.yaml 所在目录

POSE_CLASS_FILE = _REPO_ROOT / _PCFG.get("pose_classes_file", "pose/dataset/pose_classes.yaml")
MAX_KPTS = _PCFG.get("max_keypoints", 4)


def _load_pose_classes(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    classes = cfg.get("classes", {})
    class_names = list(classes.keys())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return class_names, class_to_idx


def _default_yaml_path(dst_root: Path) -> Path:
    """Infer yaml file name from dst_root (e.g. yolo_dataset421 -> data421.yaml)."""
    name = dst_root.name
    prefix = "yolo_dataset"
    if name.startswith(prefix):
        suffix = name[len(prefix):]
        if suffix:
            return SCRIPT_DIR / f"data{suffix}.yaml"
    return SCRIPT_DIR / "data.yaml"


def write_data_yaml(
    dst_root: Path,
    class_names,
    yaml_out: Path | None = None,
):
    """Write Ultralytics dataset yaml for the exported YOLO-pose dataset."""
    dst_root = Path(dst_root).resolve()
    out_path = Path(yaml_out) if yaml_out is not None else _default_yaml_path(dst_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "path": str(dst_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [MAX_KPTS, 3],
        "flip_idx": list(range(MAX_KPTS)),
        "names": {i: n for i, n in enumerate(class_names)},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote dataset yaml: {out_path}")


def _norm_bbox_from_rectangle(points, img_w: int, img_h: int):
    # 兼容 2 点对角框与 4 点角点框，统一取 min/max 包围盒。
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def _kpts_to_fixed(points, img_w: int, img_h: int, k_max: int = MAX_KPTS):
    # points: list[(label, x, y)]
    if not points:
        pts = []
    else:
        def _pidx(lbl: str):
            s = lbl.strip().lower()
            if s.startswith("p") and s[1:].isdigit():
                return int(s[1:])
            return 10_000

        if any(_pidx(lbl) < 10_000 for lbl, _, _ in points):
            # 优先按 p1,p2,p3,p4... 的语义顺序
            pts = sorted(points, key=lambda t: (_pidx(t[0]), t[2], t[1]))
        elif any(lbl.strip().lower() == "center" for lbl, _, _ in points):
            pts = sorted(
                points,
                key=lambda t: (
                    0 if t[0].strip().lower() == "center" else 1,
                    t[2],
                    t[1],
                ),
            )
        else:
            pts = sorted(points, key=lambda t: (t[2], t[1]))

    vals = []
    for i in range(k_max):
        if i < len(pts):
            _, x, y = pts[i]
            vals.extend([x / img_w, y / img_h, 2])  # visible
        else:
            vals.extend([0.0, 0.0, 0])  # padded invisible
    return vals


def convert_labelme_json(json_file, class_to_idx):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = int(data["imageWidth"])
    img_h = int(data["imageHeight"])
    shapes = data.get("shapes", [])

    # group_id -> {"cls": str, "bbox": tuple, "points": list[(x,y)]}
    groups = defaultdict(lambda: {"cls": None, "bbox": None, "points": []})
    auto_gid = -1

    for shp in shapes:
        label = str(shp.get("label", "")).strip()
        gid = shp.get("group_id")
        if gid is None:
            auto_gid -= 1
            gid = auto_gid
        rec = groups[gid]

        st = shp.get("shape_type", "")
        pts = shp.get("points", [])
        if st == "rectangle" and len(pts) >= 2:
            # 类别由矩形标注给出（Triangle/Circle/Trapezium/Rectangle）
            if label in class_to_idx:
                rec["cls"] = label
            rec["bbox"] = _norm_bbox_from_rectangle(pts, img_w, img_h)
        elif st == "point" and len(pts) >= 1:
            # 点标签可能是 p1/p2/.../center，不要求与类别名一致，按 group_id 归属目标
            rec["points"].append((label, float(pts[0][0]), float(pts[0][1])))

    lines = []
    for gid, rec in groups.items():
        cls_name = rec["cls"]
        bbox = rec["bbox"]
        if cls_name is None or bbox is None:
            continue
        cls_idx = class_to_idx[cls_name]
        kpt_flat = _kpts_to_fixed(rec["points"], img_w, img_h, k_max=MAX_KPTS)
        line = [cls_idx, *bbox, *kpt_flat]
        lines.append(" ".join(map(str, line)))

        if len(rec["points"]) > MAX_KPTS:
            print(
                f"Warning: {json_file.name} group {gid} class={cls_name} "
                f"has {len(rec['points'])} keypoints, truncated to {MAX_KPTS}"
            )
    if not lines:
        print(f"Warning: No valid labeled object in {json_file}")
    return lines


def prepare_dataset(
    src_root,
    dst_root,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    pose_classes_path: Path = POSE_CLASS_FILE,
    write_yaml: bool = True,
    yaml_out: Path | None = None,
):
    """Split images into train / val / test and write YOLO-pose .txt labels."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    src_images = src_root / "images"
    src_labels = src_root / "labels"
    class_names, class_to_idx = _load_pose_classes(pose_classes_path)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < -1e-6:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    splits = {
        "train": dst_root / "images" / "train",
        "val": dst_root / "images" / "val",
        "test": dst_root / "images" / "test",
    }
    lbl_splits = {
        "train": dst_root / "labels" / "train",
        "val": dst_root / "labels" / "val",
        "test": dst_root / "labels" / "test",
    }
    for p in list(splits.values()) + list(lbl_splits.values()):
        p.mkdir(parents=True, exist_ok=True)

    image_files = (
        list(src_images.glob("*.jpg"))
        + list(src_images.glob("*.jpeg"))
        + list(src_images.glob("*.png"))
        + list(src_images.glob("*.bmp"))
    )
    image_files.sort()
    n = len(image_files)
    if n == 0:
        raise FileNotFoundError(f"No images found under {src_images}")

    random.seed(seed)
    random.shuffle(image_files)

    # Slice indices: [0, i1) train, [i1, i2) val, [i2, n) test
    i1 = max(1, int(n * train_ratio)) if n else 0
    i2 = max(i1, int(n * (train_ratio + val_ratio)))
    i2 = min(i2, n)
    i1 = min(i1, n)

    train_files = image_files[:i1]
    val_files = image_files[i1:i2]
    test_files = image_files[i2:]

    for img_path in train_files:
        _process_file(
            img_path, src_labels, src_images, splits["train"], lbl_splits["train"], class_to_idx
        )
    for img_path in val_files:
        _process_file(
            img_path, src_labels, src_images, splits["val"], lbl_splits["val"], class_to_idx
        )
    for img_path in test_files:
        _process_file(
            img_path, src_labels, src_images, splits["test"], lbl_splits["test"], class_to_idx
        )

    print(f"Dataset prepared at {dst_root}")
    print(f"  train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")
    if write_yaml:
        write_data_yaml(dst_root, class_names, yaml_out=yaml_out)

def _process_file(img_path, src_labels, src_images, dst_img_dir, dst_lbl_dir, class_to_idx):
    # Copy image
    shutil.copy(img_path, dst_img_dir / img_path.name)

    # Process label
    label_name = img_path.stem + ".json"
    label_path = src_labels / label_name
    if not label_path.exists():
        # dataset417 的 json 在 images/ 下
        label_path = src_images / label_name

    if label_path.exists():
        yolo_lines = convert_labelme_json(label_path, class_to_idx)
        if yolo_lines:
            txt_name = img_path.stem + ".txt"
            with open(dst_lbl_dir / txt_name, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")
    else:
        print(f"Warning: Label file not found for {img_path.name}")


def split_yolo_labels(
    src_root: Path,
    dst_root: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """仅做 train/val/test 分割（已有 YOLO txt），不做 JSON 转换。
    原 split_only.py 的功能已合并至此函数。
    """
    src_images = src_root / "images"
    src_labels = src_root / "labels"
    if not src_images.exists() or not src_labels.exists():
        raise FileNotFoundError(f"Need {src_images} and {src_labels}")

    imgs = (
        list(src_images.glob("*.jpg")) + list(src_images.glob("*.jpeg"))
        + list(src_images.glob("*.png")) + list(src_images.glob("*.bmp"))
        + list(src_images.glob("*.webp"))
    )
    imgs.sort()
    if not imgs:
        raise FileNotFoundError(f"No images under {src_images}")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < -1e-9:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    random.seed(seed)
    random.shuffle(imgs)
    n = len(imgs)
    i1 = min(max(1, int(n * train_ratio)), n)
    i2 = min(max(i1, int(n * (train_ratio + val_ratio))), n)
    split_map = {"train": imgs[:i1], "val": imgs[i1:i2], "test": imgs[i2:]}

    for split in ("train", "val", "test"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    missing_labels = []
    for split, files in split_map.items():
        out_img = dst_root / "images" / split
        out_lbl = dst_root / "labels" / split
        for img in files:
            lbl = src_labels / f"{img.stem}.txt"
            shutil.copy2(img, out_img / img.name)
            if lbl.exists():
                shutil.copy2(lbl, out_lbl / lbl.name)
            else:
                missing_labels.append(img.name)

    print(f"Split done -> {dst_root}")
    print(f"  train: {len(split_map['train'])}  val: {len(split_map['val'])}  test: {len(split_map['test'])}")
    if missing_labels:
        print(f"Warning: missing labels for {len(missing_labels)} images")
    else:
        print("All images have matching txt labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "数据集准备工具。\n"
            "  --mode convert    : LabelMe JSON → YOLO txt，然后 train/val/test 分割（默认）\n"
            "  --mode split-only : 跳过 JSON 转换，直接对已有 YOLO txt 做分割（原 split_only.py 功能）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["convert", "split-only"],
        default=_PCFG.get("mode", "convert"),
        help="运行模式（默认 convert）",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=_REPO_ROOT / _PCFG.get("src_dir", "pose/dataset"),
        help="Source folder with images/ and labels/*.json (convert mode) or labels/*.txt (split-only mode)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=_REPO_ROOT / _PCFG.get("dst_dir", "pose/yolo_dataset"),
        help="Output YOLO dataset root",
    )
    parser.add_argument("--train", type=float, default=_PCFG.get("train_ratio", 0.8), help="Train fraction")
    parser.add_argument("--val", type=float, default=_PCFG.get("val_ratio", 0.1), help="Val fraction")
    parser.add_argument(
        "--seed", type=int, default=_PCFG.get("seed", 42), help="Shuffle seed for reproducible splits"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove dst root before writing (fresh export)",
    )
    parser.add_argument(
        "--pose-classes",
        type=Path,
        default=POSE_CLASS_FILE,
        help="pose_classes.yaml (class -> keypoint names)，仅 convert 模式使用",
    )
    parser.add_argument(
        "--write-yaml",
        dest="write_yaml",
        action="store_true",
        default=True,
        help="Generate data*.yaml for the exported dataset (default: enabled)",
    )
    parser.add_argument(
        "--no-write-yaml",
        dest="write_yaml",
        action="store_false",
        help="Disable auto generation of data*.yaml",
    )
    parser.add_argument(
        "--yaml-out",
        type=Path,
        default=None,
        help="Optional output path for generated yaml (default auto: data<suffix>.yaml)",
    )
    args = parser.parse_args()
    if args.clear and args.dst.exists():
        shutil.rmtree(args.dst)

    if args.mode == "split-only":
        split_yolo_labels(
            src_root=args.src,
            dst_root=args.dst,
            train_ratio=args.train,
            val_ratio=args.val,
            seed=args.seed,
        )
    else:
        prepare_dataset(
            args.src,
            args.dst,
            train_ratio=args.train,
            val_ratio=args.val,
            seed=args.seed,
            pose_classes_path=args.pose_classes,
            write_yaml=args.write_yaml,
            yaml_out=args.yaml_out,
        )
