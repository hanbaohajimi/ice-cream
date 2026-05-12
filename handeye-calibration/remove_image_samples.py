#!/usr/bin/env python3
"""Remove calibration samples by image index, renumber jpg / poses.txt / capture_meta.jsonl.

Usage:
  python3 remove_image_samples.py --remove 0,2,19,24
  python3 remove_image_samples.py --images-dir /path/to/images --remove 0,2,19,24
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


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


def main() -> None:
    _cfg = _load_config().get("capture", {})
    _default_img_dir = Path(__file__).resolve().parent / _cfg.get("images_dir", "images")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images-dir",
        type=Path,
        default=_default_img_dir,
        help="directory containing N.jpg, poses.txt, capture_meta.jsonl",
    )
    p.add_argument(
        "--remove",
        type=str,
        default="0,2,19,24",
        help="comma-separated image indices to drop",
    )
    args = p.parse_args()
    img_dir: Path = args.images_dir.expanduser().resolve()
    remove = {int(x.strip()) for x in args.remove.split(",") if x.strip()}

    jpg_ids: list[int] = []
    for path in img_dir.glob("*.jpg"):
        m = re.match(r"^(\d+)\.jpg$", path.name)
        if m:
            jpg_ids.append(int(m.group(1)))
    jpg_ids.sort()
    if not jpg_ids:
        raise SystemExit(f"no *.jpg under {img_dir}")

    missing_remove = remove - set(jpg_ids)
    if missing_remove:
        print(f"[warn] remove set not all present as files: {sorted(missing_remove)}")

    kept_old = [i for i in jpg_ids if i not in remove]
    k = len(kept_old)
    print(f"[info] jpg count={len(jpg_ids)} remove={sorted(remove)} kept={k} new indices 0..{k - 1}")

    # --- images: stage1 -> __tmp_{new}.jpg ---
    for new_i, old_i in enumerate(kept_old):
        src = img_dir / f"{old_i}.jpg"
        if not src.is_file():
            raise SystemExit(f"missing expected file: {src}")
        dst = img_dir / f"__tmp_{new_i}.jpg"
        if dst.exists():
            raise SystemExit(f"collision: {dst} already exists")
        src.rename(dst)

    # delete removed originals (still on disk if not in kept_old)
    for old_i in remove:
        leftover = img_dir / f"{old_i}.jpg"
        if leftover.is_file():
            leftover.unlink()

    # stage2: __tmp_{i}.jpg -> {i}.jpg (clear high indices first to avoid clashes)
    for new_i in range(k - 1, -1, -1):
        src = img_dir / f"__tmp_{new_i}.jpg"
        dst = img_dir / f"{new_i}.jpg"
        if dst.exists() and src != dst:
            dst.unlink()
        src.rename(dst)

    # --- poses.txt: line i matches image i.jpg ---
    pose_path = img_dir / "poses.txt"
    if not pose_path.is_file():
        raise SystemExit(f"missing {pose_path}")
    lines = [ln.rstrip("\n") for ln in pose_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < max(jpg_ids) + 1:
        print(f"[warn] poses.txt has {len(lines)} lines, expected at least {max(jpg_ids) + 1}")
    new_lines = []
    for old_i in kept_old:
        if old_i >= len(lines):
            raise SystemExit(f"poses.txt has only {len(lines)} lines, need line for image index {old_i}")
        new_lines.append(lines[old_i])
    pose_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[info] wrote {pose_path} ({len(new_lines)} lines)")

    # --- capture_meta.jsonl ---
    meta_path = img_dir / "capture_meta.jsonl"
    if not meta_path.is_file():
        raise SystemExit(f"missing {meta_path}")
    records: list[dict] = []
    for ln in meta_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        records.append(json.loads(ln))
    by_index = {int(r["index"]): r for r in records}
    for idx in remove:
        by_index.pop(idx, None)
    new_records: list[dict] = []
    for new_i, old_i in enumerate(kept_old):
        r = by_index.get(old_i)
        if r is None:
            print(f"[warn] no meta line for old index {old_i}, skipping")
            continue
        r = dict(r)
        r["old_index"] = old_i
        r["index"] = new_i
        r["image"] = f"{new_i}.jpg"
        new_records.append(r)
    meta_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in new_records) + "\n", encoding="utf-8")
    print(f"[info] wrote {meta_path} ({len(new_records)} lines)")

    # --- robotToolPose.csv in eyeInHand package root ---
    repo_eye = img_dir.parent
    csv_path = repo_eye / "robotToolPose.csv"
    if csv_path.is_file():
        csv_path.unlink()
        print(f"[info] removed stale {csv_path} (regenerate with eyeInHand.py poses_save_csv)")

    print("[done] next capture: use --reset or set next_index to", k, "if your tool tracks it manually")


if __name__ == "__main__":
    main()
