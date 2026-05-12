#!/usr/bin/env python3
"""
Realtime YOLO Pose detection from webcam.

配置：优先读取上级目录 config.yaml 中 inference 节，CLI 参数可覆盖。

Example:
  python3 realtime_detect_pose.py
  python3 realtime_detect_pose.py --weights pose/runs/pose_training/weights/best.pt
  python3 realtime_detect_pose.py --source 0 --imgsz 640 --conf 0.25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
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


def parse_args():
    _cfg = _load_config().get("inference", {})
    _repo_root = Path(__file__).resolve().parent.parent
    _default_weights = str(_repo_root / _cfg.get("weights_path",
        "pose/runs/pose_training/weights/best.pt"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=_default_weights,
        help="Path to trained pose weights (.pt)",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=_cfg.get("camera_source", 0),
        help="Webcam index, e.g. 0",
    )
    parser.add_argument("--imgsz", type=int, default=_cfg.get("imgsz", 640), help="Inference image size")
    parser.add_argument("--conf", type=float, default=_cfg.get("conf_threshold", 0.25), help="Confidence threshold")
    parser.add_argument(
        "--device",
        type=str,
        default=_cfg.get("device", ""),
        help='Inference device: "", "cpu", "0", "0,1"... (empty=auto)',
    )
    parser.add_argument(
        "--window",
        type=str,
        default="YOLO Pose Realtime (q=quit)",
        help="Display window title",
    )
    return parser.parse_args()


def pick_device(user_device: str) -> str | int:
    if user_device:
        if user_device.isdigit():
            return int(user_device)
        return user_device
    return 0 if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")

    device = pick_device(args.device)
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera source: {args.source}")

    t_last = time.time()
    fps_ema = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, exiting.")
                break

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )
            r = results[0]
            vis = r.plot()

            # Lightweight FPS display
            t_now = time.time()
            dt = max(t_now - t_last, 1e-6)
            fps = 1.0 / dt
            fps_ema = fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * fps)
            t_last = t_now
            cv2.putText(
                vis,
                f"FPS: {fps_ema:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(args.window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        try:
            cv2.destroyWindow(args.window)
        except cv2.error:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
