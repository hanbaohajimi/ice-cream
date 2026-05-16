#!/usr/bin/env python3
"""
从 data_video 目录下的视频中按固定帧率抽帧（默认每秒 2 帧）。
每张视频单独输出到子目录，文件名带序号。

配置：优先读取同目录 config.yaml 中 extract_frames 节，CLI 参数可覆盖。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".MP4", ".AVI", ".MOV"}


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


def extract_video(
    video_path: Path,
    out_root: Path,
    fps_target: float,
    image_ext: str = ".jpg",
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] 无法打开: {video_path}")
        return 0

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if src_fps <= 1e-3:
        src_fps = 25.0

    # 每隔多少帧保存一张，使平均速率接近 fps_target
    frame_interval = max(1, int(round(src_fps / fps_target)))
    stem = video_path.stem
    vid_out = out_root / stem
    vid_out.mkdir(parents=True, exist_ok=True)

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            out_path = vid_out / f"{stem}_{saved:06d}{image_ext}"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1

    cap.release()
    eff_fps = src_fps / frame_interval
    print(
        f"{video_path.name}: 源 FPS≈{src_fps:.2f}, 间隔={frame_interval} 帧, "
        f"保存 {saved} 张 (约 {eff_fps:.2f} 帧/秒)"
    )
    return saved


def main() -> None:
    _cfg = _load_config().get("extract_frames", {})
    _root = Path(__file__).resolve().parent
    default_input = _root / _cfg.get("input_dir", "data_video")
    default_output = _root / _cfg.get("output_dir", "data_video_frames")

    p = argparse.ArgumentParser(description="从目录中批量抽帧（默认 2 帧/秒）")
    p.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="视频目录",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="输出根目录（每个视频一个子文件夹）",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=_cfg.get("fps", 2.0),
        help="目标抽帧率（帧/秒）",
    )
    p.add_argument(
        "--ext",
        type=str,
        default=_cfg.get("ext", ".jpg"),
        help="输出图片扩展名，如 .jpg / .png",
    )
    args = p.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"输入目录不存在: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        f
        for f in args.input.iterdir()
        if f.is_file() and f.suffix in VIDEO_EXTS
    )
    if not videos:
        print(f"在 {args.input} 中未找到支持的视频 ({', '.join(sorted(VIDEO_EXTS))})")
        return

    total = 0
    for v in videos:
        total += extract_video(v, args.output, args.fps, image_ext=args.ext)

    print(f"完成，共保存 {total} 张图，输出目录: {args.output}")


if __name__ == "__main__":
    main()
