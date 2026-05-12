#!/usr/bin/env python3
# coding=utf-8
"""Online image + robot pose collector for eye-in-hand calibration.

Output format follows eyeInHand.py:
  images/0.jpg
  images/1.jpg
  images/poses.txt

Each poses.txt line is:
  x,y,z,Rx,Ry,Rz

Units:
  translation: meter
  rotation: radian

Rotation convention:
  feedback.link5_hmat is treated as T_ee2base.
  Its rotation is decomposed as intrinsic XYZ, then saved as columns Rx,Ry,Rz.
  The saved angles reconstruct the matrix as R = Rx @ Ry @ Rz, which matches
  the current eyeInHand.py pose_to_homogeneous_matrix convention.

配置：优先读取同目录 config.yaml 中 capture 节，CLI 参数可覆盖。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import websockets
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, Image


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


_CFG = _load_config().get("capture", {})


@dataclass
class FrameSnapshot:
    image: np.ndarray
    recv_monotonic: float


@dataclass
class PoseSnapshot:
    T_ee2base: np.ndarray
    recv_monotonic: float
    seq: Optional[int]
    age_ms: Optional[float]


def hmat_to_pose_xyz_rpy_xyz(T_ee2base: np.ndarray) -> np.ndarray:
    """Return [x,y,z,Rx,Ry,Rz] in meters/radians from T_ee2base.

    SciPy ``as_euler("XYZ")`` returns [Rx, Ry, Rz] for the same rotation
    composition used by eyeInHand.py: R = Rx @ Ry @ Rz.
    """
    T = np.array(T_ee2base, dtype=np.float64).reshape(4, 4)
    xyz = T[:3, 3].astype(np.float64)
    rx, ry, rz = Rotation.from_matrix(T[:3, :3]).as_euler("XYZ", degrees=False)
    return np.array([xyz[0], xyz[1], xyz[2], rx, ry, rz], dtype=np.float64)


def validate_hmat(raw) -> Optional[np.ndarray]:
    try:
        mat = np.array(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if mat.shape != (4, 4):
        return None
    if not np.isfinite(mat).all():
        return None
    if not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        return None
    R = mat[:3, :3]
    if abs(float(np.linalg.det(R)) - 1.0) > 0.05:
        return None
    return mat


class Link5WebSocketReader:
    def __init__(self, uri: str):
        self.uri = uri
        self._lock = threading.Lock()
        self._snapshot: Optional[PoseSnapshot] = None
        self._connected = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="link5_ws_reader")

    def start(self) -> None:
        self._thread.start()

    def snapshot(self) -> tuple[Optional[PoseSnapshot], bool]:
        with self._lock:
            snap = self._snapshot
            connected = self._connected
        if snap is None:
            return None, connected
        return (
            PoseSnapshot(
                T_ee2base=snap.T_ee2base.copy(),
                recv_monotonic=snap.recv_monotonic,
                seq=snap.seq,
                age_ms=snap.age_ms,
            ),
            connected,
        )

    def _run(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        while True:
            try:
                async with websockets.connect(self.uri, ping_interval=10, ping_timeout=10) as ws:
                    with self._lock:
                        self._connected = True
                    async for raw in ws:
                        self._update(raw)
            except asyncio.CancelledError:
                return
            except Exception:
                with self._lock:
                    self._connected = False
                    self._snapshot = None
                await asyncio.sleep(1.0)

    def _update(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        data = msg.get("data") or {}
        feedback = data.get("feedback") or msg.get("feedback") or {}
        udp = data.get("udp") or msg.get("udp") or {}
        hmat_raw = feedback.get("link5_hmat") or msg.get("link5_hmat")
        mat = validate_hmat(hmat_raw)
        if mat is None:
            return

        seq = udp.get("seq")
        age_ms = udp.get("age_ms")
        try:
            seq = None if seq is None else int(seq)
        except (TypeError, ValueError):
            seq = None
        try:
            age_ms = None if age_ms is None else float(age_ms)
        except (TypeError, ValueError):
            age_ms = None

        snap = PoseSnapshot(
            T_ee2base=mat,
            recv_monotonic=time.monotonic(),
            seq=seq,
            age_ms=age_ms,
        )
        with self._lock:
            self._snapshot = snap
            self._connected = True


class ImageCollectorNode(Node):
    def __init__(self, topic: str, use_raw: bool):
        super().__init__("eye_in_hand_online_capture")
        self._lock = threading.Lock()
        self._latest: Optional[FrameSnapshot] = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        msg_type = Image if use_raw else CompressedImage
        self.create_subscription(msg_type, topic, self._on_image, qos)
        self.get_logger().info(f"subscribed image topic: {topic}")

    def latest_frame(self) -> Optional[FrameSnapshot]:
        with self._lock:
            snap = self._latest
        if snap is None:
            return None
        return FrameSnapshot(image=snap.image.copy(), recv_monotonic=snap.recv_monotonic)

    def _on_image(self, msg) -> None:
        try:
            if isinstance(msg, CompressedImage):
                image = compressed_to_bgr8(msg)
            else:
                image = imgmsg_to_bgr8(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"image decode failed: {exc}")
            return
        with self._lock:
            self._latest = FrameSnapshot(image=image, recv_monotonic=time.monotonic())


def compressed_to_bgr8(msg: CompressedImage) -> np.ndarray:
    try:
        buf = np.frombuffer(memoryview(msg.data), dtype=np.uint8)
    except TypeError:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cv2.imdecode failed, format={msg.format!r}")
    return image


def imgmsg_to_bgr8(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    dtype_map = {
        "mono8": np.uint8,
        "8uc1": np.uint8,
        "bgr8": np.uint8,
        "rgb8": np.uint8,
        "8uc3": np.uint8,
        "mono16": np.uint16,
        "16uc1": np.uint16,
        "32fc1": np.float32,
    }
    dt = dtype_map.get(enc, np.uint8)
    try:
        buf = np.frombuffer(memoryview(msg.data), dtype=dt)
    except TypeError:
        buf = np.frombuffer(bytes(msg.data), dtype=dt)
    channels = msg.step // msg.width // np.dtype(dt).itemsize
    image = buf.reshape((msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width))
    if enc == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif enc in ("mono8", "8uc1"):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif enc in ("mono16", "16uc1"):
        image = cv2.cvtColor((image >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif enc == "32fc1":
        image = cv2.cvtColor(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def prepare_output_dir(images_dir: Path, reset: bool) -> int:
    if reset and images_dir.exists():
        for jpg in images_dir.glob("*.jpg"):
            jpg.unlink()
        poses = images_dir / "poses.txt"
        if poses.exists():
            poses.unlink()
        meta = images_dir / "capture_meta.jsonl"
        if meta.exists():
            meta.unlink()

    images_dir.mkdir(parents=True, exist_ok=True)
    pose_path = images_dir / "poses.txt"
    image_indices = sorted(
        int(p.stem)
        for p in images_dir.glob("*.jpg")
        if p.stem.isdigit()
    )
    pose_count = 0
    if pose_path.exists():
        pose_count = len([line for line in pose_path.read_text(encoding="utf-8").splitlines() if line.strip()])

    if image_indices:
        expected = list(range(max(image_indices) + 1))
        if image_indices != expected:
            raise RuntimeError(f"images dir has non-contiguous numeric jpg files: {image_indices[:5]} ... {image_indices[-5:]}")
        next_index = max(image_indices) + 1
    else:
        next_index = 0

    if pose_count != next_index:
        raise RuntimeError(
            f"existing data mismatch: jpg_count={next_index}, poses_lines={pose_count}. "
            "Fix images/poses.txt or rerun with --reset."
        )
    return next_index


def append_pose_line(path: Path, pose: np.ndarray) -> None:
    line = ",".join(f"{float(v):.12g}" for v in pose)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def append_meta(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    _default_img_dir = str(script_dir / _CFG.get("images_dir", "images"))
    _default_compressed = _CFG.get("rgb_compressed_topic",
        "/ascamera_hp60c/camera_publisher/rgb0/image/compressed")
    parser = argparse.ArgumentParser(
        description="Collect paired calibration images and T_ee2base poses for eyeInHand.py"
    )
    parser.add_argument("--images-dir", default=_default_img_dir, help="output directory containing jpg files and poses.txt")
    parser.add_argument("--ws-host", default=_CFG.get("ws_host", "192.168.31.142"), help="websocket host")
    parser.add_argument("--ws-port", type=int, default=_CFG.get("ws_port", 8765), help="websocket port")
    parser.add_argument("--image-topic", default=_default_compressed, help="RGB image topic")
    parser.add_argument("--raw-image", action="store_true", help="subscribe raw sensor_msgs/Image instead of CompressedImage")
    parser.add_argument("--max-image-age-s", type=float, default=_CFG.get("max_image_age_s", 0.5), help="reject capture if latest image is older than this")
    parser.add_argument("--max-pose-age-s", type=float, default=_CFG.get("max_pose_age_s", 0.5), help="reject capture if latest websocket pose is older than this")
    parser.add_argument("--max-skew-s", type=float, default=_CFG.get("max_skew_s", 0.25), help="reject capture if image/ws receive time differs more than this")
    parser.add_argument("--reset", action="store_true", help="remove existing jpg/poses.txt in images-dir before capture")
    parser.add_argument("--no-preview", action="store_true", help="run without cv2 preview window")
    return parser.parse_args()


def draw_status(image: np.ndarray, text_lines: list[str]) -> np.ndarray:
    out = image.copy()
    y = 28
    for text in text_lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return out


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    next_index = prepare_output_dir(images_dir, reset=bool(args.reset))
    pose_path = images_dir / "poses.txt"
    meta_path = images_dir / "capture_meta.jsonl"

    topic = args.image_topic
    _default_compressed = _CFG.get("rgb_compressed_topic",
        "/ascamera_hp60c/camera_publisher/rgb0/image/compressed")
    _default_raw = _CFG.get("rgb_raw_topic",
        "/ascamera_hp60c/camera_publisher/rgb0/image")
    if args.raw_image and topic == _default_compressed:
        topic = _default_raw

    uri = f"ws://{args.ws_host}:{args.ws_port}"
    ws_reader = Link5WebSocketReader(uri)
    ws_reader.start()

    rclpy.init()
    node = ImageCollectorNode(topic=topic, use_raw=bool(args.raw_image))

    print(f"[info] output images dir: {images_dir}")
    print(f"[info] pose file: {pose_path}")
    print(f"[info] websocket: {uri}")
    print("[info] keys: c/space=Capture, q/ESC=Quit")
    print("[info] pose columns: x,y,z,Rx,Ry,Rz  units: m,rad  rotation order: XYZ (R=Rx@Ry@Rz)")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            frame = node.latest_frame()
            pose_snap, ws_connected = ws_reader.snapshot()
            now = time.monotonic()

            image_age = None if frame is None else now - frame.recv_monotonic
            pose_age = None if pose_snap is None else now - pose_snap.recv_monotonic
            skew = None if frame is None or pose_snap is None else abs(frame.recv_monotonic - pose_snap.recv_monotonic)

            status = [
                f"next={next_index}  c/space=capture  q/esc=quit",
                f"img_age={'--' if image_age is None else f'{image_age:.3f}s'}  "
                f"ws={'OK' if ws_connected else 'DISCONNECTED'}  "
                f"pose_age={'--' if pose_age is None else f'{pose_age:.3f}s'}  "
                f"skew={'--' if skew is None else f'{skew:.3f}s'}",
            ]

            if not args.no_preview and frame is not None:
                cv2.imshow("eyeInHand online capture", draw_status(frame.image, status))

            key = cv2.waitKey(1) & 0xFF if not args.no_preview else -1
            if key in (ord("q"), 27):
                break
            if key not in (ord("c"), ord(" "), 13):
                continue

            if frame is None:
                print("[reject] no image frame yet")
                continue
            if pose_snap is None:
                print("[reject] no websocket link5_hmat yet")
                continue
            if image_age is None or image_age > args.max_image_age_s:
                print(f"[reject] image stale: age={image_age}")
                continue
            if pose_age is None or pose_age > args.max_pose_age_s:
                print(f"[reject] pose stale: age={pose_age}")
                continue
            if skew is None or skew > args.max_skew_s:
                print(f"[reject] image/ws skew too large: skew={skew}")
                continue

            image_path = images_dir / f"{next_index}.jpg"
            pose = hmat_to_pose_xyz_rpy_xyz(pose_snap.T_ee2base)
            ok = cv2.imwrite(str(image_path), frame.image)
            if not ok:
                print(f"[reject] failed to write image: {image_path}")
                continue

            append_pose_line(pose_path, pose)
            append_meta(
                meta_path,
                {
                    "index": next_index,
                    "image": image_path.name,
                    "pose": pose.tolist(),
                    "ws_seq": pose_snap.seq,
                    "ws_age_ms": pose_snap.age_ms,
                    "image_age_s": image_age,
                    "pose_age_s": pose_age,
                    "image_ws_skew_s": skew,
                    "T_ee2base": pose_snap.T_ee2base.tolist(),
                    "created_epoch": time.time(),
                },
            )
            print(
                f"[saved] {image_path.name}  "
                f"pose=({pose[0]:+.5f},{pose[1]:+.5f},{pose[2]:+.5f},"
                f"{pose[3]:+.5f},{pose[4]:+.5f},{pose[5]:+.5f})"
            )
            next_index += 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
