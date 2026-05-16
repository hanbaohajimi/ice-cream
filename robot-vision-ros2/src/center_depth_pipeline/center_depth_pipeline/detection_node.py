#!/usr/bin/env python3
"""
订阅 RGB 图像，运行 YOLO（pose），发布 2D 中心点。

实时性优化
  - 所有 ROS 参数在 __init__ 中缓存（避免每帧调用 get_parameter 的开销）。
  - 工作线程 + queue(maxsize=1)：回调函数立即返回，仅处理最新帧，旧帧静默丢弃。
  - CUDA 下使用 FP16 半精度推理，GPU 速度提升约 1.5 倍。
  - NumPy 向量化中心点计算。
  - 发布者 QoS：BEST_EFFORT depth=1，最小化 DDS 缓冲延迟。
  - JPEG 解码：对 CompressedImage.data 使用 memoryview（无额外字节拷贝）。
  - predict(..., stream=False)：无生成器开销。
  - log_detections 默认关闭：逐帧 INFO 日志开销较大。

数据有效性优化
  - 稳定计数器：检测结果连续出现 ≥ min_stable_frames 帧后才发布，
    消除单帧误检（假阳性）。
  - 目标消失时清除 EMA 状态，避免重新出现的目标从过期角度缓慢收敛。
  - 角度使用循环 EMA：正确处理 ±90° 边界的环绕。

位姿模式角度约定（dataset421 关键点语义）
  三角形  : kpt0=顶点, kpt1=底边左, kpt2=底边右, kpt3=不可见（填充）
              角度 = 底边方向 kpt1→kpt2
              质心 = 可见关键点（kpt0,1,2）几何均值 = 真实质心
  圆形    : kpt0=圆心, kpt1-3=不可见
              角度 = 0.0（无方向）
              质心 = kpt0
  梯形    : kpt0=短边左, kpt1=短边右（短边/上边）
              kpt2=长边右, kpt3=长边左（长边/下边）
              角度 = 短边（kpt0→kpt1）与长边（kpt3→kpt2，方向对齐）的圆形均值
              质心 = 面积加权：从长边向短边 t=(2b+a)/(3(a+b))
  正方形  : kpt0=前左, kpt1=前右,
              kpt2=后右, kpt3=后左
              角度 = 前边（kpt0→kpt1）与后边（kpt3→kpt2，方向对齐）的圆形均值，
                     EMA 后折叠到 [-45°,45°]（4 重对称）
              质心 = 四角几何均值 = 正方形中心

类别名称别名：
  "trapezoid" 与 "trapezium" 完全等价。
"""

from __future__ import annotations

import math
import queue
import re
import threading
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CompressedImage, Image

# BEST_EFFORT depth=1：最小化 DDS 缓冲，订阅者始终获取最新结果。
_QOS_RESULT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

from center_depth_msgs.msg import Centers2D
from center_depth_pipeline.image_numpy import (
    compressed_imgmsg_to_bgr8,
    imgmsg_to_bgr8,
)
from center_depth_pipeline.param_utils import (
    declare_and_get_bool,
    declare_and_get_float,
    declare_and_get_int,
    declare_and_get_str,
)
from center_depth_pipeline.queue_utils import replace_latest

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]


class DetectionItem(NamedTuple):
    label: str
    score: float
    u: int
    v: int
    angle_rad: float
    cx_norm: float
    cy_norm: float
    kpts_xy: object = None  # np.ndarray (K, 2) or None
    kpts_v: object = None   # np.ndarray (K,)  or None


_NO_ORIENT_CLASSES = {"circle"}

_RECT_ALIASES = {"square"}
_CLASS_ALIASES = {
    "square":                  "square",
    "trapezoid":               "trapezium",
    # 派生类别（底座/盖子）→ 基础形状，复用相同的关键点几何逻辑
    "triangle_pedestal_red":   "triangle",
    "triangle_pedestal_blue":  "triangle",
    "triangle_cover":          "triangle",
    "circle_pedestal_red":     "circle",
    "circle_pedestal_blue":    "circle",
    "trapezium_pedestal_red":  "trapezium",
    "trapezium_pedestal_blue": "trapezium",
    "trapezium_cover":         "trapezium",
    "square_pedestal_red":     "square",
    "square_pedestal_blue":    "square",
}
_ROLE_ALIASES = {
    "circle": {
        "center": {"center", "centre"},
    },
    "triangle": {
        "apex": {"apex", "tip", "vertex", "peak", "top"},
        "base_left": {"base_left", "left_base", "base_l", "bottom_left", "left_bottom"},
        "base_right": {"base_right", "right_base", "base_r", "bottom_right", "right_bottom"},
    },
    "trapezium": {
        "short_base_left": {
            "short_base_left", "short_left", "top_left", "upper_left", "small_base_left",
        },
        "short_base_right": {
            "short_base_right", "short_right", "top_right", "upper_right", "small_base_right",
        },
        "long_base_right": {
            "long_base_right", "long_right", "bottom_right", "lower_right", "large_base_right",
        },
        "long_base_left": {
            "long_base_left", "long_left", "bottom_left", "lower_left", "large_base_left",
        },
    },
    "square": {
        "front_left": {"front_left", "top_left", "upper_left", "left_top", "left_upper"},
        "front_right": {"front_right", "top_right", "upper_right", "right_top", "right_upper"},
        "back_right": {"back_right", "bottom_right", "lower_right", "right_bottom", "rear_right"},
        "back_left": {"back_left", "bottom_left", "lower_left", "left_bottom", "rear_left"},
    },
}

# ─────────────────────────── 几何辅助函数 ────────────────────────────────

def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _canonical_class_key(name: str) -> str:
    key = _normalize_token(name)
    return _CLASS_ALIASES.get(key, key)


def _load_yaml_file(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_pose_classes_path(weights_path: str, explicit_path: str = "") -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        return path.resolve() if path.exists() else None

    weights = Path(weights_path).expanduser()
    candidates: List[Path] = []
    if weights.exists():
        run_dir = weights.parent.parent if weights.parent.name == "weights" else weights.parent
        args_yaml = run_dir / "args.yaml"
        candidates.append(run_dir / "pose_classes.yaml")
        if args_yaml.exists():
            args_cfg = _load_yaml_file(args_yaml)
            data_cfg_path = args_cfg.get("data")
            if data_cfg_path:
                data_yaml = Path(str(data_cfg_path)).expanduser()
                candidates.append(data_yaml.parent / data_yaml.stem / "pose_classes.yaml")
                candidates.append(data_yaml.with_suffix("") / "pose_classes.yaml")
                if data_yaml.exists():
                    data_cfg = _load_yaml_file(data_yaml)
                    dataset_root = data_cfg.get("path")
                    if dataset_root:
                        root_path = Path(str(dataset_root)).expanduser()
                        candidates.append(root_path / "pose_classes.yaml")
                        candidates.append(root_path.parent / "pose_classes.yaml")

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _default_roles_for_class(cls_key: str, n_kpts: int) -> Dict[str, int]:
    if cls_key == "circle" and n_kpts >= 1:
        return {"center": 0}
    if cls_key == "triangle" and n_kpts >= 3:
        return {"apex": 0, "base_left": 1, "base_right": 2}
    if cls_key == "trapezium" and n_kpts >= 4:
        return {
            "short_base_left": 0,
            "short_base_right": 1,
            "long_base_right": 2,
            "long_base_left": 3,
        }
    if cls_key in _RECT_ALIASES and n_kpts >= 4:
        return {
            "front_left": 0,
            "front_right": 1,
            "back_right": 2,
            "back_left": 3,
        }
    return {}


def _load_pose_class_specs(path: Optional[Path]) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    raw = _load_yaml_file(path)
    classes = raw.get("classes")
    if not isinstance(classes, dict):
        return {}

    specs: Dict[str, dict] = {}
    for class_name, keypoints in classes.items():
        if not isinstance(keypoints, list):
            continue
        cls_key = _canonical_class_key(str(class_name))
        kp_names = [_normalize_token(name) for name in keypoints]
        role_map: Dict[str, int] = {}
        alias_map = _ROLE_ALIASES.get(cls_key, {})
        for idx, kp_name in enumerate(kp_names):
            for role, aliases in alias_map.items():
                if role not in role_map and kp_name in aliases:
                    role_map[role] = idx
        role_map.update({k: v for k, v in _default_roles_for_class(cls_key, len(kp_names)).items() if k not in role_map})
        spec = {
            "class_name": str(class_name),
            "keypoint_names": kp_names,
            "roles": role_map,
        }
        specs[cls_key] = spec
    return specs


def _norm_angle(a: float) -> float:
    """将角度限制到 [-π/2, π/2]（无向轴）。"""
    if a > math.pi / 2.0:
        a -= math.pi
    elif a < -math.pi / 2.0:
        a += math.pi
    return a


def _circular_ema(prev: float, new: float, alpha: float) -> float:
    """在 [-π/2,π/2] 范围内对角度做 EMA，正确处理边界环绕。"""
    diff = new - prev
    if diff > math.pi / 2.0:
        diff -= math.pi
    elif diff < -math.pi / 2.0:
        diff += math.pi
    return _norm_angle(prev + alpha * diff)


def _axis_delta(a: float, b: float) -> float:
    diff = abs(a - b)
    while diff > math.pi / 2.0:
        diff = abs(diff - math.pi)
    return diff


def _axis_from_segment(p0: np.ndarray, p1: np.ndarray) -> Optional[float]:
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    if abs(dx) + abs(dy) < 1e-9:
        return None
    return _norm_angle(math.atan2(dy, dx))


def _undirected_angle_mean(angles: List[float]) -> Optional[float]:
    if not angles:
        return None
    return _norm_angle(math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles)))


def _visible_points(kps_xy: np.ndarray, kps_v: np.ndarray) -> np.ndarray:
    mask = np.asarray(kps_v, dtype=np.float32) > 0.0
    return np.asarray(kps_xy, dtype=np.float32)[mask]


def _required_role_indices(cls_key: str, roles: Dict[str, int], n_kpts: int) -> Optional[List[int]]:
    if cls_key == "circle":
        required = ["center"]
    elif cls_key == "triangle":
        required = ["apex", "base_left", "base_right"]
    elif cls_key == "trapezium":
        required = [
            "short_base_left",
            "short_base_right",
            "long_base_right",
            "long_base_left",
        ]
    elif cls_key in _RECT_ALIASES:
        required = ["front_left", "front_right", "back_right", "back_left"]
    else:
        return None

    out: List[int] = []
    for role in required:
        idx = roles.get(role)
        if not isinstance(idx, int) or not (0 <= idx < n_kpts):
            return None
        out.append(idx)
    return out


def _visible_role_points(
    kps_xy: np.ndarray, kps_v: np.ndarray, indices: List[int]
) -> Optional[np.ndarray]:
    pts: List[np.ndarray] = []
    for idx in indices:
        if float(kps_v[idx]) <= 0.0:
            return None
        pts.append(np.asarray(kps_xy[idx], dtype=np.float64))
    return np.asarray(pts, dtype=np.float64)


def _order_polygon_points(points_xy: np.ndarray) -> np.ndarray:
    center = np.mean(points_xy, axis=0)
    angles = np.arctan2(points_xy[:, 1] - center[1], points_xy[:, 0] - center[0])
    return points_xy[np.argsort(angles)]


def _polygon_centroid(points_xy: np.ndarray) -> Optional[np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] == 0:
        return None
    if pts.shape[0] < 3:
        return np.mean(pts, axis=0)

    poly = _order_polygon_points(pts)
    x = poly[:, 0]
    y = poly[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area2 = float(np.sum(cross))
    if abs(area2) < 1e-9:
        return np.mean(poly, axis=0)
    cx = float(np.sum((x + x_next) * cross) / (3.0 * area2))
    cy = float(np.sum((y + y_next) * cross) / (3.0 * area2))
    return np.array([cx, cy], dtype=np.float64)


def _parallel_edge_axis(points_xy: np.ndarray) -> Optional[float]:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    poly = _order_polygon_points(pts)
    edge_angles: List[Optional[float]] = []
    for i in range(4):
        ang = _axis_from_segment(poly[i], poly[(i + 1) % 4])
        edge_angles.append(ang)

    best_score = -1.0
    best_angle: Optional[float] = None
    for a_idx, b_idx in ((0, 2), (1, 3)):
        a = edge_angles[a_idx]
        b = edge_angles[b_idx]
        if a is None or b is None:
            continue
        score = math.cos(_axis_delta(a, b))
        mean_angle = _undirected_angle_mean([a, b])
        if mean_angle is not None and score > best_score:
            best_score = score
            best_angle = mean_angle
    return best_angle


def _spec_for_class(pose_specs: Dict[str, dict], cls_name: str) -> dict:
    return pose_specs.get(_canonical_class_key(cls_name), {})


def _center_from_kpts(
    kps_xy: np.ndarray, kps_v: np.ndarray, cls_name: str, pose_specs: Dict[str, dict]
) -> Optional[Tuple[float, float]]:
    n = int(kps_xy.shape[0])
    if n == 0:
        return None
    cls_l = _canonical_class_key(cls_name)
    spec = _spec_for_class(pose_specs, cls_name)
    roles = spec.get("roles", {})

    if cls_l in _NO_ORIENT_CLASSES:
        indices = _required_role_indices(cls_l, roles, n)
        if indices is None:
            return None
        center_idx = indices[0]
        if float(kps_v[center_idx]) > 0.0:
            return float(kps_xy[center_idx, 0]), float(kps_xy[center_idx, 1])
        return None

    required = _required_role_indices(cls_l, roles, n)
    if required is not None:
        vis_required = _visible_role_points(kps_xy, kps_v, required)
        if vis_required is None:
            return None
        cxy = _polygon_centroid(vis_required)
        if cxy is None:
            return None
        return float(cxy[0]), float(cxy[1])

    vis = _visible_points(kps_xy, kps_v)
    if vis.shape[0] < 3:
        return None
    cxy = _polygon_centroid(vis)
    if cxy is None:
        return None
    return float(cxy[0]), float(cxy[1])


def _angle_from_kpts(
    kps_xy: np.ndarray, kps_v: np.ndarray, cls_name: str, pose_specs: Dict[str, dict]
) -> Optional[float]:
    cls_l = _canonical_class_key(cls_name)
    if cls_l in _NO_ORIENT_CLASSES:
        return 0.0
    n = int(kps_xy.shape[0])
    spec = _spec_for_class(pose_specs, cls_name)
    roles = spec.get("roles", {})

    if cls_l == "triangle" and n >= 3:
        required = _required_role_indices(cls_l, roles, n)
        if required is None:
            return None
        vis_required = _visible_role_points(kps_xy, kps_v, required)
        if vis_required is None:
            return None
        ang = _axis_from_segment(
            np.asarray(kps_xy[required[1]], dtype=np.float64),
            np.asarray(kps_xy[required[2]], dtype=np.float64),
        )
        if ang is not None:
            return ang
        return None

    if cls_l in _RECT_ALIASES and n >= 4:
        required = _required_role_indices(cls_l, roles, n)
        if required is None:
            return None
        vis_required = _visible_role_points(kps_xy, kps_v, required)
        if vis_required is None:
            return None
        edge_angles = []
        front = _axis_from_segment(
            np.asarray(kps_xy[required[0]], dtype=np.float64),
            np.asarray(kps_xy[required[1]], dtype=np.float64),
        )
        back = _axis_from_segment(
            np.asarray(kps_xy[required[3]], dtype=np.float64),
            np.asarray(kps_xy[required[2]], dtype=np.float64),
        )
        if front is not None:
            edge_angles.append(front)
        if back is not None:
            edge_angles.append(back)
        ang = _undirected_angle_mean(edge_angles)
        if ang is not None:
            return ang
        return None

    if cls_l == "trapezium" and n >= 4:
        required = _required_role_indices(cls_l, roles, n)
        if required is None:
            return None
        vis_required = _visible_role_points(kps_xy, kps_v, required)
        if vis_required is None:
            return None
        edge_angles = []
        top = _axis_from_segment(
            np.asarray(kps_xy[required[0]], dtype=np.float64),
            np.asarray(kps_xy[required[1]], dtype=np.float64),
        )
        bottom = _axis_from_segment(
            np.asarray(kps_xy[required[3]], dtype=np.float64),
            np.asarray(kps_xy[required[2]], dtype=np.float64),
        )
        if top is not None:
            edge_angles.append(top)
        if bottom is not None:
            edge_angles.append(bottom)
        ang = _undirected_angle_mean(edge_angles)
        if ang is not None:
            return ang
        return None

    # ── Fallback：对可见关键点做 PCA
    vis = _visible_points(kps_xy, kps_v)
    if vis.shape[0] < 3:
        return None
    cov = np.cov(vis.T)
    if cov.ndim < 2:
        dx, dy = float(vis[-1, 0] - vis[0, 0]), float(vis[-1, 1] - vis[0, 1])
    else:
        eigvals, eigvecs = np.linalg.eigh(cov)
        ax = eigvecs[:, np.argmax(eigvals)]
        dx, dy = float(ax[0]), float(ax[1])
    if abs(dx) + abs(dy) < 1e-9:
        return None
    return _norm_angle(math.atan2(dy, dx))


# ─────────────────────────── 消息构建 ──────────────────────────────────────

def _build_centers_msg(
    header: Any, frame_w: int, frame_h: int, items: List[DetectionItem]
) -> Centers2D:
    msg = Centers2D()
    msg.header = header
    msg.frame_width = int(max(frame_w, 0))
    msg.frame_height = int(max(frame_h, 0))

    kpts_per_det = 0
    for it in items:
        if it.kpts_xy is not None:
            kpts_per_det = int(np.asarray(it.kpts_xy).shape[0])
            break
    msg.kpts_per_det = kpts_per_det

    for it in items:
        msg.obj_class.append(str(it.label))
        msg.score.append(float(it.score))
        msg.u.append(int(it.u))
        msg.v.append(int(it.v))
        msg.angle_rad.append(float(it.angle_rad))
        msg.angle_deg.append(float(math.degrees(it.angle_rad)))
        msg.cx_norm.append(float(it.cx_norm))
        msg.cy_norm.append(float(it.cy_norm))
        if kpts_per_det > 0:
            xy = np.asarray(it.kpts_xy) if it.kpts_xy is not None else np.zeros((kpts_per_det, 2))
            vv = np.asarray(it.kpts_v) if it.kpts_v is not None else np.zeros(kpts_per_det)
            msg.kpt_u.extend(float(v) for v in xy[:, 0])
            msg.kpt_v.extend(float(v) for v in xy[:, 1])
            msg.kpt_conf.extend(float(v) for v in vv)
    return msg


# ─────────────────────────── ROS 节点 ────────────────────────────────────────

class DetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_center_detection_node")

        # 默认使用压缩话题（节省 Wi-Fi 带宽）。
        # 传入 use_compressed:=false 可回退到原始 sensor_msgs/Image。
        rgb_topic = declare_and_get_str(
            self, "rgb_topic", "/ascamera_hp60c/camera_publisher/rgb0/image/compressed"
        )
        self._use_compressed = declare_and_get_bool(self, "use_compressed", True)
        centers_topic = declare_and_get_str(self, "centers_topic", "/object_centers_2d")
        weights = declare_and_get_str(self, "weights_path", "")
        pose_classes_path = declare_and_get_str(self, "pose_classes_path", "")
        self._conf = declare_and_get_float(self, "conf_threshold", 0.25)
        self._device = declare_and_get_str(self, "device", "cuda:0")
        self._imgsz = declare_and_get_int(self, "imgsz", 640)
        self._ema_a = declare_and_get_float(self, "ema_alpha", 0.4)
        # FP16：GPU 速度提升约 1.5 倍；在 CPU 上自动禁用
        self._use_half = declare_and_get_bool(self, "use_half", True)
        # 稳定门控：连续 N 帧检测到目标后才发布。
        # 在真实目标不增加延迟的前提下消除单帧假阳性。
        self._min_stable = declare_and_get_int(self, "min_stable_frames", 2)
        # 限制 NMS 前最大检测数，防止模型生成大量低质量候选时
        # 出现罕见的 NMS 超时（>2 s）。
        self._max_det = declare_and_get_int(self, "max_det", 50)
        # 逐帧 INFO 日志开销较大，调试时再开启。
        self._log_detections = declare_and_get_bool(self, "log_detections", False)
        # 无目标通过稳定门控时，最多每隔此时间（秒）记录一次 "Detection: None"。
        # 不要设为 0——以处理帧率（约 5~30 Hz）每帧打印日志会损伤实时性能。
        self._log_empty_sec = declare_and_get_float(self, "log_empty_throttle_sec", 2.0)
        if self._log_empty_sec <= 0.0:
            self._log_empty_sec = 2.0

        # 非 CUDA 设备自动禁用 FP16
        if self._use_half and ("cuda" not in self._device.lower()):
            self._use_half = False
            self.get_logger().info("use_half disabled: device is not CUDA")

        # ── 各类别状态（目标消失时重置）────────────────────────────────────
        self._angle_ema: Dict[str, float] = {}    # 类别 -> 平滑后的角度（弧度）
        self._stability: Dict[str, int]   = {}    # 类别 -> 连续帧计数

        self._pose_classes_path = _resolve_pose_classes_path(weights, pose_classes_path)
        self._pose_specs = _load_pose_class_specs(self._pose_classes_path)

        self._model: Optional[Any] = None
        if YOLO is None:
            self.get_logger().error("ultralytics not installed.")
        elif not weights:
            self.get_logger().warn("weights_path is empty: publishing empty centers.")
        else:
            self._model = YOLO(weights)
            self.get_logger().info(f"Loaded YOLO weights: {weights}")

        self.get_logger().info(
            f"conf={self._conf}  device={self._device}  "
            f"imgsz={self._imgsz}  half={self._use_half}  "
            f"ema_alpha={self._ema_a}  min_stable={self._min_stable}"
        )
        if self._pose_classes_path is not None:
            self.get_logger().info(f"pose_classes: {self._pose_classes_path}")
        else:
            self.get_logger().warn("pose_classes.yaml 未找到，将回退到几何兜底逻辑。")
        self._predict_kwargs = self._build_predict_kwargs()

        self._pub = self.create_publisher(Centers2D, centers_topic, _QOS_RESULT)
        # 使用传感器数据 QoS（BEST_EFFORT / VOLATILE）以匹配相机驱动
        # （驱动以 BEST_EFFORT 发布，与默认的 RELIABLE 不兼容）。
        if self._use_compressed:
            self.create_subscription(
                CompressedImage, rgb_topic, self._on_image_compressed,
                qos_profile=qos_profile_sensor_data,
            )
            self.get_logger().info(f"Subscribing RGB (compressed): {rgb_topic}")
        else:
            self.create_subscription(
                Image, rgb_topic, self._on_image,
                qos_profile=qos_profile_sensor_data,
            )
            self.get_logger().info(f"Subscribing RGB (raw): {rgb_topic}")

        self._img_queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="yolo_infer")
        self._worker.start()

        self.get_logger().info(f"Publishing centers: {centers_topic}")

    # ── 推理辅助函数 ──────────────────────────────────────────────────────────

    def _build_predict_kwargs(self) -> dict:
        kw: dict = {
            "conf":    self._conf,
            "imgsz":   self._imgsz,
            "max_det": self._max_det,
            "verbose": False,
            "stream":  False,  # 单帧预测，无生成器开销
        }
        if self._device:
            kw["device"] = self._device
        if self._use_half:
            kw["half"] = True
        return kw

    def _smooth_angle(self, cls: str, angle_rad: float) -> float:
        if self._ema_a >= 1.0:
            self._angle_ema[cls] = angle_rad
            return angle_rad
        if cls not in self._angle_ema:
            self._angle_ema[cls] = angle_rad
            return angle_rad
        s = _circular_ema(self._angle_ema[cls], angle_rad, self._ema_a)
        self._angle_ema[cls] = s
        return s

    def _update_stability(self, items: List[DetectionItem]) -> List[DetectionItem]:
        """递增已检测类别的计数器；对消失的类别重置。
        仅返回已稳定 ≥ min_stable_frames 帧的目标。"""
        detected = {it.label for it in items}

        # 重置消失的类别并清除其过期 EMA
        for cls in list(self._stability):
            if cls not in detected:
                self._stability[cls] = 0
                self._angle_ema.pop(cls, None)

        # 递增当前帧检测到的类别
        for it in items:
            self._stability[it.label] = self._stability.get(it.label, 0) + 1

        if self._min_stable <= 1:
            return items
        return [it for it in items if self._stability.get(it.label, 0) >= self._min_stable]

    def _infer_pose(self, frame_bgr) -> List[DetectionItem]:
        if self._model is None:
            return []
        results = self._model.predict(source=frame_bgr, **self._predict_kwargs)
        out: List[DetectionItem] = []
        h0, w0 = frame_bgr.shape[:2]
        max_u, max_v = max(w0 - 1, 0), max(h0 - 1, 0)
        fw, fh = float(max(w0, 1)), float(max(h0, 1))
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            kps = getattr(r, "keypoints", None)
            if kps is None:
                continue
            names = getattr(r, "names", {}) or {}
            boxes  = r.boxes
            confs  = boxes.conf.cpu().numpy()
            clss   = boxes.cls.cpu().numpy().astype(int)
            xywh   = boxes.xywh.cpu().numpy()
            kps_xy = kps.xy.cpu().numpy()
            kps_v  = (kps.conf.cpu().numpy() if kps.conf is not None
                      else np.ones(kps_xy.shape[:2], dtype=np.float32))
            for i in range(len(confs)):
                label = names.get(int(clss[i]), str(int(clss[i])))
                xy, v = kps_xy[i], kps_v[i]
                center = _center_from_kpts(xy, v, label, self._pose_specs)
                angle_rad = _angle_from_kpts(xy, v, label, self._pose_specs)
                if center is None or angle_rad is None:
                    continue
                cx_f, cy_f = center
                # EMA 作用于原始 [-90°,90°]。对矩形，折叠到 [-45°,45°] 在 EMA 之后执行，
                # 这样平滑信号在越过 ±45° 边界时会平缓过渡，无需额外状态即实现隐式迟滞。
                angle_rad = self._smooth_angle(label, angle_rad)
                if _canonical_class_key(label) in _RECT_ALIASES:
                    if angle_rad > math.pi / 4.0:
                        angle_rad -= math.pi / 2.0
                    elif angle_rad < -math.pi / 4.0:
                        angle_rad += math.pi / 2.0
                u   = min(max(int(round(cx_f)), 0), max_u)
                v_p = min(max(int(round(cy_f)), 0), max_v)
                out.append(DetectionItem(label, float(confs[i]), u, v_p, angle_rad,
                                         float(u)/fw, float(v_p)/fh, kpts_xy=xy, kpts_v=v))
        return out

    def _infer(self, frame_bgr) -> List[DetectionItem]:
        return self._infer_pose(frame_bgr)

    # ── ROS 回调 ──────────────────────────────────────────────────────────────

    def _enqueue(self, msg) -> None:
        """将最新帧（Image 或 CompressedImage）入队，丢弃过期帧。"""
        replace_latest(self._img_queue, msg)

    def _on_image(self, msg: Image) -> None:
        self._enqueue(msg)

    def _on_image_compressed(self, msg: CompressedImage) -> None:
        self._enqueue(msg)

    def _worker_loop(self) -> None:
        # GPU 预热：在真实帧到来前执行一次哑推理，触发 CUDA JIT 编译。
        # 否则第一帧真实推理可能耗时 2~5 秒，导致消息积压。
        if self._model is not None and "cuda" in self._device.lower():
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self._model.predict(source=dummy, **self._predict_kwargs)
                self.get_logger().info("YOLO GPU warmup done.")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"GPU warmup failed (ignored): {e}")

        while True:
            try:
                msg = self._img_queue.get(timeout=1.0)
            except queue.Empty:
                if not rclpy.ok():
                    break
                continue

            try:
                if isinstance(msg, CompressedImage):
                    frame = compressed_imgmsg_to_bgr8(msg)
                else:
                    frame = imgmsg_to_bgr8(msg)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"RGB decode failed: {e}")
                continue

            h, w = frame.shape[:2]
            try:
                items = self._infer(frame)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"Inference error: {e}", throttle_duration_sec=2.0)
                continue

            # 稳定门控：丢弃不稳定检测，重置过期状态
            items = self._update_stability(items)

            if self._log_detections:
                if items:
                    self.get_logger().info(
                        "Detected: " + ", ".join(
                            f"{it.label}({it.score:.2f}) {math.degrees(it.angle_rad):+.1f}deg"
                            for it in items),
                        throttle_duration_sec=1.0,
                    )
                else:
                    # 无目标：节流间隔比有目标时更长，避免场景为空时日志刷屏。
                    self.get_logger().info(
                        "Detection: None",
                        throttle_duration_sec=self._log_empty_sec,
                    )

            self._pub.publish(_build_centers_msg(msg.header, w, h, items))


def main() -> None:
    rclpy.init()
    node = DetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
