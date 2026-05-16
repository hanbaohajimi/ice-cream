#!/usr/bin/env python3
"""Realtime console logger: Centers3D + T_cam→ee + Pi WebSocket → base xyz.

变换链::
  - Pi ``feedback.link5_hmat`` = :math:`T_{ee→base}`（4×4）
  - T_cam2ee 从仓库根 config.yaml hand_eye.T_cam2ee 读取
  - 相机系点 :math:`p_{cam}`（Centers3D 的 x,y,z）
  - :math:`p_{base} = T_{ee2base} T_{cam2ee} p_{cam}`
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import rclpy
import websockets
from center_depth_msgs.msg import Centers3D
from center_depth_pipeline.param_utils import read_repo_config
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


def _load_T_cam2ee(cfg: dict) -> np.ndarray:
    """从 config hand_eye.T_cam2ee 读取 4×4 矩阵；格式不对则抛出 ValueError。"""
    rows = cfg.get("hand_eye", {}).get("T_cam2ee")
    if rows and len(rows) == 4 and all(len(r) == 4 for r in rows):
        return np.array(rows, dtype=np.float64)
    raise ValueError("config.yaml hand_eye.T_cam2ee 缺失或格式错误（需要 4×4 矩阵）")

_QOS_LATEST = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class ObjectBaseLoggerNode(Node):
    """实时控制台日志：Centers3D + T_cam→ee + Pi WebSocket → 基座 xyz 坐标。"""

    def __init__(self) -> None:
        super().__init__("object_base_logger_node")

        _cfg = read_repo_config()
        _net = _cfg.get("network", {})
        _log = _cfg.get("logger",  {})

        centers_3d_topic = self.declare_parameter(
            "centers_3d_topic", "/object_centers_3d"
        ).get_parameter_value().string_value
        target_class = self.declare_parameter(
            "target_class", _log.get("target_class", "")
        ).get_parameter_value().string_value.strip()
        log_empty_throttle_sec = self.declare_parameter(
            "log_empty_throttle_sec", 2.0
        ).get_parameter_value().double_value
        ws_host = self.declare_parameter(
            "ws_host", _net.get("ws_host", "127.0.0.1")
        ).get_parameter_value().string_value
        ws_port = self.declare_parameter(
            "ws_port", int(_net.get("ws_port", 8765))
        ).get_parameter_value().integer_value
        log_detection_min_interval_sec = self.declare_parameter(
            "log_detection_min_interval_sec", _log.get("log_detection_min_interval_sec", 1.0)
        ).get_parameter_value().double_value
        log_best_detection_only = self.declare_parameter(
            "log_best_detection_only", bool(_log.get("log_best_detection_only", False))
        ).get_parameter_value().bool_value
        head_ingestion_enabled = self.declare_parameter(
            "head_ingestion_enabled", bool(_log.get("head_ingestion_enabled", False))
        ).get_parameter_value().bool_value
        head_http_url = self.declare_parameter(
            "head_http_url", _net.get("head_http_url", "")
        ).get_parameter_value().string_value.strip()
        head_http_timeout_sec = self.declare_parameter(
            "head_http_timeout_sec", _net.get("head_http_timeout_sec", 0.25)
        ).get_parameter_value().double_value
        head_role = self.declare_parameter(
            "head_role", _log.get("head_role", "object")
        ).get_parameter_value().string_value.strip()
        head_label_prefix = self.declare_parameter(
            "head_label_prefix", _log.get("head_label_prefix", "")
        ).get_parameter_value().string_value.strip()
        head_position_z_offset_m = self.declare_parameter(
            "head_position_z_offset_m",
            float(_log.get("head_position_z_offset_m", 0.0)),
        ).get_parameter_value().double_value

        self._target_class = target_class
        self._log_empty_throttle_sec = log_empty_throttle_sec
        self._last_empty_log_time = 0.0
        self._log_det_min_interval = max(0.0, log_detection_min_interval_sec)
        self._log_best_only = log_best_detection_only
        self._last_det_log_time = 0.0
        self._head_ingestion_enabled = head_ingestion_enabled
        self._head_http_url = self._normalize_head_http_url(head_http_url)
        self._head_http_timeout = max(0.05, head_http_timeout_sec)
        self._head_role = head_role if head_role in {"object", "target", "lid"} else "object"
        self._head_label_prefix = head_label_prefix
        self._head_z_offset_m = head_position_z_offset_m
        self._frame_seq = 0

        try:
            self._T_cam2ee = _load_T_cam2ee(_cfg)
        except ValueError as exc:
            self.get_logger().fatal(str(exc))
            raise
        self.get_logger().info(
            f"T_cam→ee: config.yaml hand_eye.T_cam2ee  "
            f"|t|={float(np.linalg.norm(self._T_cam2ee[:3, 3]) * 1000):.2f} mm"
        )

        self._ws_uri = f"ws://{ws_host}:{int(ws_port)}"
        self._t_ee2base: np.ndarray | None = None
        self._t_lock = threading.Lock()
        self._ws_stop = threading.Event()
        self._ws_thread = threading.Thread(
            target=self._run_ws_thread, daemon=True, name="base_logger_ws",
        )
        self._ws_thread.start()

        self.create_subscription(Centers3D, centers_3d_topic, self._on_centers, _QOS_LATEST)
        self.get_logger().info(f"det_logger 订阅: {centers_3d_topic}")
        self.get_logger().info(f"det_logger 目标类别过滤: {target_class or '<全部>'}")
        self.get_logger().info(f"det_logger WebSocket 末端位姿: {self._ws_uri} (link5_hmat=T_ee2base)")
        self.get_logger().info(
            f"det_logger 检测日志: min_interval={self._log_det_min_interval}s "
            f"best_only={self._log_best_only}"
        )
        if self._head_ingestion_enabled and self._head_http_url:
            self.get_logger().info(
                f"det_logger head ingestion HTTP: {self._head_http_url} role={self._head_role} "
                f"head_position_z_offset_m={self._head_z_offset_m:+.4f}"
            )
        else:
            self.get_logger().info("det_logger head ingestion: disabled")

    def destroy_node(self) -> bool:
        self._ws_stop.set()
        if self._ws_thread.is_alive():
            self._ws_thread.join(timeout=1.5)
        return super().destroy_node()

    def _run_ws_thread(self) -> None:
        asyncio.run(self._ws_async_loop())

    async def _ws_async_loop(self) -> None:
        while not self._ws_stop.is_set():
            try:
                async with websockets.connect(
                    self._ws_uri, ping_interval=20, ping_timeout=20
                ) as ws:
                    while not self._ws_stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        h = None
                        if msg.get("type") == "state":
                            data = msg.get("data") or {}
                            fb = data.get("feedback") or {}
                            h = data.get("link5_hmat") or fb.get("link5_hmat")
                        if h is None:
                            h = msg.get("link5_hmat")
                        if h is None:
                            continue
                        arr = np.array(h, dtype=np.float64)
                        if arr.shape != (4, 4) or not np.isfinite(arr).all():
                            continue
                        with self._t_lock:
                            self._t_ee2base = arr
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"WebSocket 断开/异常: {exc!s}，0.8s 后重连 {self._ws_uri}",
                    throttle_duration_sec=3.0,
                )
                await asyncio.sleep(0.8)

    def _point_cam_to_base(self, x: float, y: float, z: float) -> tuple[float, float, float] | None:
        """p_base = T_ee2base @ T_cam2ee @ p_cam。"""
        with self._t_lock:
            t_e2b = None if self._t_ee2base is None else self._t_ee2base.copy()
        if t_e2b is None:
            return None
        p_cam = np.array([x, y, z, 1.0], dtype=np.float64)
        p_b = t_e2b @ self._T_cam2ee @ p_cam
        return float(p_b[0]), float(p_b[1]), float(p_b[2])

    @staticmethod
    def _normalize_head_http_url(value: str) -> str:
        if not value:
            return ""
        url = value if "://" in value else f"http://{value}"
        return url.rstrip("/") + "/api/detection"

    def _post_head_detection(self, objects: list[dict[str, object]]) -> None:
        if not self._head_ingestion_enabled or not self._head_http_url:
            return
        self._frame_seq += 1
        payload = {
            "cmd": "detection",
            "frame": "robot_base",
            "frame_id": self._frame_seq,
            "ts": time.time(),
            "objects": objects,
        }
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = urllib_request.Request(
            self._head_http_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self._head_http_timeout) as resp:
                resp.read(256)
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            self.get_logger().warn(
                f"head ingestion HTTP 发送失败: {exc!s}",
                throttle_duration_sec=3.0,
            )

    def _build_head_object(
        self,
        idx: int,
        cls: str,
        score: float,
        base_xyz: tuple[float, float, float],
        angle_deg: float,
    ) -> dict[str, object]:
        label = f"{self._head_label_prefix}{cls}" if cls else f"{self._head_label_prefix}object"
        bx, by, bz = base_xyz
        return {
            "role": self._head_role,
            "track_id": f"{self._head_role}:{cls or 'object'}:{idx}",
            "class_id": cls or idx,
            "label": label,
            "confidence": score,
            "position": {"x": bx, "y": by, "z": bz},
            "wrist_yaw_deg": angle_deg,
        }

    def _on_centers(self, msg: Centers3D) -> None:
        count = min(len(msg.x), len(msg.y), len(msg.z), len(msg.u), len(msg.v))
        if count == 0:
            now = time.monotonic()
            if now - self._last_empty_log_time >= self._log_empty_throttle_sec:
                self._last_empty_log_time = now
                self.get_logger().info("[det-log] Detection=None")
            return

        now = time.monotonic()
        candidates: list[tuple[int, float]] = []
        for idx in range(count):
            cls = msg.obj_class[idx] if idx < len(msg.obj_class) else ""
            if self._target_class and cls != self._target_class:
                continue
            score = float(msg.score[idx]) if idx < len(msg.score) else 0.0
            candidates.append((idx, score))

        matched = len(candidates)
        if matched == 0:
            if now - self._last_empty_log_time >= self._log_empty_throttle_sec:
                self._last_empty_log_time = now
                self.get_logger().info(
                    f"[det-log] Detection filtered out by target_class={self._target_class!r}"
                )
            return

        if (
            self._log_det_min_interval > 0.0
            and (now - self._last_det_log_time) < self._log_det_min_interval
        ):
            return

        if self._log_best_only:
            candidates.sort(key=lambda t: (-t[1], t[0]))
            to_log = [candidates[0]]
        else:
            to_log = candidates

        head_objects: list[dict[str, object]] = []
        for idx, score in to_log:
            cls = msg.obj_class[idx] if idx < len(msg.obj_class) else ""
            u = int(msg.u[idx])
            v = int(msg.v[idx])
            x = float(msg.x[idx])
            y = float(msg.y[idx])
            z = float(msg.z[idx])
            angle_deg = float(msg.angle_deg[idx]) if idx < len(msg.angle_deg) else 0.0

            base_txt = ""
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                base_xyz = self._point_cam_to_base(x, y, z)
                if base_xyz is not None:
                    bx, by, bz = base_xyz
                    head_objects.append(
                        self._build_head_object(
                            idx,
                            cls,
                            score,
                            (bx, by, bz + self._head_z_offset_m),
                            angle_deg,
                        )
                    )
                    base_txt = (
                        f" base_xyz_m=({bx:+.4f},{by:+.4f},{bz:+.4f})"
                    )
                else:
                    base_txt = " base_xyz_m=pending_ws"
            else:
                base_txt = " base_xyz_m=nan"

            self.get_logger().info(
                f"[det-log] idx={idx} class={cls or '-'} score={score:.2f} "
                f"center_px=({u},{v}) "
                f"z={z:+.4f}m "
                f"cam_xyz_m=({x:+.4f},{y:+.4f},{z:+.4f}) "
                f"{base_txt} "
                f"angle={angle_deg:+.1f}deg"
            )

        self._post_head_detection(head_objects)
        self._last_det_log_time = now


def main() -> None:
    rclpy.init()
    node = ObjectBaseLoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
