#!/usr/bin/env python3
"""
实时叠加可视化。

架构
────────────
旧设计使用 ApproximateTimeSynchronizer(RGB, Centers3D)。
问题：Centers3D 的发布频率 ≤ YOLO 推理频率（通常 5~15 fps），同步器
仅以较慢频率触发，即使相机以 30 fps 发布，显示窗口也会卡顿。

新设计——完全解耦的订阅：
  RGB 订阅（30 fps）  → _rgb_queue(maxsize=1) → _worker_loop
  Centers3D 订阅（≤30 fps）→ _latest_det（单次原子属性写入）

_worker_loop 由每个新 RGB 帧触发，解码帧并将 _latest_det 中的
当前检测状态叠加（可能比推理周期滞后 1~2 帧，但对慢速移动目标
在视觉上可忽略）。视频画面始终以相机帧率平滑刷新。

线程映射
  ROS spin 线程 : _on_rgb / _on_det — O(1)，禁止阻塞
  _worker 线程  : JPEG 解码 + 绘制 + 入队显示帧
  _display 线程 : cv2.imshow + waitKey（GUI 事件循环）
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from typing import List, Optional, Tuple

from center_depth_pipeline.qt_env import configure_qt_font_env

configure_qt_font_env()
import cv2
import numpy as np
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

from center_depth_msgs.msg import Centers3D
from center_depth_pipeline.image_numpy import (
    bgr8_to_imgmsg,
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

# ── 各类别颜色（BGR）────────────────────────────────────────────────────────
_CLASS_COLOR: dict = {
    "triangle":  (0,   165, 255),
    "circle":    (255,   0, 255),
    "trapezium": (0,   255, 255),
    "square":    (255, 220,   0),
}
_DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 255)

# ── 关键点骨架定义 ─────────────────────────────────────────────────────────
_SKELETON: dict = {
    "triangle":  [(0, 1), (1, 2), (2, 0)],
    "circle":    [],
    "trapezium": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "square":    [(0, 1), (1, 2), (2, 3), (3, 0)],
}
_KPT_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
]
_SKELETON_COLOR = (180, 180, 180)

# Window name
_WIN = "center_depth_overlay (q=quit)"

_QOS_VIZ_IMAGE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class OverlayVizNode(Node):
    def __init__(self) -> None:
        super().__init__("center_depth_overlay_node")

        rgb_topic = declare_and_get_str(
            self, "rgb_topic", "/ascamera_hp60c/camera_publisher/rgb0/image/compressed"
        )
        self._use_comp = declare_and_get_bool(self, "use_compressed", True)
        centers_3d_topic = declare_and_get_str(self, "centers_3d_topic", "/object_centers_3d")
        viz_topic = declare_and_get_str(self, "viz_image_topic", "/object_overlay/image")
        self._show_window = declare_and_get_bool(self, "show_window", True)
        self._publish_image = declare_and_get_bool(self, "publish_image", True)
        self._font_scale = declare_and_get_float(self, "font_scale", 0.52)
        self._thickness = declare_and_get_int(self, "line_thickness", 2)
        self._show_kpts = declare_and_get_bool(self, "show_keypoints", True)
        self._kpt_thr = declare_and_get_float(self, "keypoint_conf_threshold", 0.3)
        # 超过此时间（秒）无新 Centers3D 则隐藏过期检测叠加。
        # 当机械臂转向或流水线卡顿时，防止旧检测结果持续显示。
        self._det_stale_sec = declare_and_get_float(self, "det_stale_timeout_sec", 1.0)
        # imshow / 叠加发布前缩小分辨率（0.5 = 半分辨率，像素数减少约 4 倍）。
        self._disp_scale = declare_and_get_float(self, "display_scale", 1.0)
        # 每隔 N 帧才解码/绘制一次（2 = 可视化 CPU 占用减半，检测不受影响）。
        self._rgb_stride = max(1, declare_and_get_int(self, "rgb_process_stride", 1))
        # OpenCV 内部线程数（默认使用所有核心 → 与 PyTorch 争抢资源），1~2 通常最优。
        opencv_nt = max(1, declare_and_get_int(self, "opencv_num_threads", 2))
        cv2.setUseOptimized(True)
        cv2.setNumThreads(opencv_nt)

        # ── 发布者 ────────────────────────────────────────────────────────────
        self._pub: Optional[rclpy.publisher.Publisher] = None
        if self._publish_image:
            self._pub = self.create_publisher(Image, viz_topic, _QOS_VIZ_IMAGE)

        # ── 解耦订阅 ──────────────────────────────────────────────────────────
        # RGB：以相机帧率（30 fps）触发工作线程
        qos = qos_profile_sensor_data
        if self._use_comp:
            self.create_subscription(CompressedImage, rgb_topic, self._on_rgb, qos)
            self.get_logger().info(f"RGB (compressed): {rgb_topic}")
        else:
            self.create_subscription(Image, rgb_topic, self._on_rgb, qos)
            self.get_logger().info(f"RGB (raw): {rgb_topic}")

        # Centers3D：仅存储最新检测状态（不做同步）
        self.create_subscription(Centers3D, centers_3d_topic, self._on_det, qos)
        self.get_logger().info(f"Centers3D in: {centers_3d_topic}")

        # ── 共享检测状态（GIL 原子属性读写）──────────────────────────────────
        # _latest_det_time 记录最后一次调用 _on_det 的挂钟时间。
        # 若距上次超过 det_stale_timeout_sec，则无论 _latest_det 内容如何
        # 都隐藏叠加，防止机械臂转向或深度/同步流水线卡顿时显示过期检测。
        self._latest_det:      Optional[Centers3D] = None
        self._latest_det_time: float               = 0.0   # time.time() at last _on_det

        # ── 工作线程（JPEG 解码 + 绘制）──────────────────────────────────────
        # maxsize=1：工作线程忙时丢弃过期 RGB 帧（最新帧策略）
        self._rgb_queue:  queue.Queue = queue.Queue(maxsize=1)
        self._disp_queue: Optional[queue.Queue] = (
            queue.Queue(maxsize=1) if self._show_window else None
        )
        self._quit = threading.Event()

        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="viz_draw"
        )
        self._worker.start()

        # ── 显示线程 ───────────────────────────────────────────────────────────
        if self._show_window:
            self._disp_thread = threading.Thread(
                target=self._display_loop, daemon=True, name="cv2_display"
            )
            self._disp_thread.start()

        # FPS 统计（以工作线程收到的帧数为准）
        self._fps_t0    = time.time()
        self._fps_count = 0
        self._fps       = 0.0
        self._rgb_stride_i = 0  # 用于 rgb_process_stride

        if self._publish_image:
            self.get_logger().info(f"Overlay out: {viz_topic}")

    # ── ROS spin 线程回调（O(1)）──────────────────────────────────────────────

    def _on_rgb(self, msg) -> None:
        """以相机帧率触发。丢弃旧帧，入队最新帧。"""
        replace_latest(self._rgb_queue, msg)

    def _on_det(self, msg: Centers3D) -> None:
        """存储最新检测状态。GIL 保证属性写入是原子操作。"""
        self._latest_det      = msg
        self._latest_det_time = time.time()

    # ── 工作线程 ──────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while not self._quit.is_set():
            try:
                rgb_msg = self._rgb_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._rgb_stride_i += 1
            if (self._rgb_stride_i % self._rgb_stride) != 0:
                continue

            # Decode RGB frame
            try:
                if isinstance(rgb_msg, CompressedImage):
                    frame = compressed_imgmsg_to_bgr8(rgb_msg)
                else:
                    frame = imgmsg_to_bgr8(rgb_msg)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"RGB decode failed: {e}", throttle_duration_sec=2.0)
                continue

            # 获取最新检测状态快照。
            # 若近期无检测结果则视为"无检测"——处理机械臂转向导致
            # 流水线卡顿或深度同步停止触发时叠加框持续残留的情况。
            det       = self._latest_det
            det_age   = time.time() - self._latest_det_time
            if det_age > self._det_stale_sec:
                det = None   # 抑制过期叠加

            try:
                self._draw(frame, det, rgb_msg.header)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"Draw error: {e}", throttle_duration_sec=2.0)

    def _draw(self, frame: np.ndarray, det: Optional[Centers3D], header) -> None:
        h, w   = frame.shape[:2]
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fs     = float(self._font_scale)
        th     = int(max(self._thickness, 1))

        # ── FPS（以 RGB 帧为准，即相机帧率）────────────────────────────────
        self._fps_count += 1
        now = time.time()
        if now - self._fps_t0 >= 1.0:
            self._fps       = self._fps_count / (now - self._fps_t0)
            self._fps_count = 0
            self._fps_t0    = now
        cv2.putText(frame, f"FPS {self._fps:.1f}", (8, 22),
                    font, 0.65, (0, 255, 128), 2, cv2.LINE_AA)

        # ── 尚无检测结果 ──────────────────────────────────────────────────────
        if det is None or len(det.u) == 0:
            self._finalize_output(frame, header)
            return

        # ── 收集每个检测的信息 ─────────────────────────────────────────────────
        n      = min(len(det.u), len(det.v), len(det.depth_m))
        n_adeg = len(det.angle_deg)
        det_info = []

        for i in range(n):
            u = int(det.u[i])
            v = int(det.v[i])
            if u < 0 or v < 0 or u >= w or v >= h:
                continue
            cls       = str(det.obj_class[i]) if i < len(det.obj_class) else "obj"
            score     = float(det.score[i])   if i < len(det.score) else 0.0
            angle_deg = float(det.angle_deg[i]) if i < n_adeg else 0.0
            depth_raw = float(det.depth_m[i])
            depth_str = f"{depth_raw:.3f}m" if math.isfinite(depth_raw) else "N/A"
            # 用 "deg" 而非 "°"：OpenCV HERSHEY 字体仅支持 ASCII
            text  = f"{cls}({score:.2f}) {angle_deg:+.1f}deg ({u},{v}) {depth_str}"
            color = _CLASS_COLOR.get(cls.lower(), _DEFAULT_COLOR)
            det_info.append((i, u, v, cls, angle_deg, text, color))

        # ── 一次 addWeighted 处理所有文字背景（O(1) 混合操作）──────────────
        if det_info:
            overlay = frame.copy()
            for _, u, v, _, _, text, _ in det_info:
                (tw2, th2), bl = cv2.getTextSize(text, font, fs, th)
                tx = min(max(u + 10, 0), max(w - tw2 - 4, 0))
                ty = min(max(v - 12, th2 + 2), h - 2)
                cv2.rectangle(overlay,
                              (tx - 2, ty - th2 - 2),
                              (tx + tw2 + 2, ty + bl + 2),
                              (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # ── 绘制关键点、箭头和文字 ────────────────────────────────────────────
        for i, u, v, cls, angle_deg, text, color in det_info:
            if self._show_kpts:
                self._draw_keypoints(frame, det, i, cls)

            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1, cv2.LINE_AA)

            ar   = math.radians(angle_deg)
            alen = 45
            ax = min(max(int(u + alen * math.cos(ar)), 0), w - 1)
            ay = min(max(int(v + alen * math.sin(ar)), 0), h - 1)
            cv2.arrowedLine(frame, (u, v), (ax, ay), (0, 220, 0), 2,
                            cv2.LINE_AA, tipLength=0.30)

            (tw2, th2), _ = cv2.getTextSize(text, font, fs, th)
            tx = min(max(u + 10, 0), max(w - tw2 - 4, 0))
            ty = min(max(v - 12, th2 + 2), h - 2)
            cv2.putText(frame, text, (tx, ty), font, fs, color, th, cv2.LINE_AA)

        # ── 转发到显示 / 发布 ────────────────────────────────────────────────
        self._finalize_output(frame, header)

    def _finalize_output(self, frame: np.ndarray, header) -> None:
        """可选缩小分辨率用于 GUI / DDS，然后入队并发布。"""
        out = frame
        s = float(self._disp_scale)
        if s > 0.0 and s < 0.999:
            nh = max(1, int(round(frame.shape[0] * s)))
            nw = max(1, int(round(frame.shape[1] * s)))
            out = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        self._enqueue_display(out)
        if self._publish_image and self._pub is not None:
            if self._pub.get_subscription_count() > 0:
                self._pub.publish(bgr8_to_imgmsg(out, header))

    # ── 显示线程 ──────────────────────────────────────────────────────────────

    def _enqueue_display(self, frame: np.ndarray) -> None:
        if self._disp_queue is None:
            return
        replace_latest(self._disp_queue, frame)

    def _display_loop(self) -> None:
        """专用 GUI 线程。轮询 disp_queue 并在每次迭代调用 waitKey，
        保持 Qt/X11 事件循环响应，不受帧到达频率影响。"""
        while not self._quit.is_set():
            if self._disp_queue is None:
                break
            try:
                frame = self._disp_queue.get(timeout=0.016)  # 最多等待 16 ms
                cv2.imshow(_WIN, frame)
            except queue.Empty:
                pass
            # 始终调用 waitKey 以处理 Qt/GTK 事件，避免按键事件积压
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info("Quit key pressed.")
                self._quit.set()
                cv2.destroyAllWindows()
                break

    # ── 关键点绘制 ────────────────────────────────────────────────────────────

    def _draw_keypoints(
        self, frame: np.ndarray, det: Centers3D, det_idx: int, cls_name: str
    ) -> None:
        K = int(det.kpts_per_det)
        if K == 0:
            return
        base = det_idx * K
        if base + K > len(det.kpt_u):
            return
        h, w = frame.shape[:2]
        skeleton = _SKELETON.get(cls_name.lower(), [])

        pts: List[Optional[Tuple[int, int]]] = []
        for k in range(K):
            conf = float(det.kpt_conf[base + k])
            if conf < self._kpt_thr:
                pts.append(None)
                continue
            px = int(round(float(det.kpt_u[base + k])))
            py = int(round(float(det.kpt_v[base + k])))
            pts.append((px, py) if 0 <= px < w and 0 <= py < h else None)

        for a, b in skeleton:
            if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                cv2.line(frame, pts[a], pts[b], _SKELETON_COLOR, 1, cv2.LINE_AA)

        for k, pt in enumerate(pts):
            if pt is None:
                continue
            color = _KPT_COLORS[k % len(_KPT_COLORS)]
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
            cv2.putText(frame, str(k), (pt[0] + 6, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def main() -> None:
    rclpy.init()
    node = OverlayVizNode()
    try:
        rclpy.spin(node)
    finally:
        node._quit.set()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
