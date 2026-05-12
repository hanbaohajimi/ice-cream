#!/usr/bin/env python3
"""
在每个检测中心附近采样中位深度值，反投影到三维坐标。

架构（工作线程模式，与 detection_node / visualization_node 相同）
────────────────────────────────────────────────────────────────────────────────
_on_synced（ROS spin 线程）
  └─ put_nowait → _work_queue（maxsize=1，丢弃旧帧）
        └─ _worker_loop（后台线程）
              ├─ 解码深度图（C RVL 约 0.2 ms，Python 约 300 ms）
              └─ 发布 Centers3D

spin 线程从 _on_synced 返回的时间复杂度为 O(1)，不会被深度解码或 numpy 运算阻塞。
这消除了由于 RVL Python 解码占满 spin 线程而导致的"运行 1 秒、卡顿 10 秒"问题。
"""

from __future__ import annotations

import queue
import threading
from typing import Optional, Tuple

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

_QOS_RESULT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

from center_depth_msgs.msg import Centers2D, Centers3D
from center_depth_pipeline.image_numpy import (
    compressed_depth_to_np,
    imgmsg_to_depth_np,
)
from center_depth_pipeline.param_utils import (
    declare_and_get_bool,
    declare_and_get_float,
    declare_and_get_int,
    declare_and_get_str,
)
from center_depth_pipeline.queue_utils import replace_latest


class DepthLookupNode(Node):
    def __init__(self) -> None:
        super().__init__("center_depth_lookup_node")

        depth_topic = declare_and_get_str(
            self, "depth_topic", "/ascamera_hp60c/camera_publisher/depth0/image_raw/compressedDepth"
        )
        self._use_comp_d = declare_and_get_bool(self, "use_compressed_depth", True)
        camera_info_topic = declare_and_get_str(
            self, "camera_info_topic", "/ascamera_hp60c/camera_publisher/rgb0/camera_info"
        )
        centers_topic = declare_and_get_str(self, "centers_topic", "/object_centers_2d")
        out_topic = declare_and_get_str(self, "centers_3d_topic", "/object_centers_3d")
        slop = declare_and_get_float(self, "sync_slop_sec", 0.08)
        self._aligned = declare_and_get_bool(self, "depth_aligned_to_rgb", True)
        self._sample_r = declare_and_get_int(self, "sample_radius", 2)
        self._min_d = declare_and_get_float(self, "min_depth_m", 0.05)
        self._max_d = declare_and_get_float(self, "max_depth_m", 4.0)

        # CameraInfo 由 CameraInfo 回调（spin 线程）写入，由工作线程读取。
        # Python 属性赋值在 GIL 保护下是原子操作，无需显式加锁。
        self._camera_info: Optional[CameraInfo] = None
        self._depth_width:  int = 0
        self._depth_height: int = 0

        if not self._aligned:
            self.get_logger().warn(
                "depth_aligned_to_rgb=false: RGB (u,v) may not map to correct depth pixels."
            )

        self._pub = self.create_publisher(Centers3D, out_topic, _QOS_RESULT)
        self.create_subscription(CameraInfo, camera_info_topic, self._on_cam_info, 10)

        if self._use_comp_d:
            depth_sub = Subscriber(
                self, CompressedImage, depth_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self.get_logger().info(f"Depth (compressedDepth): {depth_topic}")
        else:
            depth_sub = Subscriber(
                self, Image, depth_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self.get_logger().info(f"Depth (raw): {depth_topic}")

        centers_sub = Subscriber(
            self, Centers2D, centers_topic,
            qos_profile=qos_profile_sensor_data,
        )
        # queue_size=2：同步器缓冲区中只保留最新的一对消息。
        # 值过大会导致旧消息对积累，输出时间错位。
        self._sync = ApproximateTimeSynchronizer(
            [depth_sub, centers_sub], queue_size=2, slop=slop,
        )
        self._sync.registerCallback(self._on_synced)

        # ── 工作线程 ──────────────────────────────────────────────────────────
        self._work_queue: queue.Queue = queue.Queue(maxsize=1)
        self._quit = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="depth_decode",
        )
        self._worker.start()

        self.get_logger().info(f"CameraInfo: {camera_info_topic}")
        self.get_logger().info(f"Centers in: {centers_topic}")
        self.get_logger().info(f"Centers 3D out: {out_topic}")
        self.get_logger().info(
            f"Sync slop: {slop}s | depth range [{self._min_d}, {self._max_d}] m"
        )

    # ── ROS spin 线程回调（必须为 O(1)）──────────────────────────────────────

    def _on_cam_info(self, msg: CameraInfo) -> None:
        self._camera_info  = msg
        self._depth_width  = int(msg.width)
        self._depth_height = int(msg.height)

    def _on_synced(self, depth_msg, centers_msg: Centers2D) -> None:
        """丢弃旧帧入队：spin 线程以 O(1) 时间返回。"""
        replace_latest(self._work_queue, (depth_msg, centers_msg))

    # ── 工作线程（繁重的解码 + 查询在此执行）────────────────────────────────

    def _worker_loop(self) -> None:
        while not self._quit.is_set():
            try:
                pair = self._work_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            depth_msg, centers_msg = pair

            if self._camera_info is None:
                continue

            try:
                depth, enc = self._decode_depth(depth_msg)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(
                    f"Depth decode failed: {e}", throttle_duration_sec=2.0
                )
                continue

            try:
                out = self._process(depth, enc, centers_msg)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(
                    f"Depth lookup error: {e}", throttle_duration_sec=2.0
                )
                continue

            self._pub.publish(out)

    # ── 处理辅助函数 ──────────────────────────────────────────────────────────

    def _decode_depth(self, msg) -> Tuple[np.ndarray, str]:
        if isinstance(msg, CompressedImage):
            return compressed_depth_to_np(
                msg,
                width=self._depth_width,
                height=self._depth_height,
            )
        return imgmsg_to_depth_np(msg)

    def _read_depth_m(self, depth: np.ndarray, enc: str, u: int, v: int) -> Optional[float]:
        h, w = depth.shape[:2]
        if u < 0 or v < 0 or u >= w or v >= h:
            return None
        r = self._sample_r
        patch = depth[max(0, v - r):min(h, v + r + 1),
                      max(0, u - r):min(w, u + r + 1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None
        d = float(np.median(valid))
        if enc == "16UC1":
            d /= 1000.0
        if d < self._min_d or d > self._max_d:
            return None
        return d

    @staticmethod
    def _pixel_to_xyz(u: int, v: int, z: float, info: CameraInfo) -> Tuple[float, float, float]:
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]
        if fx == 0.0 or fy == 0.0:
            return float("nan"), float("nan"), float("nan")
        return (u - cx) * z / fx, (v - cy) * z / fy, z

    def _process(self, depth: np.ndarray, enc: str, centers_msg: Centers2D) -> Centers3D:
        info = self._camera_info  # 本地引用；GIL 原子读，安全
        n = len(centers_msg.u)
        obj_class = centers_msg.obj_class
        score = centers_msg.score
        u_arr = centers_msg.u
        v_arr = centers_msg.v
        angle_rad = centers_msg.angle_rad
        angle_deg = centers_msg.angle_deg
        cx_norm = centers_msg.cx_norm
        cy_norm = centers_msg.cy_norm

        out = Centers3D()
        out.header       = centers_msg.header
        out.frame_width  = int(centers_msg.frame_width)
        out.frame_height = int(centers_msg.frame_height)
        out.kpts_per_det = int(centers_msg.kpts_per_det)
        out.kpt_u        = list(centers_msg.kpt_u)
        out.kpt_v        = list(centers_msg.kpt_v)
        out.kpt_conf     = list(centers_msg.kpt_conf)

        out_obj_class = out.obj_class
        out_score = out.score
        out_u = out.u
        out_v = out.v
        out_angle_rad = out.angle_rad
        out_angle_deg = out.angle_deg
        out_cx_norm = out.cx_norm
        out_cy_norm = out.cy_norm
        out_depth_m = out.depth_m
        out_x = out.x
        out_y = out.y
        out_z = out.z

        score_n = len(score)
        angle_rad_n = len(angle_rad)
        angle_deg_n = len(angle_deg)
        cx_norm_n = len(cx_norm)
        cy_norm_n = len(cy_norm)

        n_invalid = 0
        for i in range(n):
            u = int(u_arr[i])
            v = int(v_arr[i])
            zm = self._read_depth_m(depth, enc, u, v)

            out_obj_class.append(str(obj_class[i]))
            out_score.append(float(score[i]) if i < score_n else 0.0)
            out_u.append(u)
            out_v.append(v)
            out_angle_rad.append(float(angle_rad[i]) if i < angle_rad_n else 0.0)
            out_angle_deg.append(float(angle_deg[i]) if i < angle_deg_n else 0.0)
            out_cx_norm.append(float(cx_norm[i]) if i < cx_norm_n else 0.0)
            out_cy_norm.append(float(cy_norm[i]) if i < cy_norm_n else 0.0)

            if zm is not None:
                x, y, z = self._pixel_to_xyz(u, v, zm, info)
                out_depth_m.append(float(zm))
                out_x.append(float(x))
                out_y.append(float(y))
                out_z.append(float(z))
            else:
                n_invalid += 1
                nan = float("nan")
                out_depth_m.append(nan)
                out_x.append(nan)
                out_y.append(nan)
                out_z.append(nan)

        if n_invalid > 0:
            self.get_logger().debug(
                f"{n_invalid}/{n} detections: no valid depth "
                f"(range {self._min_d}–{self._max_d} m)",
                throttle_duration_sec=2.0,
            )
        return out


def main() -> None:
    rclpy.init()
    node = DepthLookupNode()
    try:
        rclpy.spin(node)
    finally:
        node._quit.set()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
