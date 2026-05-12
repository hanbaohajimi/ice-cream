#!/usr/bin/env python3
"""
将 Centers3D 检测结果转换为机械臂的 6-DOF 抓取位姿。

对每个检测到的目标，本节点执行以下操作：
  1. 用各类别独立的 EMA 平滑三维坐标 (x, y, z)，降低深度噪声。
  2. 根据图像平面内的 angle_rad 构建夹爪四元数：
       - 接近方向：相机 +Z 轴（朝场景/桌面方向）。
       - 夹爪张开轴：在相机 XY 平面内从 +X 旋转 angle_rad。
       - 四元数：Rz(angle_rad) → qx=0, qy=0, qz=sin(a/2), qw=cos(a/2)。
  3. 可选地沿 -Z 方向后退，生成预抓取悬停点。
  4. 可选地将所有位姿变换到目标 TF 坐标系（如 base_link）。

发布话题
─────────
  /grasp_poses   geometry_msgs/PoseArray   所有有效抓取位姿（相机坐标系或目标坐标系）
  /grasp_target  geometry_msgs/PoseStamped 得分最高的有效抓取位姿

参数
──────────
  centers_3d_topic   : 输入 Centers3D 话题              （默认 /object_centers_3d）
  grasp_poses_topic  : PoseArray 输出话题               （默认 /grasp_poses）
  grasp_target_topic : 最优抓取 PoseStamped 输出话题    （默认 /grasp_target）
  target_frame       : 目标 TF 坐标系                   （默认 "" = 相机坐标系）
  pos_ema_alpha      : 位置 EMA 步长（0=冻结, 1=原始值）（默认 0.35）
  min_score          : 最低检测置信度                   （默认 0.25）
  approach_offset_m  : 预抓取悬停距离（沿 -Z）          （默认 0.05 m）
  min_depth_m        : 跳过距离小于此值的抓取           （默认 0.05 m）
  max_depth_m        : 跳过距离大于此值的抓取           （默认 4.0 m）

抓取方向说明
───────────────────────
  相机光学坐标系：Z 指向场景，X 向右，Y 向下（ROS 标准约定）。
  对于正上方俯视工作台的相机：
    - 相机 Z  ≡ 世界 -Z 方向（向下）。
    - angle_rad 为图像平面中目标长轴的偏航角。
    - Rz(angle_rad) 给出俯视抓取的正确夹爪偏航角。
  对于其他安装角度，调用方应设置 target_frame，
  依靠 TF 将方向变换到正确的世界坐标系。
"""

from __future__ import annotations

import math
import queue
import threading
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion

from center_depth_msgs.msg import Centers3D

_QOS_RESULT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


# ─────────────────────────── 数学辅助函数 ────────────────────────────────────

def _yaw_to_quat(angle_rad: float) -> Quaternion:
    """纯 Z 轴旋转 → ROS 四元数，无需 scipy。"""
    half = angle_rad * 0.5
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


def _ema(prev: float, new: float, alpha: float) -> float:
    return prev + alpha * (new - prev)


# ─────────────────────────── ROS 节点 ────────────────────────────────────────

class GraspNode(Node):

    def __init__(self) -> None:
        super().__init__("grasp_pose_node")

        self.declare_parameter("centers_3d_topic",   "/object_centers_3d")
        self.declare_parameter("grasp_poses_topic",  "/grasp_poses")
        self.declare_parameter("grasp_target_topic", "/grasp_target")
        self.declare_parameter("target_frame",       "")
        self.declare_parameter("pos_ema_alpha",      0.35)
        self.declare_parameter("min_score",          0.25)
        self.declare_parameter("approach_offset_m",  0.05)
        self.declare_parameter("min_depth_m",        0.05)
        self.declare_parameter("max_depth_m",        4.0)

        centers_topic  = self.get_parameter("centers_3d_topic").get_parameter_value().string_value
        grasp_topic    = self.get_parameter("grasp_poses_topic").get_parameter_value().string_value
        target_topic   = self.get_parameter("grasp_target_topic").get_parameter_value().string_value
        self._tf_frame = self.get_parameter("target_frame").get_parameter_value().string_value.strip()
        self._pos_a    = float(self.get_parameter("pos_ema_alpha").get_parameter_value().double_value)
        self._min_sc   = float(self.get_parameter("min_score").get_parameter_value().double_value)
        self._offset   = float(self.get_parameter("approach_offset_m").get_parameter_value().double_value)
        self._min_d    = float(self.get_parameter("min_depth_m").get_parameter_value().double_value)
        self._max_d    = float(self.get_parameter("max_depth_m").get_parameter_value().double_value)

        # 各类别位置 EMA：类别 → (x, y, z)
        self._pos_ema: Dict[str, Tuple[float, float, float]] = {}

        # TF2（可选）
        self._tf_buffer   = None
        self._tf_listener = None
        if self._tf_frame:
            try:
                from tf2_ros import Buffer, TransformListener
                self._tf_buffer   = Buffer()
                self._tf_listener = TransformListener(self._tf_buffer, self)
                self.get_logger().info(f"TF2 enabled: will transform poses to '{self._tf_frame}'")
            except ImportError:
                self.get_logger().warn("tf2_ros not importable; publishing in camera frame.")
                self._tf_frame = ""

        self._pub_poses  = self.create_publisher(PoseArray,   grasp_topic,  _QOS_RESULT)
        self._pub_target = self.create_publisher(PoseStamped, target_topic, _QOS_RESULT)

        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="grasp_worker")
        self._worker.start()

        self.create_subscription(
            Centers3D, centers_topic, self._on_centers,
            qos_profile=_QOS_RESULT,
        )

        self.get_logger().info(
            f"GraspNode: {centers_topic} → {grasp_topic} | {target_topic} | "
            f"pos_ema={self._pos_a}  min_score={self._min_sc}  "
            f"approach_offset={self._offset:.3f}m  "
            f"depth=[{self._min_d},{self._max_d}]m"
        )

    # ── ROS 回调（O(1)，禁止阻塞）───────────────────────────────────────────

    def _on_centers(self, msg: Centers3D) -> None:
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(msg)
        except queue.Full:
            pass

    # ── 工作线程 ──────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while True:
            try:
                msg: Centers3D = self._q.get(timeout=1.0)
            except queue.Empty:
                if not rclpy.ok():
                    break
                continue
            try:
                self._process(msg)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(
                    f"Grasp compute error: {e}", throttle_duration_sec=2.0
                )

    # ── 抓取计算 ──────────────────────────────────────────────────────────────

    def _smooth_pos(
        self, cls: str, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        if cls not in self._pos_ema:
            self._pos_ema[cls] = (x, y, z)
            return x, y, z
        px, py, pz = self._pos_ema[cls]
        a = self._pos_a
        sx, sy, sz = _ema(px, x, a), _ema(py, y, a), _ema(pz, z, a)
        self._pos_ema[cls] = (sx, sy, sz)
        return sx, sy, sz

    def _process(self, msg: Centers3D) -> None:
        n = len(msg.obj_class)

        pose_arr = PoseArray()
        pose_arr.header = msg.header

        best_score   = -1.0
        best_stamped: Optional[PoseStamped] = None
        detected_cls: List[str] = []

        for i in range(n):
            cls   = str(msg.obj_class[i])
            score = float(msg.score[i]) if i < len(msg.score) else 0.0

            if score < self._min_sc:
                continue

            x = float(msg.x[i]) if i < len(msg.x) else float("nan")
            y = float(msg.y[i]) if i < len(msg.y) else float("nan")
            z = float(msg.z[i]) if i < len(msg.z) else float("nan")

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if z < self._min_d or z > self._max_d:
                continue

            detected_cls.append(cls)
            sx, sy, sz = self._smooth_pos(cls, x, y, z)
            angle_rad = float(msg.angle_rad[i]) if i < len(msg.angle_rad) else 0.0

            # 预抓取悬停：将夹爪沿相机 -Z 方向后退 offset 距离
            hover_z = sz - self._offset

            pose = Pose()
            pose.position.x  = sx
            pose.position.y  = sy
            pose.position.z  = hover_z
            pose.orientation = _yaw_to_quat(angle_rad)
            pose_arr.poses.append(pose)

            if score > best_score:
                best_score = score
                ps = PoseStamped()
                ps.header = msg.header
                ps.pose   = pose
                best_stamped = ps

        # 清除本帧消失的类别的 EMA 状态
        for cls in list(self._pos_ema):
            if cls not in detected_cls:
                del self._pos_ema[cls]

        if self._tf_frame and self._tf_buffer is not None:
            pose_arr, best_stamped = self._transform_to_target(
                pose_arr, best_stamped, msg.header
            )

        self._pub_poses.publish(pose_arr)
        if best_stamped is not None:
            self._pub_target.publish(best_stamped)

        if n > 0:
            cls_summary = ", ".join(
                f"{str(msg.obj_class[i])}({float(msg.score[i]) if i < len(msg.score) else 0.:.2f})"
                for i in range(n)
                if i < len(msg.score) and float(msg.score[i]) >= self._min_sc
            )
            self.get_logger().info(
                f"Grasp: {len(pose_arr.poses)}/{n} valid [{cls_summary}] "
                f"best_score={best_score:.2f}",
                throttle_duration_sec=1.0,
            )

    # ── TF2 变换（可选）───────────────────────────────────────────────────────

    def _transform_to_target(
        self,
        pose_arr: PoseArray,
        best: Optional[PoseStamped],
        header,
    ) -> Tuple[PoseArray, Optional[PoseStamped]]:
        try:
            import tf2_geometry_msgs  # noqa: F401  (registers do_transform_pose)
            from tf2_ros import (
                ConnectivityException,
                ExtrapolationException,
                LookupException,
            )

            tf = self._tf_buffer.lookup_transform(
                self._tf_frame,
                header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )

            new_arr = PoseArray()
            new_arr.header.frame_id = self._tf_frame
            new_arr.header.stamp    = header.stamp

            for pose in pose_arr.poses:
                ps_in = PoseStamped()
                ps_in.header = header
                ps_in.pose   = pose
                ps_out = tf2_geometry_msgs.do_transform_pose(ps_in, tf)
                new_arr.poses.append(ps_out.pose)

            best_out: Optional[PoseStamped] = None
            if best is not None:
                best_out = tf2_geometry_msgs.do_transform_pose(best, tf)
                best_out.header.frame_id = self._tf_frame

            return new_arr, best_out

        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f"TF transform to '{self._tf_frame}' failed: {e}; "
                "publishing in camera frame.",
                throttle_duration_sec=5.0,
            )
            return pose_arr, best


def main() -> None:
    rclpy.init()
    node = GraspNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
