#!/usr/bin/env python3
"""YOLO + 深度流水线的运行时健康检查工具。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Type

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from center_depth_msgs.msg import Centers2D, Centers3D


DEFAULT_TOPICS = {
    "rgb": "/ascamera_hp60c/camera_publisher/rgb0/image/compressed",
    "depth": "/ascamera_hp60c/camera_publisher/depth0/image_raw/compressedDepth",
    "camera_info": "/ascamera_hp60c/camera_publisher/rgb0/camera_info",
    "centers2d": "/object_centers_2d",
    "centers3d": "/object_centers_3d",
    "overlay": "/object_overlay/image",
}


def _prepare_fastdds_profile() -> str:
    """将 transport_descriptors 移至 participant 之前，以兼容旧版 FastDDS 解析器。"""
    source = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE") or os.environ.get(
        "FASTDDS_DEFAULT_PROFILES_FILE"
    )
    if not source:
        return ""

    source_path = Path(source).expanduser()
    if not source_path.exists():
        return ""

    try:
        text = source_path.read_text(encoding="utf-8")
        participant_idx = text.find("<participant")
        transport_idx = text.find("<transport_descriptors")
        if participant_idx < 0 or transport_idx < 0 or transport_idx < participant_idx:
            return str(source_path)

        block_start = text.rfind("\n", 0, transport_idx) + 1
        block_end = text.find("</transport_descriptors>", transport_idx)
        if block_end < 0:
            return str(source_path)
        block_end += len("</transport_descriptors>")

        transport_block = text[block_start:block_end].strip() + "\n\n"
        without_transport = text[:block_start] + text[block_end:]
        insert_at = without_transport.find(">", without_transport.find("<profiles"))
        if insert_at < 0:
            return str(source_path)
        insert_at += 1

        fixed_text = (
            without_transport[:insert_at]
            + "\n"
            + transport_block
            + without_transport[insert_at:].lstrip()
        )
        fixed_path = Path.home() / ".yolo_ros_fastdds_peers.fixed.xml"
        fixed_path.write_text(fixed_text, encoding="utf-8")
        return str(fixed_path)
    except Exception:
        return ""


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查 center-depth 流水线各端点的连通性和实时帧数。"
    )
    parser.add_argument("--duration", type=float, default=3.0, help="帧数统计窗口（秒）")
    parser.add_argument("--rgb-topic", default=DEFAULT_TOPICS["rgb"])
    parser.add_argument("--depth-topic", default=DEFAULT_TOPICS["depth"])
    parser.add_argument("--camera-info-topic", default=DEFAULT_TOPICS["camera_info"])
    parser.add_argument("--centers2d-topic", default=DEFAULT_TOPICS["centers2d"])
    parser.add_argument("--centers3d-topic", default=DEFAULT_TOPICS["centers3d"])
    parser.add_argument("--overlay-topic", default=DEFAULT_TOPICS["overlay"])
    parser.add_argument(
        "--no-fix-fastdds",
        action="store_true",
        help="不为本进程重写 FASTRTPS_DEFAULT_PROFILES_FILE 的节点排序",
    )
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="诊断结果不健康时返回非零退出码",
    )
    return parser


def _topic_specs(args) -> Dict[str, Tuple[str, Type]]:
    return {
        "RGB": (args.rgb_topic, CompressedImage),
        "Depth": (args.depth_topic, CompressedImage),
        "CameraInfo": (args.camera_info_topic, CameraInfo),
        "Centers2D": (args.centers2d_topic, Centers2D),
        "Centers3D": (args.centers3d_topic, Centers3D),
        "Overlay": (args.overlay_topic, Image),
    }


def _qos_for(label: str):
    if label in {"RGB", "Depth", "Centers2D", "Centers3D"}:
        return qos_profile_sensor_data
    return 10


def _print_table(rows: Iterable[dict]) -> None:
    print("\n流水线端点及实时帧数")
    print("-" * 100)
    print(f"{'名称':<11} {'发布者':>10} {'订阅者':>11} {'帧数':>8}  话题")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['name']:<11} {row['pubs']:>10} {row['subs']:>11} "
            f"{row['frames']:>8}  {row['topic']}"
        )
    print("-" * 100)


def _diagnose(stats: Dict[str, dict]) -> Tuple[bool, str]:
    if stats["RGB"]["pubs"] == 0:
        return False, "无 RGB 相机发布者。请启动/检查 HP60C 相机节点、网络和 ROS_DOMAIN_ID。"
    if stats["RGB"]["frames"] == 0:
        return False, "RGB 发布者存在，但未收到 RGB 帧。请检查 DDS 对端/网络/QoS。"
    if stats["Depth"]["pubs"] == 0 or stats["CameraInfo"]["pubs"] == 0:
        return False, "RGB 正常，但深度图或 CameraInfo 发布者缺失。"
    if stats["Centers2D"]["pubs"] == 0:
        return False, "检测节点未发布 /object_centers_2d。"
    if stats["Centers2D"]["frames"] == 0:
        return False, "已收到 RGB 帧，但 YOLO 未发布 Centers2D。请检查模型/设备/日志。"
    if stats["Centers3D"]["pubs"] == 0:
        return False, "深度节点未发布 /object_centers_3d。"
    if stats["Centers3D"]["frames"] == 0:
        return False, "Centers2D 存在，但深度同步未产生 Centers3D。请检查深度帧/时间窗口。"
    if stats["Overlay"]["pubs"] == 0:
        return False, "可视化节点未发布叠加图像。"
    if stats["Overlay"]["frames"] == 0:
        return False, "叠加发布者存在，但未收到叠加帧。请检查可视化节点。"
    return True, "流水线正常：RGB、深度、检测、三维查询和叠加均已产生帧。"


def run_check(args) -> int:
    if not args.no_fix_fastdds:
        fixed_profile = _prepare_fastdds_profile()
        if fixed_profile:
            os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = fixed_profile
            os.environ.pop("FASTDDS_DEFAULT_PROFILES_FILE", None)
            print(f"FastDDS profile: {fixed_profile}")

    context = rclpy.context.Context()
    rclpy.init(args=None, context=context)
    node: Node = rclpy.create_node("center_depth_pipeline_doctor", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)

    counts = {name: 0 for name in _topic_specs(args)}
    subscriptions = []

    for name, (topic, msg_type) in _topic_specs(args).items():
        subscriptions.append(
            node.create_subscription(
                msg_type,
                topic,
                lambda _msg, key=name: counts.__setitem__(key, counts[key] + 1),
                _qos_for(name),
            )
        )

    try:
        end = node.get_clock().now().nanoseconds / 1e9 + max(args.duration, 0.5)
        while context.ok() and node.get_clock().now().nanoseconds / 1e9 < end:
            executor.spin_once(timeout_sec=0.1)

        stats: Dict[str, dict] = {}
        for name, (topic, _msg_type) in _topic_specs(args).items():
            stats[name] = {
                "name": name,
                "topic": topic,
                "pubs": len(node.get_publishers_info_by_topic(topic)),
                "subs": len(node.get_subscriptions_info_by_topic(topic)),
                "frames": counts[name],
            }

        _print_table(stats.values())
        ok, message = _diagnose(stats)
        print(f"\n诊断结果：{message}")
        return 0 if ok or not args.strict_exit_code else 2
    finally:
        subscriptions.clear()
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    raise SystemExit(run_check(args))


if __name__ == "__main__":
    main()
