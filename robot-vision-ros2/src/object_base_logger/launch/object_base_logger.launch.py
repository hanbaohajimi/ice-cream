"""
完整流水线 launch：detection_node + depth_node + base_logger_node。
所有参数默认值从仓库根 config.yaml 读取，命令行传参优先级更高。

用法：
  ros2 launch object_base_logger object_base_logger.launch.py
"""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _read_config() -> dict:
    try:
        here = Path(__file__).resolve().parent
        for _ in range(6):
            candidate = here / "config.yaml"
            if candidate.exists():
                return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            here = here.parent
    except Exception as e:
        print(f"[object_base_logger launch] 读取 config.yaml 失败: {e}")
    return {}


def generate_launch_description() -> LaunchDescription:
    cfg = _read_config()
    net = cfg.get("network", {})
    log = cfg.get("logger",  {})

    # ── 相机流水线（detection + depth）────────────────────────────────────────
    pipeline_launch = os.path.join(
        get_package_share_directory("center_depth_pipeline"),
        "launch", "yolo_center_depth_hp60c.launch.py",
    )

    # ── base_logger 参数 ───────────────────────────────────────────────────────
    ws_host                        = LaunchConfiguration("ws_host")
    ws_port                        = LaunchConfiguration("ws_port")
    target_class                   = LaunchConfiguration("target_class")
    log_detection_min_interval_sec = LaunchConfiguration("log_detection_min_interval_sec")
    log_best_detection_only        = LaunchConfiguration("log_best_detection_only")
    head_ingestion_enabled         = LaunchConfiguration("head_ingestion_enabled")
    head_http_url                  = LaunchConfiguration("head_http_url")
    head_http_timeout_sec          = LaunchConfiguration("head_http_timeout_sec")
    head_role                      = LaunchConfiguration("head_role")
    head_label_prefix              = LaunchConfiguration("head_label_prefix")
    head_position_z_offset_m       = LaunchConfiguration("head_position_z_offset_m")

    return LaunchDescription([
        # ── base_logger 参数声明 ───────────────────────────────────────────────
        DeclareLaunchArgument("ws_host",      default_value=str(net.get("ws_host",      "127.0.0.1"))),
        DeclareLaunchArgument("ws_port",      default_value=str(net.get("ws_port",      8765))),
        DeclareLaunchArgument("target_class", default_value=str(log.get("target_class", ""))),
        DeclareLaunchArgument("log_detection_min_interval_sec",
                              default_value=str(log.get("log_detection_min_interval_sec", 1.0))),
        DeclareLaunchArgument("log_best_detection_only",
                              default_value=str(log.get("log_best_detection_only", False)).lower()),
        DeclareLaunchArgument("head_ingestion_enabled",
                              default_value=str(log.get("head_ingestion_enabled", False)).lower()),
        DeclareLaunchArgument("head_http_url",
                              default_value=str(net.get("head_http_url", ""))),
        DeclareLaunchArgument("head_http_timeout_sec",
                              default_value=str(net.get("head_http_timeout_sec", 0.25))),
        DeclareLaunchArgument("head_role",         default_value=str(log.get("head_role",         "object"))),
        DeclareLaunchArgument("head_label_prefix", default_value=str(log.get("head_label_prefix", ""))),
        DeclareLaunchArgument(
            "head_position_z_offset_m",
            default_value=str(log.get("head_position_z_offset_m", 0.0)),
        ),

        # ── 包含相机流水线（detection_node + depth_node）─────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pipeline_launch),
        ),

        # ── base_logger_node ──────────────────────────────────────────────────
        Node(
            package="object_base_logger",
            executable="base_logger_node",
            name="object_base_logger_node",
            parameters=[{
                "ws_host":                        ws_host,
                "ws_port":                        ParameterValue(ws_port, value_type=int),
                "target_class":                   target_class,
                "log_detection_min_interval_sec": ParameterValue(log_detection_min_interval_sec, value_type=float),
                "log_best_detection_only":        ParameterValue(log_best_detection_only, value_type=bool),
                "head_ingestion_enabled":         ParameterValue(head_ingestion_enabled, value_type=bool),
                "head_http_url":                  head_http_url,
                "head_http_timeout_sec":          ParameterValue(head_http_timeout_sec, value_type=float),
                "head_role":                      head_role,
                "head_label_prefix":              head_label_prefix,
                "head_position_z_offset_m":       ParameterValue(head_position_z_offset_m, value_type=float),
            }],
            output="screen",
        ),
    ])
