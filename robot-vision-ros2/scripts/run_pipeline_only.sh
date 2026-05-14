#!/usr/bin/env bash
# 仅启动相机流水线：detection_node + depth_node（不含 base_logger）
# 所有参数从 config.yaml 读取，也可通过命令行覆盖，例如：
#   bash run_pipeline_only.sh conf_threshold:=0.3
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

ros2 launch center_depth_pipeline yolo_center_depth_hp60c.launch.py "$@"
