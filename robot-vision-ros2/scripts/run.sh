#!/usr/bin/env bash
# 一键启动完整流水线：detection_node + depth_node + base_logger_node
# 所有参数从 config.yaml 读取，也可通过命令行覆盖，例如：
#   bash run.sh conf_threshold:=0.3 ws_host:=192.168.1.100
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

ros2 launch object_base_logger object_base_logger.launch.py "$@"
