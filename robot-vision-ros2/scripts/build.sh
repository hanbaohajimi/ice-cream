#!/usr/bin/env bash
# 编译 robot-vision-ros2 工作区
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"

source /opt/ros/humble/setup.bash

cd "$WS"
colcon build --symlink-install
source "$WS/install/setup.bash"

echo ""
echo "编译完成。运行以下命令启动："
echo "  bash $WS/scripts/run.sh"
