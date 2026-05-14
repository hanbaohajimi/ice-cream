#!/usr/bin/env bash
# 编译 robot-vision-ros2 工作区
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"

source /opt/ros/humble/setup.bash

cd "$WS"
colcon build --symlink-install

echo ""
echo "编译完成。在当前终端执行以下命令激活环境，然后再启动节点："
echo "  source $WS/install/setup.bash"
echo "  bash $WS/scripts/run.sh"
