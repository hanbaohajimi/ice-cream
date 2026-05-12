#!/usr/bin/env bash
# rosidl + CMake use "base:relative" tuples with a colon; some non-ASCII paths
# break interface generation. Building only center_depth_msgs under /tmp avoids that.
set -eo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d /tmp/center_depth_msgs_build.XXXXXX)"
mkdir -p "$TMP/src"
cp -a "$WS/src/center_depth_msgs" "$TMP/src/"
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash 2>/dev/null || source /opt/ros/jazzy/setup.bash
cd "$TMP"
colcon build --packages-select center_depth_msgs --symlink-install
echo ""
echo "Built center_depth_msgs in: $TMP/install"
echo "Overlay and build the rest from your workspace, e.g.:"
echo "  source $TMP/install/setup.bash"
echo "  cd \"$WS\" && colcon build --packages-select center_depth_pipeline --symlink-install"
