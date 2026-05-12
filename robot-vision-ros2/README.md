# robot-vision-ros2

基于 ROS2 的机器人视觉节点：YOLO 目标检测 + 深度解算 + 手眼坐标变换 + 抓取位姿输出。

## 目录结构

```
robot-vision-ros2/
├── config.yaml                    ← 所有参数在此配置（含 T_CAM2EE 外参）
├── scripts/
│   └── build_msgs_in_tmp.sh       ← 在临时目录编译自定义消息（避免污染工作区）
└── src/
    ├── center_depth_msgs/         ← 自定义 ROS2 消息（Centers2D, Centers3D）
    ├── center_depth_pipeline/     ← 主视觉流水线
    │   ├── detection_node.py      ← YOLO 检测节点
    │   ├── depth_node.py          ← 深度解算节点
    │   ├── grasp_node.py          ← 抓取位姿节点
    │   ├── visualization_node.py  ← 可视化节点
    │   ├── cam_hand_eye.py        ← 手眼外参（从 config.yaml 读取）
    │   └── launch/
    │       ├── yolo_center_depth.launch.py       ← 通用 launch
    │       └── yolo_center_depth_hp60c.launch.py ← HP60C 相机专用 launch
    └── object_base_logger/        ← 三维坐标日志 + HTTP 上报节点
        └── config/
            └── head_ingestion.yaml
```

## 快速开始

### 1. 配置参数

**首次使用必须修改 `config.yaml` 中的以下字段：**

```yaml
hand_eye:
  T_cam2ee:          # 将 eyeInHand.py 的 PARK 方法结果填入此处（4×4 矩阵）
    - [...]

model:
  weights_path: "/home/ubuntu22/robot_projects/yolo-pose-pipeline/pose/runs/dataset54/weights/best.pt"
  pose_classes_path: "/home/ubuntu22/robot_projects/yolo-pose-pipeline/yolo_dataset54_421_merged/pose_classes.yaml"

network:
  ws_host: "192.168.31.142"   # 机械臂 IP
  head_http_url: "192.168.31.71:8776"  # Head 设备 IP:port
```

### 2. 编译工作区

```bash
cd robot-vision-ros2
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 3. 启动 HP60C 相机流水线

```bash
ros2 launch center_depth_pipeline yolo_center_depth_hp60c.launch.py

# 覆盖参数示例：
ros2 launch center_depth_pipeline yolo_center_depth_hp60c.launch.py \
  weights_path:=/absolute/path/to/best.pt \
  conf_threshold:=0.3
```

### 4. 启动日志节点

```bash
# 使用 config/head_ingestion.yaml 中的网络参数：
ros2 run object_base_logger base_logger_node \
  --ros-args --params-file src/object_base_logger/config/head_ingestion.yaml
```

### 5. 诊断工具

```bash
# 检查各节点话题连通性：
ros2 run center_depth_pipeline pipeline_doctor
```

## 更换手眼标定结果

1. 运行 `handeye-calibration/eyeInHand.py` 得到新的 PARK 方法旋转矩阵和平移向量
2. 将结果填入本仓库 `config.yaml` 的 `hand_eye.T_cam2ee`（4×4 格式）
3. 重启 ROS2 节点即可生效，**无需重新编译**

## 话题结构

```
相机 RGB ──→ detection_node ──→ /object_centers_2d
相机 Depth ──→ depth_node ────→ /object_centers_3d ──→ grasp_node → /grasp_target
                                                    ──→ visualization_node → /object_overlay/image
                                                    ──→ base_logger_node → 日志/HTTP上报
```

## 参数说明

所有参数均在 `config.yaml` 中配置，每个参数都有注释说明其含义和原始来源。
ROS2 launch 参数始终优先于 `config.yaml`。
