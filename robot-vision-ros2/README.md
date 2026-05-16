# robot-vision-ros2

基于 ROS2 的机器人视觉节点：YOLO 目标检测 + 深度解算 + 手眼坐标变换 + 基座坐标日志上报。

## 目录结构

```
robot-vision-ros2/
├── config.yaml                    ← 所有参数在此配置（含 T_CAM2EE 外参）
└── src/
    ├── center_depth_msgs/         ← 自定义 ROS2 消息（Centers2D, Centers3D）
    ├── center_depth_pipeline/     ← 主视觉流水线
    │   ├── detection_node.py      ← YOLO 检测节点
    │   ├── depth_node.py          ← 深度解算节点
    │   └── launch/
    │       ├── yolo_center_depth.launch.py       ← 通用 launch
    │       └── yolo_center_depth_hp60c.launch.py ← HP60C 相机专用 launch
    └── object_base_logger/        ← 三维坐标日志 + HTTP 上报节点
        └── launch/
            └── object_base_logger.launch.py
```

## 快速开始

### 1. 配置参数

**首次使用必须修改 `config.yaml` 中的以下字段：**

```yaml
hand_eye:
  T_cam2ee:          # 将 eyeInHand.py 的 PARK 方法结果填入此处（4×4 矩阵）
    - [...]

model:
  weights_path: "/path/to/best.pt"
  pose_classes_path: "/path/to/pose_classes.yaml"

network:
  ws_host: "192.168.31.142"   # 机械臂 IP
  head_http_url: "192.168.31.71:8776"  # Head 设备 IP:port
```

### 2. 编译工作区

```bash
bash scripts/build.sh
```

### 3. 一键启动完整流水线

```bash
bash scripts/run.sh

# 覆盖参数示例：
bash scripts/run.sh conf_threshold:=0.3 ws_host:=192.168.1.100
```

这条命令同时启动：`detection_node` + `depth_node` + `base_logger_node`。

### 4. 仅启动相机流水线（不含日志节点）

```bash
bash scripts/run_pipeline_only.sh
```

### 5. 诊断工具

```bash
# 检查各节点话题连通性：
ros2 run center_depth_pipeline pipeline_doctor
```

## 更换手眼标定结果

1. 运行 `handeye-calibration/src/eyeInHand.py` 得到新的 PARK 方法旋转矩阵和平移向量
2. 将结果填入本仓库 `config.yaml` 的 `hand_eye.T_cam2ee`（4×4 格式）
3. 重启 ROS2 节点即可生效，**无需重新编译**

## 话题结构

```
相机 RGB  ──→ detection_node ──→ /object_centers_2d
相机 Depth ──→ depth_node ────→ /object_centers_3d ──→ base_logger_node → 日志/HTTP上报
```

## 参数说明

所有参数均在 `config.yaml` 中配置，每个参数都有注释说明其含义和原始来源。
ROS2 launch 命令行参数优先级高于 `config.yaml`。
