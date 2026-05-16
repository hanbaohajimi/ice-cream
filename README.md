# Robot Vision Grasping Pipeline

基于 YOLO Pose + ROS2 的机器人视觉感知与抓取坐标上报系统。通过相机识别几何目标，计算目标在机械臂基座坐标系下的三维位置和姿态角，供机械臂抓取使用。

---

## 系统概览

```
handeye-calibration/   → 手眼标定（输出 T_cam2ee）
yolo-pose-pipeline/    → YOLO Pose 模型训练（输出 best.pt）
robot-vision-ros2/     → ROS2 在线部署（检测 → 深度 → 基座坐标 → 上报）
```

三个子项目单向依赖：标定输出外参 → ROS2；训练输出模型 → ROS2。ROS2 项目在运行时读取两者结果，完成实时感知。

### 完整数据流

```
HP60C 相机
  RGB (compressed)  ──► detection_node  ──► /object_centers_2d
                                                   │
  Depth (compressedDepth) ─────────────► depth_node ──► /object_centers_3d
  CameraInfo ──────────────────────────────┘                   │
                                                    base_logger_node
                                                    ├── WebSocket (T_ee2base)
                                                    ├── T_cam2ee (config.yaml)
                                                    └── 控制台日志 / HTTP 上报
```

坐标变换链：`p_base = T_ee2base @ T_cam2ee @ p_cam`

---

## 子项目说明

### 1. 手眼标定（handeye-calibration）

相机安装在机械臂末端，求解相机坐标系到末端执行器坐标系的变换 T_cam2ee。

**采集：**
```bash
cd handeye-calibration/src
python online_capture.py   # 按空格采集，同步保存图像与末端位姿
```
- 输出图像：`images/*.jpg`
- 输出位姿：`images/poses.txt`（x,y,z,Rx,Ry,Rz，米/弧度）
- 输出元数据：`capture_meta.jsonl`（含时间戳偏差，用于质量评估）

**标定：**
```bash
python eyeInHand.py
```
- 使用棋盘格角点做相机标定，调用 OpenCV `calibrateHandEye`
- 自动评估 TSAI / PARK / HORAUD 三种方法，选择最优
- 将最优 T_cam2ee 自动写入 `robot-vision-ros2/config.yaml`

---

### 2. YOLO Pose 训练（yolo-pose-pipeline）

为几何目标（圆形、三角形、梯形、正方形）训练带关键点的 YOLO Pose 模型。

```bash
cd yolo-pose-pipeline

# 从视频抽帧（约 2fps）
python pose/src/extract_frames.py

# LabelMe JSON → YOLO Pose txt + 数据集划分
python pose/src/prepare_data.py

# 训练（从 yolo11n-pose.pt 预训练权重开始）
python pose/src/train_pose.py

# 实时验证（摄像头直接推流）
python pose/src/realtime_detect_pose.py
```

训练完成后更新 `robot-vision-ros2/config.yaml` 中的 `model.weights_path` 和 `model.pose_classes_path`。

---

### 3. ROS2 在线部署（robot-vision-ros2）

#### 构建与启动

```bash
cd robot-vision-ros2
source /opt/ros/humble/setup.bash
bash scripts/build.sh

# 启动完整流水线（detection + depth + base_logger）
bash scripts/run.sh

# 仅启动相机流水线（不含坐标上报）
bash scripts/run_pipeline_only.sh
```

#### 包结构

| 包 | 说明 |
|----|------|
| `center_depth_msgs` | 自定义消息：`Centers2D`、`Centers3D` |
| `center_depth_pipeline` | 检测节点 + 深度节点 |
| `object_base_logger` | 坐标变换 + 日志/HTTP 上报 |

#### 节点说明

**detection_node**
- 订阅压缩 RGB，运行 YOLO Pose，发布 `/object_centers_2d`
- 使用关键点计算几何中心（而非框中心），按形状定义姿态角
- 推理线程 + `queue(maxsize=1)`：只处理最新帧，避免延迟累积
- 稳定帧门控（`min_stable_frames=2`）过滤单帧误检
- 循环 EMA 平滑角度，正确处理 ±90° 边界
- CUDA FP16 + GPU 预热，降低延迟

**depth_node**
- 时间戳同步深度图与 `Centers2D`，后台线程完成反投影
- 在中心点 `sample_radius` 像素范围内取有效深度中位数
- 反投影公式：`x=(u-cx)*z/fx, y=(v-cy)*z/fy`
- 无效深度写 NaN，不丢弃目标

**base_logger_node**
- 订阅 `Centers3D`，WebSocket 接收机械臂末端位姿 T_ee2base
- 完成坐标变换后输出控制台日志或 POST 到 Head 设备
- 支持按类别过滤、只上报最高置信度目标、z 方向额外偏移

---

## 配置

所有参数集中在 `robot-vision-ros2/config.yaml`，分区如下：

| 分区 | 说明 |
|------|------|
| `hand_eye` | T_cam2ee 4×4 矩阵（标定后自动写入） |
| `camera` | RGB/深度话题、是否使用压缩格式 |
| `model` | 权重路径、关键点配置、推理设备、FP16 |
| `detection` | 置信度阈值、稳定帧数、EMA 系数、日志开关 |
| `depth` | 同步窗口、采样半径、深度有效范围 |
| `network` | 机械臂 WebSocket 地址、Head HTTP 上报地址 |
| `logger` | 目标类别过滤、日志频率、HTTP 上报开关、z 偏移 |

更换标定结果只需修改 `hand_eye.T_cam2ee`，无需重新编译。CLI 参数优先级高于配置文件。

---

## 关键设计决策

| 问题 | 方案 |
|------|------|
| 框中心不准 | 用关键点质心作为几何中心 |
| 角度抖动 | 循环 EMA，处理 ±90° 边界 |
| 单帧误检 | 稳定帧门控（连续 N 帧才发布） |
| 高帧率 + 慢推理 | 最新帧队列，只处理当前帧 |
| 带宽压力 | 压缩话题 + BEST_EFFORT depth=1 QoS |
| cv_bridge 兼容 | 直接用 numpy + cv2，支持 RVL/PNG 深度解码 |
| 现场调参 | 全部参数集中 config.yaml，无需改代码 |

---

## 环境依赖

- ROS2 Humble
- Python 3.10+
- Ultralytics YOLO（yolo11n-pose）
- OpenCV、NumPy
- websockets（base_logger WebSocket 客户端）

---

## 诊断工具

```bash
# 检查各话题发布者/订阅者/帧率
python robot-vision-ros2/src/center_depth_pipeline/center_depth_pipeline/pipeline_doctor.py
```
