# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 仓库结构

三个子项目形成完整的机器人视觉抓取流水线：

```
handeye-calibration/   → 手眼标定（输出 T_cam2ee 矩阵）
yolo-pose-pipeline/    → YOLO Pose 模型训练（输出 best.pt + pose_classes.yaml）
robot-vision-ros2/     → ROS2 部署（实时检测 → 深度 → 基座坐标 → 抓取）
```

跨项目依赖：`robot-vision-ros2/config.yaml` 中 `model.weights_path` 和 `model.pose_classes_path` 引用 `yolo-pose-pipeline` 的绝对路径输出。

---

## ROS2 工作区（robot-vision-ros2）

### 构建与启动

```bash
cd robot-vision-ros2
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# 启动完整流水线（HP60C 相机）
ros2 launch center_depth_pipeline yolo_center_depth_hp60c.launch.py

# 启动目标位姿记录节点（参数从 config.yaml 读取）
ros2 launch object_base_logger object_base_logger.launch.py
```

### 包结构

| 包 | 说明 |
|----|------|
| `center_depth_msgs` | 自定义消息：`Centers2D`、`Centers3D`、`DetectionItem` |
| `center_depth_pipeline` | 核心流水线：检测→深度→可视化→抓取 |
| `object_base_logger` | 订阅 Centers3D，转换到基座坐标并记录/上报 |

### 节点话题流

```
相机 → /rgb (compressed)
     → /depth (compressedDepth)
     → /camera_info

detection_node  ←── /rgb
                ──► /object_centers_2d  (Centers2D)

depth_node      ←── /object_centers_2d + /depth + /camera_info
                ──► /object_centers_3d  (Centers3D)

visualization_node ←── /rgb + /object_centers_3d
                   ──► /object_overlay/image

grasp_node      ←── /object_centers_3d
                ──► /grasp_target       (PoseStamped)

base_logger_node ←── /object_centers_3d + WebSocket(T_ee2base)
                 ──► 控制台日志 / HTTP 上报
```

### 关键实现细节

- **推理线程**：`detection_node` 用 `queue.Queue(maxsize=1)` + 独立线程，保证实时性、避免帧积压
- **深度解码**：`image_numpy.py` 通过 ctypes 调用 C 动态库（RVL/PNG compressedDepth 两种格式）
- **角度平滑**：循环 EMA，处理 ±90° 边界翻转（`_update_angle_ema`）
- **坐标变换链**：`p_base = T_ee2base @ T_cam2ee @ p_cam`，`T_cam2ee` 来自 `config.yaml` → `cam_hand_eye.py`
- **线程安全**：GIL 原子属性读写，无需显式锁（单帧引用赋值）
- **压缩话题**：默认使用 `CompressedImage` / `compressedDepth` 节省 Wi-Fi 带宽

### 配置

所有参数集中在 `robot-vision-ros2/config.yaml`，CLI 参数优先级高于配置文件。
更换标定结果只需修改 `hand_eye.T_cam2ee` 矩阵。

---

## 手眼标定（handeye-calibration）

```bash
cd handeye-calibration
python eyeInHand.py   # 输出 T_cam2ee，填入 config.yaml hand_eye.T_cam2ee
```

---

## YOLO 训练（yolo-pose-pipeline）

```bash
cd yolo-pose-pipeline
# 数据集准备
python merge_datasets.py
# 训练
python train_pose.py
# 输出：pose/runs/<dataset>/weights/best.pt
#       <dataset>/pose_classes.yaml
```

训练完成后更新 `robot-vision-ros2/config.yaml` 中的 `model.weights_path` 和 `model.pose_classes_path`。
