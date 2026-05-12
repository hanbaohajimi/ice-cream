# object_base_logger

控制台输出检测结果；结合 **代码内 T_cam→ee**（`center_depth_pipeline.cam_hand_eye`）与 **Pi WebSocket 末端矩阵**，打印物体在 **机器人基坐标系** 下的三维坐标。可选将检测结果按 `camera2head.md` 上报到 head ingestion HTTP。

## 坐标约定

- `Centers3D` 的 `(x,y,z)`：相机光学坐标系（米），见 `center_depth_msgs/msg/Centers3D.msg`。
- `T_cam2ee`：见 `ros2_ws/src/center_depth_pipeline/center_depth_pipeline/cam_hand_eye.py`（不从 YAML 读取）。
- Pi `feedback.link5_hmat`：**T_ee→base**（4×4）。
- **基坐标系点（齐次）**：`p_base = T_ee2base @ T_cam2ee @ p_cam`

需能访问：**同一网络下的 Pi WebSocket**，且 Pi 端 `link5_hmat` 非 null。

## 使用

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu22/yolo_ros/ros2_ws/install/setup.bash

ros2 run object_base_logger base_logger_node
```

使用 head ingestion YAML 配置：

```bash
ros2 run object_base_logger base_logger_node --ros-args \
  --params-file /home/ubuntu22/yolo_ros/ros2_ws/install/object_base_logger/share/object_base_logger/config/head_ingestion.yaml
```

### 常用参数

```bash
ros2 run object_base_logger base_logger_node --ros-args \
  -p centers_3d_topic:=/object_centers_3d \
  -p ws_host:=192.168.x.x \
  -p ws_port:=8765 \
  -p target_class:=Square \
  -p log_empty_throttle_sec:=2.0 \
  -p log_detection_min_interval_sec:=1.0 \
  -p log_best_detection_only:=false \
  -p head_ingestion_enabled:=false \
  -p head_http_url:=192.168.31.71:8776
```

检测行默认 **最快约 1 秒一条**（`log_detection_min_interval_sec`）。默认 **同一帧内输出所有匹配目标**（`log_best_detection_only` 默认 `false`）；设为 `true` 则只打置信度最高的一条。`log_detection_min_interval_sec:=0` 可关闭检测行节流。

`head_ingestion_enabled` 默认 `false`；设为 `true` 后，`head_http_url` 可写 `192.168.31.71:8776`，节点会按 `camera2head.md` 自动发送到 `http://192.168.31.71:8776/api/detection`。

日志示例字段：`cam_xyz_m=`（相机系）、`base_xyz_m=`（基座系，米）；若尚无末端矩阵则为 `base_xyz_m=pending_ws`。
