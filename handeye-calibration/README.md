# handeye-calibration

手眼标定工具集（眼在手上配置）。用于计算相机坐标系相对于机械臂末端执行器的旋转矩阵和平移向量。

## 目录结构

```
handeye-calibration/
├── config.yaml                      ← 所有参数在此配置，有详细注释
├── images/                          ← 运行时目录（采集的标定图片 + poses.txt）
├── robotToolPose.csv                ← eyeInHand.py 自动生成的中间文件
└── scr/
    ├── eyeInHand.py                 ← 手眼标定计算（读取图片 + 位姿 → 输出变换矩阵）
    ├── online_capture.py            ← 在线采集（ROS2 图像 + WebSocket 位姿同步采集）
    ├── remove_image_samples.py      ← 删除指定编号的标定样本并重新编号
    └── 手眼标定使用说明.docx
```

## 快速开始

### 1. 配置参数

编辑 `config.yaml`，重点修改：

```yaml
calibration:
  board_corners_long: 11      # 棋盘格长边内角点数
  board_corners_short: 8      # 棋盘格短边内角点数
  board_square_size_m: 0.015  # 单格尺寸（米）

capture:
  ws_host: "192.168.31.142"   # 机械臂 IP
  ws_port: 8765
  rgb_compressed_topic: "/ascamera_hp60c/..."
```

### 2. 在线采集标定数据

```bash
# 启动 ROS2 相机节点（确保相机已连接），然后运行采集脚本：
source /opt/ros/humble/setup.bash
python3 scr/online_capture.py

# 键位：c / 空格 = 采集，q / ESC = 退出
# 可选参数（覆盖 config.yaml）：
python3 scr/online_capture.py --ws-host 192.168.x.x --ws-port 8765
```

### 3. 计算手眼外参

```bash
python3 scr/eyeInHand.py
# 输出 TSAI / PARK / HORAUD 三种方法的旋转矩阵和平移向量
# 将 PARK 方法结果填入 robot-vision-ros2/config.yaml 的 hand_eye.T_cam2ee
```

### 4. 删除质量差的样本（可选）

```bash
python3 scr/remove_image_samples.py --remove 0,2,5
# 自动重新编号剩余样本
```

## 参数说明

所有参数均在 `config.yaml` 中配置，每个参数都有注释说明其含义和原始来源。
CLI 参数优先级高于 `config.yaml`。
