# yolo-pose-pipeline

YOLO 姿态检测训练流水线：数据准备 → 模型训练 → 实时推理。

## 目录结构

```
yolo-pose-pipeline/
├── config.yaml              ← 所有参数在此配置，有详细注释
├── extract_frames.py        ← 从视频批量抽帧
├── data_video/              ← 放入原始视频（不入库）
├── data_video_frames/       ← 抽帧输出目录（自动生成）
└── pose/
    ├── prepare_data.py      ← LabelMe JSON → YOLO txt + train/val/test 分割
    │                           (--mode split-only 可跳过 JSON 转换)
    ├── train.py             ← 训练 YOLO pose 模型
    ├── realtime_detect_pose.py  ← 摄像头实时推理
    ├── dataset/             ← 原始标注数据（images/ + labels/）
    ├── yolo_dataset/        ← 生成的 YOLO 格式数据集（可重建，不入库）
    ├── data.yaml            ← YOLO 数据集配置（指向 yolo_dataset/）
    └── runs/                ← 训练输出（权重文件）
```

## 快速开始

### 0. 安装依赖

```bash
pip install -r pose/requirements.txt
```

### 1. 配置参数

编辑 `config.yaml`，重点修改：

```yaml
prepare_data:
  src_dir: "pose/dataset"          # 原始标注数据目录
  pose_classes_file: "pose/dataset/pose_classes.yaml"
  train_ratio: 0.8

train:
  epochs: 100
  batch: 4
  run_name: "my_experiment"
```

### 2. 从视频抽帧（可选）

```bash
# 将视频放入 data_video/，然后：
python3 extract_frames.py
# 输出到 data_video_frames/，每个视频一个子目录
```

### 3. 准备数据集

```bash
# 模式1：LabelMe JSON → YOLO txt + 分割（默认）
python3 pose/prepare_data.py

# 模式2：已有 YOLO txt，只做 train/val/test 分割
python3 pose/prepare_data.py --mode split-only --src pose/dataset417 --dst pose/yolo_dataset417

# 可选 CLI 覆盖：
python3 pose/prepare_data.py --src pose/my_dataset --dst pose/yolo_my --train 0.85 --val 0.1
```

### 4. 训练

```bash
python3 pose/train.py
# 输出到 pose/runs/<run_name>/weights/best.pt
```

### 5. 实时推理

```bash
python3 pose/realtime_detect_pose.py
# 使用 config.yaml 中的 inference.weights_path 作为默认权重
# 或指定：
python3 pose/realtime_detect_pose.py --weights pose/runs/my_experiment/weights/best.pt
```

## 参数说明

所有参数均在 `config.yaml` 中配置，每个参数都有注释说明其含义和原始来源。
CLI 参数始终优先于 `config.yaml`。
