# yolo-pose-pipeline

YOLO 姿态检测训练流水线：数据准备 → 模型训练 → 实时推理。

## 目录结构

```
yolo-pose-pipeline/
├── config.yaml                  ← 所有参数在此配置，有详细注释
├── data_video/                  ← 放入原始视频（不入库）
├── data_video_frames/           ← 抽帧输出目录（自动生成，不入库）
├── dataset/                     ← 数据集目录，每个子目录为一个数据集
│   └── data516/
│       ├── images/              ← 平铺图片，分割后自动生成 train/val/test 子目录
│       └── labels/              ← 平铺 YOLO txt，分割后自动生成 train/val/test 子目录
├── runs/                        ← 训练输出（权重文件）
└── src/
    ├── extract_frames.py        ← 从视频批量抽帧
    ├── prepare_data.py          ← 原地分割 dataset 为 train/val/test
    ├── train.py                 ← 训练 YOLO pose 模型
    └── realtime_detect_pose.py  ← 摄像头实时推理
```

## 快速开始

### 0. 安装依赖

```bash
pip install -r src/requirements.txt
```

### 1. 配置参数

编辑 `config.yaml`，重点修改：

```yaml
prepare_data:
  src_dir: "dataset/data516"   # 要分割的数据集目录
  train_ratio: 0.8
  val_ratio: 0.1

train:
  epochs: 100
  batch: 4
  run_name: "my_experiment"
```

### 2. 从视频抽帧（可选）

```bash
# 将视频放入 data_video/，然后：
python3 src/extract_frames.py
# 输出到 data_video_frames/，每个视频一个子目录
```

### 3. 原地分割数据集

```bash
# 使用 config.yaml 中 prepare_data.src_dir 指定的数据集
python3 src/prepare_data.py

# 或指定具体数据集目录
python3 src/prepare_data.py --dataset dataset/data516 --train 0.8 --val 0.1
```

分割后 `images/` 和 `labels/` 下的平铺文件会直接移动到对应的 `train/val/test` 子目录，无需额外目录。

### 4. 训练

```bash
python3 src/train.py
# 输出到 runs/<run_name>/weights/best.pt
```

### 5. 实时推理

```bash
python3 src/realtime_detect_pose.py
# 使用 config.yaml 中的 inference.weights_path 作为默认权重
# 或指定：
python3 src/realtime_detect_pose.py --weights runs/my_experiment/weights/best.pt
```

## 参数说明

所有参数均在 `config.yaml` 中配置，每个参数都有注释说明其含义和原始来源。
CLI 参数优先级高于 `config.yaml`。
