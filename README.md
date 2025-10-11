# YOLO-OAD: A Deformable Lightweight Asymmetric Decoupled Head based Object Detector for Autonomous Driving

## Project Overview

YOLO-OAD is an optimized object detection model based on YOLOv5, specifically designed for autonomous driving scenarios. It introduces several novel modules including DLADH (Deformable Lightweight Asymmetric Decoupled Head), C3LKSCAA, and C3VGGAM to enhance detection accuracy for multi-scale and occluded objects while maintaining real-time inference capabilities. The model achieves state-of-the-art performance on autonomous driving datasets while maintaining efficient computational characteristics.

## Key Features

- **High Precision Detection**: Significant improvements in detecting multi-scale objects, particularly small objects like traffic signs and distant vehicles
- **Real-Time Performance**: Optimized architecture ensures 126 FPS inference speed on standard GPU hardware
- **Enhanced Adaptability**: Improved robustness to complex driving environments including occlusion, varying lighting conditions, and scale variations
- **Efficient Deployment**: Supports deployment on multiple platforms (GPU/CPU) and frameworks (PyTorch, ONNX, TensorRT)
- **Lightweight Design**: Maintains low parameter count (2.62M) and computational requirements (6.8 GFLOPs)

## Model Architecture

YOLO-OAD integrates the following key innovations:

- **DLADH**: Deformable Lightweight Asymmetric Decoupled Head for task-specific feature decoupling and dynamic spatial adaptation
- **C3LKSCAA**: Enhances receptive field and captures long-range context dependencies using star operations and context anchor attention
- **C3VGGAM**: Improves local feature extraction and preserves salient semantic information using VGG-style blocks with parameter-free attention
- **Focal-EIoU Loss**: Optimizes bounding box regression with better handling of aspect ratios and sample difficulty balancing

## Performance

- **PASCAL VOC**: mAP@0.5: 81.8% (+5.4% over YOLOv5n), mAP@0.5-0.95: 59.4% (+8.8%)
- **BDD100K**: mAP@0.5: 52.5% (+5.3%), notable gains for motor (+9.4%), person (+6.5%), and rider (+5.3%)
- **Speed**: 126 FPS on NVIDIA RTX 2080Ti, ensuring real-time capability for autonomous driving
- **Efficiency**: 2.62M parameters, 6.8 GFLOPs - suitable for embedded deployment

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU acceleration)
- NVIDIA GPU with ≥8GB VRAM recommended

### Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install DCNv4 operator
cd DCNv4_op
python setup.py build install
cd ..
```

### Verification

```bash
python verify_installation.py
```

## Usage

### Data Preparation

Organize your dataset in YOLO format:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Create `data.yaml` configuration:

```yaml
# Dataset configuration
path: /path/to/dataset
train: images/train
val: images/val

# Class names
names:
  0: car
  1: person
  2: traffic_light
  3: traffic_sign
  # ... other classes
```

### Training

#### Basic Training
```bash
python train.py \
  --data data/BDD100K.yaml \
  --cfg models/YOLO-OAD.yaml \
  --weights yolov5n.pt \
  --batch-size 16 \
  --epochs 300 \
  --img-size 640
```

#### Advanced Training with Custom Settings
```bash
python train.py \
  --data data/BDD100K.yaml \
  --cfg models/YOLO-OAD.yaml \
  --weights yolov5n.pt \
  --batch-size 32 \
  --epochs 300 \
  --img-size 640 \
  --optimizer SGD \
  --lr 0.01 \
  --cos-lr \
  --patience 50 \
  --save-period 10 \
  --device 0 \
  --workers 8 \
  --project runs/train \
  --name yoload_exp1
```

### Evaluation

```bash
python val.py \
  --data data/BDD100K.yaml \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --batch-size 32 \
  --task val \
  --verbose
```

### Inference

#### Image Detection
```bash
python detect.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --source data/images/ \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --save-txt \
  --save-conf
```

#### Video Detection
```bash
python detect.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --source data/videos/driving.mp4 \
  --conf-thres 0.3 \
  --save-vid
```

#### Real-time Webcam
```bash
python detect.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --source 0 \
  --conf-thres 0.35
```

### Export for Deployment

#### Export to ONNX
```bash
python export.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --include onnx \
  --dynamic \
  --simplify
```

#### Export to TensorRT
```bash
python export.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --include engine \
  --device 0 \
  --half
```

## Model Zoo

| Model | Dataset | mAP@0.5 | mAP@0.5-0.95 | Params | FPS | Download |
|-------|---------|---------|--------------|--------|-----|----------|
| YOLO-OAD | VOC | 81.8 | 59.4 | 2.62M | 126 | [Link]() |
| YOLO-OAD | BDD100K | 52.5 | 28.3 | 2.62M | 126 | [Link]() |

## Citation

If you use YOLO-OAD in your research, please cite:

```bibtex
@article{chen2024yoload,
  title={A Deformable Lightweight Asymmetric Decoupled Head based Object Detector for Autonomous Driving},
  author={Chen, Yusheng and Xiao, Guangdi and Miao, Linghui and Zhang, Can and Lv, Yong and Chi, Wenzheng and Sun, Lining},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is for academic and research use only. For commercial use, please contact the authors.

---

# YOLO-OAD：基于可变形轻量非对称解耦头的自动驾驶目标检测器

## 项目概述

YOLO-OAD 是基于 YOLOv5 优化的目标检测模型，专为自动驾驶场景设计。该模型引入了多个创新模块，包括 DLADH（可变形轻量非对称解耦头）、C3LKSCAA 和 C3VGGAM，在保持实时推理能力的同时，显著提升了多尺度和遮挡目标的检测精度。该模型在自动驾驶数据集上实现了最先进的性能，同时保持了高效的计算特性。

## 主要特性

- **高精度检测**：在多尺度目标检测方面显著提升，特别是交通标志和远处车辆等小目标
- **实时性能**：优化后的架构在标准 GPU 硬件上确保 126 FPS 的推理速度
- **增强适应性**：对复杂驾驶环境（包括遮挡、不同光照条件和尺度变化）具有更强的鲁棒性
- **高效部署**：支持多平台（GPU/CPU）和多框架（PyTorch、ONNX、TensorRT）部署
- **轻量设计**：保持低参数量（2.62M）和低计算需求（6.8 GFLOPs）

## 模型架构

YOLO-OAD 集成了以下核心创新：

- **DLADH**：可变形轻量非对称解耦头，实现任务特异性特征解耦和动态空间适应
- **C3LKSCAA**：通过星形操作和上下文锚点注意力扩大感受野，捕获长程上下文依赖
- **C3VGGAM**：使用 VGG 风格块和无参数注意力机制增强局部特征提取能力，保留显著语义信息
- **Focal-EIoU 损失函数**：优化边界框回归，更好地处理宽高比和样本难度平衡

## 性能表现

- **PASCAL VOC**：mAP@0.5：81.8%（比 YOLOv5n 提升 5.4%），mAP@0.5-0.95：59.4%（提升 8.8%）
- **BDD100K**：mAP@0.5：52.5%（提升 5.3%），在摩托车（+9.4%）、行人（+6.5%）和骑行者（+5.3%）等类别上提升显著
- **速度**：在 NVIDIA RTX 2080Ti 上达到 126 FPS，确保自动驾驶的实时性需求
- **效率**：2.62M 参数，6.8 GFLOPs - 适合嵌入式部署

## 安装指南

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.0+（用于 GPU 加速）
- 推荐使用 ≥8GB 显存的 NVIDIA GPU

### 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 DCNv4 算子
cd DCNv4_op
python setup.py build install
cd ..
```

### 验证安装

```bash
python verify_installation.py
```

## 使用说明

### 数据准备

按 YOLO 格式组织数据集：

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

创建 `data.yaml` 配置文件：

```yaml
# 数据集配置
path: /path/to/dataset
train: images/train
val: images/val

# 类别名称
names:
  0: car
  1: person
  2: traffic_light
  3: traffic_sign
  # ... 其他类别
```

### 模型训练

#### 基础训练
```bash
python train.py \
  --data data/BDD100K.yaml \
  --cfg models/YOLO-OAD.yaml \
  --weights yolov5n.pt \
  --batch-size 16 \
  --epochs 300 \
  --img-size 640
```

#### 高级训练配置
```bash
python train.py \
  --data data/BDD100K.yaml \
  --cfg models/YOLO-OAD.yaml \
  --weights yolov5n.pt \
  --batch-size 32 \
  --epochs 300 \
  --img-size 640 \
  --optimizer SGD \
  --lr 0.01 \
  --cos-lr \
  --patience 50 \
  --save-period 10 \
  --device 0 \
  --workers 8 \
  --project runs/train \
  --name yoload_exp1
```

### 模型评估

```bash
python val.py \
  --data data/BDD100K.yaml \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --batch-size 32 \
  --task val \
  --verbose
```

### 推理检测

#### 图像检测
```bash
python detect.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --source data/images/ \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --save-txt \
  --save-conf
```

#### 视频检测
```bash
python detect.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --source data/videos/driving.mp4 \
  --conf-thres 0.3 \
  --save-vid
```

#### 实时摄像头检测
```bash
python detect.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --source 0 \
  --conf-thres 0.35
```

### 导出部署

#### 导出为 ONNX
```bash
python export.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --include onnx \
  --dynamic \
  --simplify
```

#### 导出为 TensorRT
```bash
python export.py \
  --weights runs/train/yoload_exp1/weights/best.pt \
  --include engine \
  --device 0 \
  --half
```

## 模型库

| 模型 | 数据集 | mAP@0.5 | mAP@0.5-0.95 | 参数量 | FPS | 下载链接 |
|-------|---------|---------|--------------|--------|-----|----------|
| YOLO-OAD | VOC | 81.8 | 59.4 | 2.62M | 126 | [链接]() |
| YOLO-OAD | BDD100K | 52.5 | 28.3 | 2.62M | 126 | [链接]() |

## 引用方式

如果您在研究中使用了 YOLO-OAD，请引用：

```bibtex
@article{chen2024yoload,
  title={A Deformable Lightweight Asymmetric Decoupled Head based Object Detector for Autonomous Driving},
  author={Chen, Yusheng and Xiao, Guangdi and Miao, Linghui and Zhang, Can and Lv, Yong and Chi, Wenzheng and Sun, Lining},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

本项目仅限于学术和研究使用。商业使用请联系作者。