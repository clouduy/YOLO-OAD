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
