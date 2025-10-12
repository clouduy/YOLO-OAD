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

- **Python 3.10+**
- **PyTorch 2.0+**
- **CUDA 11.0+** (for GPU acceleration)
- **NVIDIA GPU with ≥8GB VRAM** recommended

### 🚨 Important Environment Setup Notes

#### Step 1: Install Anaconda (Recommended)
We strongly recommend using **Anaconda** to manage your Python environment. This prevents package conflicts and makes dependency management much easier.

Download Anaconda from: https://www.anaconda.com/download

#### Step 2: Check Your CUDA Version
**CRITICAL**: The CUDA version in your virtual environment MUST match the CUDA Toolkit version installed on your operating system.

**Check your system CUDA version:**
```bash
nvcc --version
```

If you don't have CUDA installed, download from:
- Official: https://developer.nvidia.com/cuda-downloads
- **For users in China**: Use domestic mirrors for faster download

**Author's Environment**: CUDA 11.8 + PyTorch 2.0 + Python 3.10

#### Step 3: Create Conda Environment
```bash
# Create a new environment
conda create -n yoload python=3.10

# Activate the environment
conda activate yoload
```

#### Step 4: Install PyTorch with Correct CUDA Version
```bash
# For CUDA 11.8 (author's version)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# For users in China, use Tsinghua mirror:
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Install Dependencies

```bash
# Install base requirements (using domestic mirror for Chinese users)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# If you encounter network issues, try:
pip install -r requirements.txt -i https://pypi.douban.com/simple/

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

**Important Note for BDD100K Dataset**: Please remove the 'train' class from the BDD100K dataset before use, as it is not a standard object category for autonomous driving detection tasks.

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
  # ... other classes (excluding 'train' class)
```

### Training

#### Basic Training
```bash
python train.py \
  --data data/BDD100K.yaml \
  --cfg models/YOLO-OAD.yaml \
  --weights YOLO-OAD pre-trained.pt \
  --batch-size 16 \
  --epochs 300 \
  --img-size 640
```

#### Advanced Training with Custom Settings
```bash
python train.py \
  --data data/BDD100K.yaml \
  --cfg models/YOLO-OAD.yaml \
  --weights YOLO-OAD pre-trained.pt \
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
| YOLO-OAD | VOC | 81.8 | 59.4 | 2.62M | 126 | [Link](http://host.robots.ox.ac.uk/pascal/VOC/) |
| YOLO-OAD | BDD100K | 52.5 | 28.3 | 2.62M | 126 | [Link](https://bair.berkeley.edu/blog/2018/05/30/bdd/) |

## ❓ Frequently Asked Questions (FAQ)

### 🐢 **Why is training so slow on CPU?**
**Answer**: Deep learning models like YOLO-OAD require massive parallel computations that GPUs are specifically designed for. 
- CPU training can be **10-50x slower** than GPU training
- **Solution**: Use an NVIDIA GPU with CUDA support for reasonable training times

### 🌐 **Why can't I download packages?**
**Answer**: This is usually a network issue, especially for users in China.
- **Solution 1**: Use domestic mirrors:
  ```bash
  # Tsinghua mirror
  pip install [package] -i https://pypi.tuna.tsinghua.edu.cn/simple
  
  # Douban mirror  
  pip install [package] -i https://pypi.douban.com/simple/
  ```
- **Solution 2**: Set permanent mirror source:
  ```bash
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```

### 📁 **Why can't I open the dataset link?**
**Answer**: Dataset servers might be temporarily unavailable or blocked in some regions.
- **Solution 1**: Try using a VPN
- **Solution 2**: Search for alternative download links or mirrors
- **Solution 3**: Contact the dataset providers directly

### 🔧 **CUDA version mismatch error**
**Answer**: This happens when PyTorch CUDA version doesn't match your system CUDA.
- **Check system CUDA**: `nvcc --version`
- **Check PyTorch CUDA**: 
  ```python
  import torch
  print(torch.version.cuda)
  ```
- **Solution**: Reinstall PyTorch with correct CUDA version from https://pytorch.org

### 💾 **Out of memory error**
**Answer**: Your GPU doesn't have enough VRAM for the batch size.
- **Solution**: Reduce batch size (--batch-size 8 or lower)
- Close other GPU applications during training
- Use smaller image size (--img-size 416)

### 📊 **Training loss is NaN**
**Answer**: This indicates numerical instability.
- **Solution**: Reduce learning rate (--lr 0.001)
- Check your dataset for corrupted images or labels
- Ensure your data normalization is correct

### 🔍 **Model not detecting anything**
**Answer**: Common issues with new datasets.
- **Solution**: Check your class names match between dataset and configuration
- Verify your labels are in correct YOLO format
- Start with pre-trained weights, not from scratch

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
