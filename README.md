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
conda create -n yolo-oad python=3.10

# Activate the environment
conda activate yolo-oad
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

### 🎯 BDD100K Dataset Preprocessing Guide

**Important Note**: The BDD100K dataset contains a 'train' class that is not suitable for autonomous driving object detection tasks. Follow these steps to properly preprocess the dataset:

#### Step 1: Analyze Dataset Classes
First, identify which classes are present in your dataset:

```bash
python classify.py
```

This will output the class indices and their occurrence statistics in your dataset.

#### Step 2: Convert BDD100K Format to YOLO Format
Use the provided conversion script:

```bash
python bdd100k.py
```

**Configuration**: Update the paths in `bdd100k.py`:
- `readpath`: Path to original BDD100K labels
- `writepath`: Output path for converted YOLO format labels
- `categorys`: Target classes to extract (exclude 'train')

#### Step 3: Remove 'train' Class from Dataset
Execute the removal script:

```bash
python quchu.py
```

**Configuration**: Modify the paths in `quchu.py`:
- `input_folder`: Converted labels folder
- `output_folder`: Cleaned labels folder  
- `target_class`: 9 (class index for 'train')

#### Complete Preprocessing Pipeline
```python
# 1. First run classify.py to identify class distribution
# 2. Then run bdd100k.py to convert format  
# 3. Finally run quchu.py to remove train class
# The processed dataset is now ready for training
```

Create `data.yaml` configuration:

```yaml
# Dataset configuration
path: /path/to/dataset
train: images/train
val: images/val

# Class names (excluding 'train')
names:
  0: car
  1: person
  2: traffic_light
  3: traffic_sign
  4: rider
  5: bus
  6: truck
  7: bike
  8: motor
  # Note: class 9 (train) has been removed
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
  --name yolo-oad_exp1
```

### Evaluation

```bash
python val.py \
  --data data/BDD100K.yaml \
  --weights runs/train/yolo-oad_exp1/weights/best.pt \
  --batch-size 32 \
  --task val \
  --verbose
```

### Inference

#### Image Detection
```bash
python detect.py \
  --weights runs/train/yolo-oad_exp1/weights/best.pt \
  --source data/images/ \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --save-txt \
  --save-conf
```

#### Video Detection
```bash
python detect.py \
  --weights runs/train/yolo-oad_exp1/weights/best.pt \
  --source data/videos/driving.mp4 \
  --conf-thres 0.3 \
  --save-vid
```

#### Real-time Webcam
```bash
python detect.py \
  --weights runs/train/yolo-oad_exp1/weights/best.pt \
  --source 0 \
  --conf-thres 0.35
```

### Export for Deployment

#### Export to ONNX
```bash
python export.py \
  --weights runs/train/yolo-oad_exp1/weights/best.pt \
  --include onnx \
  --dynamic \
  --simplify
```

#### Export to TensorRT
```bash
python export.py \
  --weights runs/train/yolo-oad_exp1/weights/best.pt \
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

### 🔧 **Dataset Preprocessing Questions**

#### **Q: Why do I need to remove the 'train' class from BDD100K dataset?**
**A**: The 'train' class is removed for two specific technical reasons:
1. **Severe Class Imbalance**: The train class has significantly fewer instances compared to other vehicle classes, creating training instability and poor convergence
2. **Autonomous Driving Relevance**: Railway trains appear in specialized contexts not relevant to standard road driving scenarios

#### **Q: How do I verify the 'train' class has been successfully removed?**
**A**: Run this verification command after preprocessing:
```bash
python classify.py
```
Check the output to ensure class index 9 (train) no longer appears in the statistics. The script should only show classes 0-8.

#### **Q: What's the exact step-by-step process for BDD100K preprocessing?**
**A**: Execute these commands in order:
```bash
# Step 1: Convert BDD100K format to YOLO format
python bdd100k.py

# Step 2: Remove train class (class index 9)
python quchu.py

# Step 3: Verify processing completed successfully
python classify.py
```

#### **Q: How do I fix "FileNotFoundError" during dataset preprocessing?**
**A**: This means the script can't find your dataset files. Check these exact paths:
```python
# In bdd100k.py - verify these paths match your system
readpath = "/home/lvyong/BDD100K/bdd100k/labels/100k/val/"
writepath = "/home/lvyong/BDD100K/bdd100k/labels/100k/val_txt/"

# In quchu.py - verify these paths
input_folder = '/home/lvyong/bdd100k/labels/100k/val_txt'
output_folder = '/home/lvyong/bdd100k/labels/100k/val_new'
```

### 🌐 **Environment Setup Issues**

#### **Q: How do I check if my CUDA installation is correct?**
**A**: Run these exact commands to verify:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA compiler
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA version:', torch.version.cuda)"
```

#### **Q: PyTorch installation fails with "Could not find a version" - what do I do?**
**A**: Use the exact installation command for your CUDA version:
```bash
# For CUDA 11.8 (most common)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

#### **Q: DCNv4 compilation fails - how to fix?**
**A**: Execute these commands in order:
```bash
# Install build dependencies
sudo apt update
sudo apt install build-essential cmake

# Navigate to DCNv4 directory and compile
cd DCNv4_op
python setup.py build develop
cd ..
```

#### **Q: "ImportError: No module named 'cv2'" - how to install OpenCV?**
**A**: Install with this exact command:
```bash
pip install opencv-python-headless
```

### 🖥️ **Hardware and Performance Issues**

#### **Q: What are the minimum GPU requirements?**
**A**: 
- **Absolute Minimum**: NVIDIA GPU with 6GB VRAM (GTX 2060)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3060/4060)
- **Training BDD100K**: 8GB VRAM allows batch size 16 with img-size 640

#### **Q: How do I fix "CUDA out of memory" errors?**
**A**: Use these exact parameter combinations:
```bash
# For 6GB VRAM cards
python train.py --batch-size 8 --img-size 416

# For 8GB+ VRAM cards
python train.py --batch-size 16 --img-size 640
```

#### **Q: Training is too slow - how to speed it up?**
**A**: Apply these optimizations:
```bash
# Enable mixed precision training (2x speedup)
python train.py --half

# Use multiple GPU workers for data loading
python train.py --workers 4

# Pin memory for faster data transfer
python train.py --workers 4 --pin-memory
```

### 📊 **Training Configuration**

#### **Q: What's the recommended training command for BDD100K?**
**A**: Use this exact command for BDD100K dataset:
```bash
python train.py \
  --data data/BDD100K.yaml \
  --cfg models/YOLO-OAD.yaml \
  --weights YOLO-OAD pretrained.pt \
  --batch-size 16 \
  --epochs 300 \
  --img-size 640 \
  --workers 4 \
  --patience 50
```

#### **Q: How do I monitor training progress?**
**A**: Use TensorBoard with this command:
```bash
tensorboard --logdir runs/train
```
Then open http://localhost:6006 in your browser.

#### **Q: Training loss shows NaN - what's the fix?**
**A**: Apply these solutions in order:
```bash
# Solution 1: Reduce learning rate
python train.py --lr 0.001

# Solution 2: Check dataset for corrupted images
python -c "from utils.datasets import LoadImages; ds = LoadImages('path/to/images'); print('Dataset check passed')"

# Solution 3: Verify label files
python -c "from utils.general import check_dataset; check_dataset('data/BDD100K.yaml')"
```

### 🔍 **Model Validation and Testing**

#### **Q: How do I test my trained model?**
**A**: Use this validation command:
```bash
python val.py \
  --data data/BDD100K.yaml \
  --weights runs/train/exp/weights/best.pt \
  --batch-size 32 \
  --task test \
  --save-json
```

#### **Q: How to run inference on custom images?**
**A**: Use this detect command:
```bash
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --source path/to/images/ \
  --conf-thres 0.25 \
  --save-txt \
  --save-conf
```

#### **Q: Model performs poorly - how to debug?**
**A**: Follow this debugging checklist:
1. Verify dataset preprocessing completed successfully
2. Check class distribution with `python classify.py`
3. Validate labels with visualization tools
4. Ensure correct class mappings in `data.yaml`
5. Confirm you're using pretrained weights

### 🐛 **Common Error Solutions**

#### **Q: "RuntimeError: Expected all tensors to be on the same device"**
**A**: Fix with device specification:
```python
# In your code, explicitly set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

#### **Q: "AttributeError: module 'numpy' has no attribute 'int'"`
**A**: This is a NumPy compatibility issue:
```bash
pip install numpy==1.23.5
```

#### **Q: "OSError: [Errno 28] No space left on device"**
**A**: Clear temporary files and check storage:
```bash
# Clear PyTorch cache
rm -rf ~/.cache/torch

# Check disk space
df -h

# Use different output directory
python train.py --project /path/to/large/disk/training
```

## Citation

If you use YOLO-OAD in your research, please cite:

```bibtex
@article{chen2024yolo-oad,
  title={A Deformable Lightweight Asymmetric Decoupled Head based Object Detector for Autonomous Driving},
  author={Chen, Yusheng and Xiao, Guangdi and Miao, Linghui and Zhang, Can and Lv, Yong and Chi, Wenzheng and Sun, Lining},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is for academic and research use only. For commercial use, please contact the authors.
