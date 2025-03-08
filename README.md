YOLO-OAD

Project Overview

This project enhances and optimizes the YOLOv5 object detection model to boost its performance and generalization in specific scenarios. The improved model maintains YOLOv5's fast detection capability while increasing accuracy and adaptability to complex situations.
Functional Features
Object Detection: Accurately detects various objects and provides their locations and categories.
Real-time Detection: Optimized model structure ensures fast inference while maintaining high accuracy, ideal for real-time detection scenarios.
Multi-scene Adaptation: Enhanced data augmentation and training methods improve the model's adaptability to different lighting, angles, and occlusions.
Flexible Deployment: Supports deployment on multiple hardware platforms (GPUs, CPUs) and frameworks (PyTorch, ONNX).
Installation Guide

Environmental Preparation

Ensure your system meets these requirements:
Python 3.10 or higher
PyTorch 2.0 or higher
CUDA 11 or higher (for GPU usage)

Dependency Installation

Install required dependencies using pip:
(1)pip install -r requirements.txt
(2)Navigate to the DCNv4_op directory and run python setup.py build install
Usage Method

Data Preparation
Organize your training dataset in YOLO format, including:
Image files (e.g., .jpg, .png)
Corresponding label files (.txt, with category and bounding box info per line)
Dataset configuration file (data.yaml), defining categories, and paths to training/validation sets

Training Process:
python train.py --data BDD100K.yaml --cfg models/YOLO-OAD.yaml --weights yolov5n.pt --batch-size 16 --epochs 300