# Deep Learning Week 6 Assignment - SimCLR Implementation

## Overview
This project implements SimCLR (Simple Framework for Contrastive Learning of Visual Representations), a self-supervised learning approach for visual representation learning. The implementation demonstrates contrastive learning on the CIFAR-10 dataset, followed by downstream classification with limited labeled data.

## Project Structure
- `W6 Exercise.py` - Main Python implementation
- `Week 6 Exercise.ipynb` - Jupyter notebook version
- `W6_exercise.ipynb - Colab.pdf` - PDF version from Google Colab

## Key Components

### 1. SimCLR Framework
- **SimCLRDataset**: Custom dataset class that applies data augmentations to create positive pairs
- **Encoder**: Convolutional neural network for feature extraction from images
- **ProjectionHead**: MLP head for contrastive learning (discarded after pre-training)
- **NT-Xent Loss**: Normalized Temperature-scaled Cross-Entropy Loss for contrastive learning

### 2. Data Augmentation Pipeline
Strong augmentations applied to CIFAR-10 images:
- Random resized crop (scale 0.2-1.0)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale conversion
- Gaussian blur

### 3. Downstream Classification
- **FCNClassifier**: Fully connected classifier using frozen pre-trained encoder
- Limited labeled data training (500 samples per class)
- Performance evaluation on CIFAR-10 test set

## Architecture Details

### Encoder Network
- 3 Convolutional layers with ReLU activation
- Kernel size: 3x3, Stride: 2, Padding: 1
- Channels: 3 → 64 → 128 → 256
- Adaptive average pooling
- Fully connected layer: 256 → 128 features
- L2 normalization of output features

### Projection Head
- 2-layer MLP: 128 → 128 → 128
- ReLU activation between layers
- L2 normalization of output

## Requirements
```python
torch
torchvision
torch.nn.functional
```

## Usage

### Training SimCLR Model
```python
# Run self-supervised pre-training
train_simclr()
```
- Trains for 10 epochs on CIFAR-10 with contrastive loss
- Saves encoder weights to `simclr_encoder.pth`

### Downstream Classification
```python
# Load pre-trained encoder and train classifier
train_classifier()
evaluate_classifier()
```
- Uses frozen encoder features
- Trains only the classification head
- Evaluates on CIFAR-10 test set

## Key Hyperparameters
- **Batch size**: 256 (SimCLR), 64 (classifier)
- **Learning rate**: 1e-3
- **Temperature**: 0.5 (NT-Xent loss)
- **Epochs**: 10 (both pre-training and fine-tuning)
- **Limited data**: 500 samples per class

## Device Support
Automatically detects and uses:
1. CUDA (if available)
2. MPS (Apple Silicon)
3. CPU (fallback)

## Expected Results
The model demonstrates the effectiveness of self-supervised learning by achieving competitive classification accuracy using only a fraction of labeled data (500 samples per class vs. full 5000 samples per class in CIFAR-10).

## Files Generated
- `simclr_encoder.pth` - Pre-trained encoder weights
- `./data/` - Downloaded CIFAR-10 dataset

## Course Information
- **Course**: Deep Learning (Masters Program)
- **Assignment**: Week 6 Exercise
- **Topic**: Self-Supervised Learning with SimCLR
- **Dataset**: CIFAR-10

## Implementation Notes
- Originally developed in Google Colab with GPU acceleration
- Includes both Python script and Jupyter notebook versions
- Implements core SimCLR concepts: data augmentation, contrastive learning, and downstream evaluation
- Demonstrates the power of self-supervised pre-training for limited labeled data scenarios