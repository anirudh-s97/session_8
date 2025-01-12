# CIFAR-10 CNN Classifier

A PyTorch implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The model uses a custom architecture with dilated convolutions and depth-wise separable convolutions to learn hierarchical features effectively.

## Features

- Custom CNN architecture with four specialized convolutional blocks
- Advanced data augmentation using Albumentations library
- Learning rate scheduling with ReduceLROnPlateau
- Comprehensive training metrics visualization
- Efficient data loading with PyTorch DataLoader
- Model checkpointing for best performance

## Project Structure

```
.
├── config.py           # Configuration parameters and hyperparameters
├── data_module.py      # Data loading and augmentation pipeline
├── model.py           # CNN model architecture definition
└── train.py          # Training loop and evaluation code
```

## Requirements

- PyTorch
- torchvision
- albumentations
- matplotlib
- numpy
- tqdm

## Model Architecture

The CNN architecture consists of four specialized convolutional blocks:

1. **Edge and Gradient Detection Block**:
   - Uses standard convolutions and depth-wise separable convolutions
   - Includes batch normalization and dropout

2. **Texture and Pattern Recognition Block**:
   - Uses dilated convolutions
   - Multiple convolutional layers with batch normalization

3. **Object Parts Detection Block**:
   - Larger dilation rates for increased receptive field
   - Deep feature extraction with multiple layers

4. **Object Detection Block**:
   - Final feature processing with varying dilation rates
   - Global Average Pooling for dimension reduction

## Data Augmentation

The training pipeline includes several augmentation techniques:
- Horizontal flipping
- Random rotation and scaling
- Coarse dropout (cutout)
- Random brightness and contrast adjustments
- Normalization using CIFAR-10 mean and standard deviation

## Training

To train the model:

```bash
python train.py
```

The training process includes:
- Automatic learning rate adjustment
- Best model checkpointing
- Real-time metrics logging
- Training visualization plots

## Model Performance Tracking

The training script automatically generates:
- Loss curves (training and validation)
- Accuracy plots (training and validation)
- Learning rate schedule visualization
- Saved metrics in NumPy format for further analysis

## Configuration

Key parameters can be modified in `config.py`:

```python
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 55
NUM_WORKERS = 4
```

## Output Directory Structure

```
.
├── data/              # Dataset storage
└── checkpoints/       # Model checkpoints and metrics
    ├── best_model.pth
    ├── metrics.npz
    └── training_metrics.png
```

## Model Checkpointing

The training script saves:
- Best model weights based on validation accuracy
- Optimizer state
- Scheduler state
- Current metrics
- Training progress information

## Future Improvements

Potential areas for enhancement:
- Implementation of additional architectures
- Support for other datasets
- Multi-GPU training support
- TensorBoard integration
- Cross-validation support
