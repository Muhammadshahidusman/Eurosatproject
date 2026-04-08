# EuroSAT Land Use & Land Cover Classification

A deep learning project for classifying satellite imagery using the EuroSAT RGB dataset. This project implements two approaches: a custom Convolutional Neural Network (CNN) built from scratch and a high-performance model using transfer learning with ResNet50.

## Table of Contents

- [Purpose](#purpose)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Purpose

This project demonstrates the application of deep learning for remote sensing and environmental monitoring. The primary objectives are:

1. **Land Use & Land Cover (LULC) Classification**: Automatically classify satellite images into meaningful land cover categories.
2. **Custom Architecture vs. Transfer Learning**: Compare the performance of a CNN built from scratch against a pretrained industry-standard model (ResNet50).
3. **Model Evaluation**: Analyze performance using comprehensive metrics including confusion matrices and per-class accuracy.

Applications of this technology include:
- Urban planning and monitoring
- Agricultural land assessment
- Environmental change detection
- Natural resource management
- Disaster response and recovery

---

## Dataset

### EuroSAT RGB

The EuroSAT dataset is based on Sentinel-2 satellite imagery containing 27,000 labeled images covering 10 different land cover classes. All images are 64x64 pixels in RGB format.

### 10 Land Cover Classes

| Class | Description |
|-------|-------------|
| **AnnualCrop** | Temporary agricultural fields with annual crops |
| **Forest** | Dense forest areas |
| **HerbaceousVegetation** | Grasslands and meadows |
| **Highway** | Major road networks |
| **Industrial** | Industrial zones and commercial areas |
| **Pasture** | Grazing lands and pastures |
| **PermanentCrop** | Orchards and vineyards |
| **Residential** | Residential and urban areas |
| **River** | Water bodies (rivers and streams) |
| **SeaLake** | Large water bodies (seas and lakes) |

### Dataset Statistics

- **Total Images**: 27,000
- **Image Resolution**: 64 x 64 pixels
- **Color Channels**: 3 (RGB)
- **Train/Val/Test Split**: 70% / 15% / 15%
  - Train: 18,900 images
  - Validation: 4,050 images
  - Test: 4,050 images

### Download Dataset

The dataset is not included in this repository due to its large size (~2GB). Download it from:

**[Google Drive - Eurosat_Dataset.zip](https://drive.google.com/file/d/1cvBQILj_CpiR7KMM2xsWYVU5c1vdv2Pq/view?usp=drive_link)**

After downloading, extract the zip file and place the `Eurosat_Dataset` folder in the project root directory:

```
Eurosatproject/
├── Eurosat_Dataset/          # <-- Extracted here
│   ├── AnnualCrop/
│   ├── Forest/
│   ├── HerbaceousVegetation/
│   └── ... (10 classes)
```

---

## Methodology

The project implements two distinct modeling strategies:

### 1. Custom CNN (Baseline)
A custom 3-block architecture designed to learn features from scratch.
- **Architecture**: `Conv → BatchNorm → ReLU` (x2 per block) $\rightarrow$ `MaxPool` $\rightarrow$ `Dropout`.
- **Input Size**: 64 x 64 pixels.
- **Complexity**: ~1.5M trainable parameters.
- **Goal**: Establish a performance baseline for the dataset.

### 2. Transfer Learning (ResNet50)
Leverages a pretrained ResNet50 model trained on ImageNet for superior feature extraction.
- **Architecture**: ResNet50 backbone with a replaced final fully connected layer.
- **Input Size**: 224 x 224 pixels (ResNet standard).
- **Strategy**: Full fine-tuning of all layers.
- **Goal**: Maximize classification accuracy using deep residual learning.

### Comparison Summary

| Feature | Custom CNN | ResNet50 (Transfer Learning) |
|----------|------------|------------------------------|
| **Input Resolution** | 64 x 64 | 224 x 224 |
| **Pretrained Weights** | None (From Scratch) | ImageNet Weights |
| **Parameters** | ~1.5 Million | ~25 Million |
| **Expected Accuracy** | ~90% | ~97-98% |

### Training Configuration (Common)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: StepLR (halves LR periodically)
- **Augmentation**: Random flips, rotations, and color jittering.

---

## Results

The models were trained for 20 epochs, yielding the following outcomes:

### Performance Comparison
- **Custom CNN**: Provides a strong baseline with high accuracy across most classes.
- **ResNet50**: Significant improvement in accuracy and robustness, particularly in distinguishing similar land covers.

### Evaluation Artifacts
The project generates the following visualizations for both models:

1. **Training Curves** (`training_curves.png`): Loss and accuracy progression.
2. **Confusion Matrix** (`confusion_matrix.png`): Visualizing misclassifications.
3. **Per-Class Accuracy** (`per_class_accuracy.png`): Breakdown of performance for each of the 10 classes.

### Output Files

| Model | Checkpoint | Results Directory |
|-------|------------|-------------------|
| **Custom CNN** | `eurosat_cnn_final.pth` | Project Root |
| **ResNet50** | `ResNet 50/eurosat_resnet50_final.pth` | `ResNet 50/` |

---

## Project Structure

```
Eurosatproject/
│
├── eurosat_classification.ipynb   # Custom CNN training pipeline
├── eurosat_resnet50.ipynb        # ResNet50 transfer learning pipeline
├── README.md                       # Project documentation
├── pyproject.toml                  # Project dependencies
├── main.py                         # Entry point script
│
├── Eurosat_Dataset/                # Directory containing 27,000 images
│
├── ResNet 50/                      # Results and checkpoints for ResNet50
│   ├── eurosat_resnet50_final.pth
│   ├── best_model.pth
│   └── ... (visualizations)
│
├── eurosat_cnn_final.pth           # Custom CNN saved model checkpoint
├── best_model.pth                  # Custom CNN best weights
│
└── outputs/                        # Generated visualizations (Custom CNN)
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── per_class_accuracy.png
    ├── eda_samples.png
    └── eda_distribution.png
```

---

## Requirements

### Dependencies
This project uses Python 3.10+ and the following packages:
- `torch`, `torchvision`
- `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `Pillow`, `tqdm`
- `jupyter`, `ipykernel`

### Hardware
- **GPU**: CUDA-compatible GPU recommended (e.g., NVIDIA Quadro P2000).
- **CPU**: Fallback available if CUDA is not present.

---

## Usage

### Setup
1. Clone the repository and navigate to the project directory.
2. Install dependencies: `uv sync` or `pip install -r requirements.txt`.

### Running the Models
- **For Custom CNN**: Run `jupyter notebook eurosat_classification.ipynb`.
- **For ResNet50**: Run `jupyter notebook eurosat_resnet50.ipynb`.
  - *Note: For ResNet50, ensure you have the pretrained weights available or have an active internet connection for the initial download.*

### Using a Trained Model
To load a checkpoint and make predictions:
```python
import torch
from torchvision import transforms

# Load ResNet50 checkpoint
checkpoint = torch.load('ResNet 50/eurosat_resnet50_final.pth')
# (Recreate model architecture as defined in the notebook)
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

---

## License
This project is developed for educational purposes.

## Acknowledgments
- EuroSAT dataset source: [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://github.com/phelber/EuroSAT)
- Sentinel-2 imagery provided by the European Space Agency (ESA)
