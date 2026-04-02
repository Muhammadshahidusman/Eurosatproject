# EuroSAT Land Use & Land Cover Classification

A deep learning project for classifying satellite imagery using the EuroSAT RGB dataset. This project implements a custom Convolutional Neural Network (CNN) from scratch to identify 10 different types of land cover from Sentinel-2 satellite imagery.

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

1. **Land Use & Land Cover (LULC) Classification**: Automatically classify satellite images into meaningful land cover categories
2. **CNN from Scratch**: Build and train a custom neural network architecture without using pre-trained models
3. **Model Evaluation**: Analyze performance using comprehensive metrics including confusion matrices and per-class accuracy

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

---

## Methodology

### Model Architecture

The project uses a custom 3-block CNN architecture:

```
Input: [3 × 64 × 64]
    |
    ├── Conv Block 1  →  32 filters  →  MaxPool  →  [32 × 32 × 32]
    ├── Conv Block 2  →  64 filters  →  MaxPool  →  [64 × 16 × 16]
    ├── Conv Block 3  → 128 filters  →  MaxPool  →  [128 × 8 × 8]
    |
    └── FC Head  →  Flatten → 512 → 256 → 10 (Softmax)
```

Each convolutional block consists of:
```
Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout2d
```

**Total Parameters**: ~1.5M trainable parameters

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Batch Size** | 64 |
| **Epochs** | 20 |
| **Learning Rate** | 0.001 |
| **Optimizer** | Adam |
| **Weight Decay** | 0.0001 |
| **Scheduler** | StepLR (gamma=0.5, step_size=7) |
| **Loss Function** | CrossEntropyLoss |

### Data Augmentation

Training images are augmented to improve model generalization:
- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.5)
- Random Rotation (±15 degrees)
- Color Jitter (brightness, contrast, saturation)

### Normalization

Images are normalized using dataset-specific statistics:
- **Mean**: [0.3444, 0.3803, 0.4078]
- **Std**: [0.2034, 0.1365, 0.1148]

---

## Results

The model was trained for 20 epochs with the following outcomes:

### Training Performance

- **Best Validation Accuracy**: Achieved during training
- **Test Accuracy**: Evaluated on held-out test set

### Evaluation Metrics

The project produces the following evaluation artifacts:

1. **Training Curves** (`training_curves.png`)
   - Loss progression over epochs
   - Accuracy progression over epochs

2. **Confusion Matrix** (`confusion_matrix.png`)
   - Visual representation of model predictions vs. true labels
   - Helps identify which classes are most confused

3. **Per-Class Accuracy** (`per_class_accuracy.png`)
   - Individual accuracy for each of the 10 land cover classes
   - Color-coded performance thresholds:
     - Green: ≥ 90%
     - Orange: 75–89%
     - Red: < 75%

### Output Files

| File | Description |
|------|-------------|
| `eurosat_cnn_final.pth` | Complete model checkpoint with architecture and hyperparameters |
| `best_model.pth` | Best model state based on validation accuracy |
| `training_curves.png` | Training and validation loss/accuracy plots |
| `confusion_matrix.png` | 10x10 confusion matrix visualization |
| `per_class_accuracy.png` | Horizontal bar chart of per-class accuracies |
| `eda_samples.png` | Sample images from each class |
| `eda_distribution.png` | Class distribution across dataset |

---

## Project Structure

```
Eurosatproject/
│
├── eurosat_classification.ipynb   # Main Jupyter notebook with training pipeline
├── README.md                       # This file
├── pyproject.toml                  # Project dependencies
├── main.py                         # Entry point script
│
├── Eurosat_Dataset/                # Directory containing 27,000 images
│
├── eurosat_cnn_final.pth           # Saved model checkpoint
├── best_model.pth                  # Best model weights
│
└── outputs/                        # Generated visualizations
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── per_class_accuracy.png
    ├── eda_samples.png
    └── eda_distribution.png
```

---

## Requirements

### Dependencies

This project uses Python 3.10 and the following packages:

```
torch==1.13.1+cu117
torchvision==0.14.1+cu117
numpy<2
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=9.0.0
tqdm>=4.60.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

### Hardware

- **GPU**: CUDA-compatible GPU (tested on NVIDIA Quadro P2000 with 4.2 GB VRAM)
- **CPU**: Fallback to CPU if CUDA is not available

---

## Usage

### Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   cd Eurosatproject
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

### Training

Run the Jupyter notebook to train the model:

```bash
jupyter notebook eurosat_classification.ipynb
```

Or use the main script:

```bash
python main.py
```

### Using the Trained Model

To load and use the saved model:

```python
import torch
from torchvision import transforms

# Load checkpoint
checkpoint = torch.load('eurosat_cnn_final.pth')

# Recreate model architecture
from eurosat_classification import EuroSAT_CNN  # Import from notebook
model = EuroSAT_CNN(num_classes=10)
model.load_state_dict(checkpoint['model_state'])

# Make predictions
model.eval()
with torch.no_grad():
    output = model(input_image)
    prediction = output.argmax(1).item()
```

---

## Notes

- The notebook includes an optional section for transfer learning using ResNet50, which could potentially improve accuracy to ~97-98%
- All training uses a fixed random seed (42) for reproducibility
- The model checkpoint includes hyperparameters for easy reproduction of results

---

## License

This project is developed for educational purposes.

## Acknowledgments

- EuroSAT dataset source: [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://github.com/phelber/EuroSAT)
- Sentinel-2 imagery provided by the European Space Agency (ESA)