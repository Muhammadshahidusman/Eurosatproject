# EuroSAT Land Use & Land Cover Classification

A deep learning project for classifying satellite imagery using the EuroSAT RGB dataset, extended to real-world testing on Lahore satellite image patches. This project implements two approaches: a custom Convolutional Neural Network (CNN) built from scratch and a high-performance model using transfer learning with ResNet50.

## Table of Contents

- [Purpose](#purpose)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Lahore Case Study](#lahore-case-study)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Purpose

This project demonstrates the application of deep learning for remote sensing and environmental monitoring. The primary objectives are:

1. **Land Use & Land Cover (LULC) Classification**: Automatically classify satellite images into meaningful land cover categories.
2. **Custom Architecture vs. Transfer Learning**: Compare the performance of a CNN built from scratch against a pretrained industry-standard model (ResNet50).
3. **Model Evaluation**: Analyze performance using comprehensive metrics including confusion matrices and per-class accuracy.
4. **Real-world Application**: Test the trained model on external satellite patches from Lahore to evaluate generalization.

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

After downloading, extract the zip file and place the `Eurosat_Dataset` folder in the project root directory.

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
- **Input Size**: 64 x 64 pixels (resized for EuroSAT).
- **Strategy**: Full fine-tuning of all layers.
- **Goal**: Maximize classification accuracy using deep residual learning.

### Comparison Summary

| Feature | Custom CNN | ResNet50 (Transfer Learning) |
|----------|------------|------------------------------|
| **Input Resolution** | 64 x 64 | 64 x 64 |
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
| **ResNet50** | `models/resnet50/eurosat_resnet50_final.pth` | `models/resnet50/` |

---

## Lahore Case Study

To evaluate the generalization of the ResNet50 model, it was applied to a set of **4,481 satellite image patches** from Lahore.

### Implementation
- **Input**: 64x64 RGB patches.
- **Preprocessing**: ImageNet normalization and resizing.
- **Inference**: The `src/lahore_inference.py` script processes all patches and generates predictions.

### Findings
The model successfully classified the Lahore imagery into EuroSAT categories. Preliminary analysis showed a high frequency of **Pasture** and **Industrial** land covers, reflecting the mixed urban and rural nature of the region.

**Results saved to:** `results/lahore_predictions.png`

---

## Project Structure

```
Eurosatproject/
│
├── notebooks/                    # Jupyter notebooks for training and acquisition
│   ├── eurosat_classification.ipynb # Custom CNN training pipeline
│   ├── eurosat_resnet50.ipynb       # ResNet50 transfer learning pipeline
│   └── patches_64_64.ipynb         # Patch generation
│
├── src/                          # Source code
│   └── lahore_inference.py       # Inference pipeline for Lahore dataset
│
├── data/                         # Datasets
│   └── Lahore_Dataset/patches/    # Lahore satellite image patches
│
├── models/                       # Model checkpoints
│   └── resnet50/
│       └── eurosat_resnet50_final.pth
│
├── results/                      # Visualizations and results
│   └── lahore_predictions.png     # Prediction grid for Lahore
│
├── README.md                     # Project documentation
├── pyproject.toml                # Project dependencies
└── uv.lock                       # Lock file for dependencies
```

---

## Requirements

### Dependencies
This project uses Python 3.10+ and the following packages:
- `torch`, `torchvision`
- `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `Pillow`, `tqdm`
- `rasterio`
- `jupyter`, `ipykernel`

### Hardware
- **GPU**: CUDA-compatible GPU recommended.
- **CPU**: Fallback available if CUDA is not present.

---

## Usage

### Setup
1. Clone the repository and navigate to the project directory.
2. Install dependencies: `uv sync` or `pip install -r requirements.txt`.

### Running the Models
- **For Custom CNN**: Run the notebook `notebooks/eurosat_classification.ipynb`.
- **For ResNet50**: Run the notebook `notebooks/eurosat_resnet50.ipynb`.

### Running Lahore Inference
To run the model on the Lahore dataset:
```bash
uv run python3 src/lahore_inference.py
```

---

## License
This project is developed for educational purposes.

## Acknowledgments
- EuroSAT dataset source: [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://github.com/phelber/EuroSAT)
- Sentinel-2 imagery provided by the European Space Agency (ESA)
