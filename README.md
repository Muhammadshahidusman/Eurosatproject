# EuroSAT Land Use & Land Cover Classification

A deep learning project for **Land Use / Land Cover (LULC) classification** of
satellite imagery, built around the EuroSAT RGB dataset and extended to a
real-world Sentinel-2 case study over the **Lahore district (Pakistan)**.

The repository compares two classifiers side-by-side:

| Model | Type | Training input | Purpose |
|-------|------|----------------|---------|
| **Custom CNN** | 3-block VGG-style network, trained from scratch | 64 × 64 RGB | Lightweight baseline |
| **ResNet50**    | `torchvision.models.resnet50` fine-tuned from ImageNet | 224 × 224 RGB | State-of-the-art transfer learning |

Beyond classification, the project delivers a complete **analysis pipeline**:
efficiency profiling, Grad-CAM explainability, and a four-bin district-scale
land-cover summary over **4,481 Sentinel-2 patches** covering the Lahore
region.

---

## Table of Contents

- [Purpose](#purpose)
- [Repository Layout](#repository-layout)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Efficiency Profile](#efficiency-profile)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [Lahore Case Study](#lahore-case-study)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility Notes](#reproducibility-notes)
- [Project Artefacts](#project-artefacts)
- [License & Acknowledgments](#license--acknowledgments)

---

## Purpose

This project demonstrates the application of deep learning for **remote
sensing and environmental monitoring**. The specific objectives are:

1. **Land Use & Land Cover (LULC) Classification** — automatically classify
   satellite images into meaningful land-cover categories.
2. **Custom Architecture vs. Transfer Learning** — compare a CNN built from
   scratch against a pretrained industry-standard backbone (ResNet50).
3. **Multi-Metric Evaluation** — analyse performance with accuracy, macro /
   weighted F1, per-class precision / recall / F1, and normalised confusion
   matrices.
4. **Efficiency Profiling** — quantify the accuracy-vs-cost trade-off with
   parameter count, FLOPs, peak GPU memory, and wall-clock latency.
5. **Explainability** — visualise what the trained ResNet50 attends to using
   Grad-CAM heatmaps.
6. **Real-World Application** — apply the classifier to Sentinel-2 patches
   over Lahore to evaluate out-of-distribution generalisation to local
   Pakistani terrain.

**Applications** include urban planning, agricultural land assessment,
environmental change detection, natural resource management, and disaster
response.

---

## Repository Layout

```
Eurosatproject/
│
├── notebooks/                                   # Jupyter notebooks
│   ├── lahore_data_acquisition.ipynb             # GEE Sentinel-2 acquisition (Lahore)
│   └── patches_64_64.ipynb                      # TIFF → 64×64 JPEG patch slicer
│
├── src/                                         # Source code
│   ├── enhanced_training.py                     # End-to-end training pipeline (both models)
│   ├── lahore_analysis.py                       # Lahore inference + 4-bin land-cover aggregation
│   ├── gradcam.py                               # Grad-CAM visualisations (ResNet50)
│   └── efficiency.py                            # Params, FLOPs, latency, peak memory
│
├── data/                                        # Datasets (not tracked in git)
│   ├── Eurosat_Dataset/                         # EuroSAT RGB JPEG flat folder (~ 2 GB)
│   └── Lahore_Dataset/
│       ├── lahore_rgb_2023.tif                  # Sentinel-2 RGB export (EPSG:32643)
│       ├── lahore_rgb_2023.tif.aux.xml
│       ├── lahore_rgb_2023.tif.ovr
│       └── patches/                             # 4,481 64×64 RGB JPEG patches
│
├── models/                                      # Trained checkpoints (Git LFS)
│   ├── custom_cnn/
│   │   ├── best_model.pth                       # Best val-accuracy weights
│   │   └── eurosat_cnn_final.pth                # Final bundle (weights + metadata)
│   └── resnet50/
│       ├── best_model.pth
│       └── eurosat_resnet50_final.pth
│
├── results/                                     # Generated artefacts (300 DPI)
│   ├── training_curves.png                      # 3564 × 1335
│   ├── confusion_matrix_cnn.png                 # 2593 × 2364
│   ├── confusion_matrix_resnet50.png            # 2593 × 2364
│   ├── per_class_metrics.png                    # 3864 × 1614
│   ├── metrics_export.xlsx                      # 4 sheets
│   ├── efficiency_bubble_chart.png              # 2516 × 1769
│   ├── efficiency_metrics.xlsx                  # 2 sheets
│   ├── gradcam_samples_grid.png                 # 4359 × 4424
│   ├── gradcam_individual/                      # 50 individual 3-panel CAM figures
│   ├── lahore_analysis.xlsx                     # 4 sheets
│   ├── lahore_land_cover_pie.png                # 2376 × 1906
│   └── lahore_class_distribution.png            # 3270 × 1470
│
├── proposals/                                   # Project proposal PDFs
│   └── ProjectProposal/
│       ├── Proposal_ShahidUsman_MSDS.pdf
│       └── Analysis_Execution_Plan_MSU.pdf
│
├── Analysis_Execution_Plan_MSU.docx            # Detailed execution plan
├── eurosat_paper.tex                            # LaTeX source for the project report
├── main.py                                      # Placeholder entry-point
├── pyproject.toml                               # Project + dependency manifest
├── uv.lock                                      # Pinned dependency graph
├── .python-version                              # 3.10
├── .gitignore
├── .gitattributes                               # Git LFS tracking for *.pth
└── README.md
```

---

## Dataset

### EuroSAT RGB

EuroSAT is based on Sentinel-2 satellite imagery and contains **27,000 labelled
64 × 64 RGB JPEG** images across 10 land-cover classes.

#### 10 Land-Cover Classes

| Class | Description |
|-------|-------------|
| `AnnualCrop` | Temporary agricultural fields (wheat, maize, etc.) |
| `Forest` | Dense forest areas |
| `HerbaceousVegetation` | Grasslands and meadows |
| `Highway` | Major road networks |
| `Industrial` | Industrial zones and commercial areas |
| `Pasture` | Grazing lands and pastures |
| `PermanentCrop` | Orchards and vineyards |
| `Residential` | Residential and urban areas |
| `River` | Rivers and streams |
| `SeaLake` | Seas and large lakes |

#### Dataset Statistics

| Property | Value |
|----------|-------|
| Total images | 27,000 |
| Resolution | 64 × 64 pixels |
| Color channels | 3 (RGB) |
| Train / Val / Test split | 70 % / 15 % / 15 % → 18,900 / 4,050 / 4,050 |
| Channel mean (RGB) | `[0.3444, 0.3803, 0.4078]` |
| Channel std (RGB)  | `[0.2034, 0.1365, 0.1148]` |

> The EuroSAT folder is expected to be a *flat* directory of JPEGs named
> `<ClassName>_<id>.jpg` (e.g. `AnnualCrop_123.jpg`). The label is recovered
> from the filename prefix.

#### Download

The dataset is **not tracked** in this repository (~ 2 GB). Download from the
official EuroSAT source and place the extracted `Eurosat_Dataset/` folder in
the project root, next to `src/`.

---

## Methodology

### 1. Custom CNN — *trained from scratch*

A compact 3-block VGG-style network with batch-norm, max-pool, and 2D dropout,
followed by a 3-layer fully-connected head. Input: **64 × 64 RGB**.

| Block | Layers | Output |
|-------|--------|--------|
| Conv-1 | `Conv 3→32`, BN, ReLU × 2, MaxPool 2, Dropout2d 0.25 | 32 × 32 × 32 |
| Conv-2 | `Conv 32→64`, BN, ReLU × 2, MaxPool 2, Dropout2d 0.25 | 64 × 16 × 16 |
| Conv-3 | `Conv 64→128`, BN, ReLU × 2, MaxPool 2, Dropout2d 0.25 | 128 × 8 × 8 |
| FC head | Flatten → 512 → 256 → 10 (Dropout 0.5, 0.3) | 10 |

- **Trainable parameters:** 4,616,618 (≈ 4.62 M)
- **Inference resolution:** native 64 × 64 (no upscaling)

### 2. ResNet50 — *transfer learning*

`torchvision.models.resnet50` with ImageNet weights, `fc` replaced by
`nn.Linear(2048, 10)`. All layers are fine-tuned (no frozen backbone).

- **Trainable parameters:** 23,528,522 (≈ 23.53 M)
- **Inference resolution:** 224 × 224 (64 × 64 patches are upsampled)

### 3. Training Configuration

| Hyperparameter | Custom CNN | ResNet50 |
|----------------|------------|----------|
| Optimiser | Adam (lr = 1e-3) | Adam (lr = 1e-4) |
| Weight decay | 1e-4 | 1e-4 |
| LR scheduler | `StepLR` (γ = 0.5, step 7) | `StepLR` (γ = 0.5, step 7) |
| Loss | `CrossEntropyLoss` | `CrossEntropyLoss` |
| Batch size | 64 | 16 (constrained by 4 GB VRAM at 224×224) |
| Epochs | 10 | 5 |
| Augmentation | H/V flip, ±15° rotation, color jitter | identical |
| Reproducibility | `SEED = 42`, deterministic cuDNN off (cuDNN benchmark on) | identical |
| Hardware | Quadro P2000 4 GB | same |

> The full training pipeline is implemented in `src/enhanced_training.py` and
> is **deterministic** (seeded RNGs, fixed split) so re-runs reproduce the
> published numbers.

---

## Results

> All numbers below are pulled directly from `results/metrics_export.xlsx`
> (4 sheets: `Overall_Accuracy`, `Per_Class_Metrics`, `Training_History`,
> `Model_Comparison`) and reflect a single deterministic re-evaluation of the
> saved `best_model.pth` checkpoints on the 4,050-image held-out test split.

### Headline Comparison

| Model | Test Accuracy | Macro F1 | Weighted F1 | Params (M) | Epochs | Throughput (ms / img)¹ |
|-------|--------------:|---------:|------------:|-----------:|-------:|----------------------:|
| **Custom CNN** | **81.38 %** | 0.8050 | 0.8113 | 4.62 | 10 | 1.82 |
| **ResNet50**   | **97.41 %** | 0.9732 | 0.9741 | 23.53 | 5  | 10.44 |

¹ *Wall-clock per image over the **entire test loader** (large effective
batch throughput). For batch-size-1 latency, see the
[Efficiency Profile](#efficiency-profile) section.*

The headline gain of **+16.0 percentage points** (81.38 % → 97.41 %) for a
5.1× increase in parameters demonstrates the value of ImageNet pretraining
on small satellite datasets.

### Per-Class Metrics — Custom CNN

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| AnnualCrop           | 0.7983 | 0.8747 | 0.8348 | 439 |
| Forest               | 0.9687 | 0.9687 | 0.9687 | 447 |
| HerbaceousVegetation | 0.8100 | 0.5963 | 0.6869 | 436 |
| Highway              | 0.5614 | 0.6230 | 0.5906 | 374 |
| Industrial           | 0.9005 | 0.9289 | 0.9145 | 380 |
| Pasture              | 0.7022 | 0.9328 | 0.8013 | 268 |
| PermanentCrop        | 0.6277 | 0.6754 | 0.6507 | 382 |
| Residential          | 0.8854 | 0.9698 | 0.9256 | 430 |
| River                | 0.8671 | 0.5905 | 0.7025 | 420 |
| SeaLake              | 0.9787 | 0.9705 | 0.9746 | 474 |
| **Macro avg**        | **0.8100** | **0.8130** | **0.8050** | **4,050** |

The Custom CNN is largely correct on the visually distinctive classes
(`Forest` 0.969, `SeaLake` 0.975, `Industrial` 0.915) and confuses the
visually similar agricultural classes (`AnnualCrop` ↔ `PermanentCrop` ↔
`Pasture` ↔ `HerbaceousVegetation`). Its biggest weakness is `Highway`
(F1 = 0.591) — linear road geometry at 64 × 64 is hard to learn from
scratch.

### Per-Class Metrics — ResNet50

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| AnnualCrop           | 0.9387 | 0.9772 | 0.9576 | 439 |
| Forest               | 0.9824 | 0.9978 | 0.9900 | 447 |
| HerbaceousVegetation | 0.9725 | 0.9748 | 0.9737 | 436 |
| Highway              | 0.9815 | 0.9947 | 0.9880 | 374 |
| Industrial           | 1.0000 | 0.9211 | 0.9589 | 380 |
| Pasture              | 0.9520 | 0.9627 | 0.9573 | 268 |
| PermanentCrop        | 0.9734 | 0.9581 | 0.9657 | 382 |
| Residential          | 0.9471 | 1.0000 | 0.9729 | 430 |
| River                | 0.9926 | 0.9619 | 0.9770 | 420 |
| SeaLake              | 1.0000 | 0.9810 | 0.9904 | 474 |
| **Macro avg**        | **0.9740** | **0.9730** | **0.9732** | **4,050** |

ResNet50 lifts every class — most dramatically `Highway` (0.591 → 0.988,
+0.397) and `HerbaceousVegetation` (0.687 → 0.974, +0.287). Two classes
reach a perfect `Industrial` precision of 1.0 and `Residential` recall of
1.0. The biggest remaining weakness is `Industrial` recall (0.9211) — the
model occasionally confuses `Industrial` zones with `Residential` blocks.

### Confusion Matrices

Normalised row-wise confusion matrices are saved at 300 DPI (2593 × 2364 px):

- `results/confusion_matrix_cnn.png` — Custom CNN
- `results/confusion_matrix_resnet50.png` — ResNet50

### Training Dynamics

`results/training_curves.png` (3564 × 1335) plots train/val loss and accuracy
for both models side-by-side. ResNet50 reaches **97.26 % validation accuracy
in the first epoch** and converges to ~ 97.5 %; the Custom CNN climbs
steadily from 52.7 % to 82.3 % val accuracy over its 10 epochs.

The full per-epoch log (loss + accuracy, train and val) is exported to the
`Training_History` sheet of `metrics_export.xlsx`.

### Multi-Sheet Excel Export

`results/metrics_export.xlsx` (9.2 KB) contains four sheets:

| Sheet | Contents |
|-------|----------|
| `Overall_Accuracy` | Per-model accuracy, F1 (macro + weighted), params, inference time |
| `Per_Class_Metrics` | Precision / Recall / F1 / Support for every class × every model (20 rows) |
| `Training_History` | Per-epoch train & val loss / accuracy (15 rows: 10 CNN + 5 ResNet50) |
| `Model_Comparison` | Side-by-side accuracy-vs-cost table |

---

## Efficiency Profile

Measured by `src/efficiency.py` on a **Quadro P2000 (4 GB)** with **batch
size 1**, median of 100 timed runs after 10 warm-up passes (CUDA-synchronised).

> Source: `results/efficiency_metrics.xlsx` → `Efficiency_Metrics` sheet.
> Host snapshot: 12 logical CPUs (6 cores), 15.78 GB total RAM, 78.8 % used.

| Model | Input | Params | Size (MB) | MACs | G-MACs | GFLOPs | Latency (ms/img) | Peak GPU (MB) |
|-------|-------|-------:|----------:|-----:|-------:|-------:|-----------------:|--------------:|
| Custom CNN | 64 × 64 | 4,616,618 | 17.61 | 160,696,832 | 0.16 | 0.32 | **1.79** | 18.74 |
| ResNet50   | 224 × 224 | 23,528,522 | 89.75 | 4,131,715,072 | 4.13 | 8.26 | **14.85** | 118.16 |

- ResNet50 is **5.1× larger** in parameters and runs at **12.25× higher input
  resolution**, producing **25.7×** more MACs per image.
- The trade-off is a **~8.3×** per-image latency overhead (1.79 → 14.85 ms)
  for a **+16.0 percentage-point** accuracy gain (81.38 → 97.41 %).
- A bubble chart (`results/efficiency_bubble_chart.png`, 2516 × 1769)
  visualises params vs. latency with bubble area proportional to G-MACs.
- The full per-model table plus a host CPU/RAM snapshot are written to
  `results/efficiency_metrics.xlsx` (sheets `Efficiency_Metrics` and
  `Host_Snapshot`).

---

## Grad-CAM Explainability

`src/gradcam.py` applies **Gradient-weighted Class Activation Mapping** to the
final convolutional block of ResNet50 (`layer4[-1].conv3`) on **50 EuroSAT
test images, balanced across the 10 classes** (5 per class). The script:

1. Loads the trained ResNet50 checkpoint (`models/resnet50/eurosat_resnet50_final.pth`).
2. Samples 5 images per class (50 total) at `SEED = 42`.
3. Runs Grad-CAM with `ClassifierOutputTarget(predicted_class)`.
4. Renders a 3-panel figure per image (input · heatmap · overlay) into
   `results/gradcam_individual/gradcam_xxx_true-<T>_pred-<P>.png`.
5. Assembles a grid into `results/gradcam_samples_grid.png` (4359 × 4424 px).

The 50 individual files cover all 10 EuroSAT classes on both the true and
predicted axes (e.g. `gradcam_000_true-River_pred-River.png`).

The heatmaps confirm that the model attends to **semantically meaningful
land-cover features** (e.g. road geometry for `Highway`, building footprints
for `Industrial` / `Residential`, water edges for `River` / `SeaLake`).

---

## Lahore Case Study

To test out-of-distribution generalisation, a Sentinel-2 scene covering the
Lahore district was acquired via **Google Earth Engine**
(`COPERNICUS/S2_SR_HARMONIZED`, 2026 Q1, cloud-cover < 10 %), exported to a
GeoTIFF in **EPSG:32643** at 10 m / pixel, and then sliced into **4,481
non-overlapping 64 × 64 RGB patches** by `patches_64_64.ipynb`.

The trained ResNet50 (`src/lahore_analysis.py`) was then run on every patch
and the 10 EuroSAT predictions were aggregated into a coarser 4-bin
district-scale taxonomy:

| Land-cover bin | EuroSAT classes mapped in | Rationale |
|----------------|---------------------------|-----------|
| `Urban`         | `Residential`, `Industrial`, `Highway` | Built-up land and transport corridors |
| `Agricultural`  | `AnnualCrop`, `PermanentCrop`, `Pasture` | Cropland and managed vegetation |
| `Water`         | `River`, `SeaLake` | Open water |
| `Natural`       | `Forest`, `HerbaceousVegetation` | Unmanaged vegetation |

### Aggregated Land-Cover Distribution (4 bins)

> Source: `results/lahore_analysis.xlsx` → `Land_Cover_Summary` sheet.

| Land-cover bin | Patches | Share | Mean confidence | Median confidence |
|----------------|--------:|------:|----------------:|------------------:|
| **Agricultural** | 2,042 | **45.57 %** | 0.6759 | 0.6807 |
| **Natural**      | 1,314 | **29.32 %** | 0.6830 | 0.6851 |
| **Urban**        | 1,042 | **23.25 %** | 0.7149 | 0.7393 |
| **Water**        |    83 |  **1.85 %** | 0.4924 | 0.4897 |
| **Total**        | 4,481 | 100 % | — | — |

The distribution is dominated by **Agricultural** land (45.6 %), with
**Natural** vegetation a close second (29.3 %) and **Urban** surfaces making
up nearly a quarter of the district (23.3 %). Water is the smallest bin at
1.85 %.

**Mean confidence is highest in the Urban bin (0.715)** and lowest in the
Water bin (0.492). The high Urban confidence is driven primarily by the
`Residential` and `Highway` sub-classes; the low Water confidence reflects
the small number of patches (only 42 `River` + 41 `SeaLake`) and the
co-occurrence of agricultural and water pixels in mixed boundary patches.

### Per-EuroSAT-Class Breakdown (10 classes)

> Source: `results/lahore_analysis.xlsx` → `EuroSAT_Class_Breakdown` sheet.

| EuroSAT class | Patches | Share | Mean confidence | Land-cover bin |
|---------------|--------:|------:|----------------:|----------------|
| PermanentCrop        | 1,645 | 36.71 % | 0.6953 | Agricultural |
| HerbaceousVegetation | 1,314 | 29.32 % | 0.6830 | Natural |
| Residential          |   620 | 13.84 % | 0.7149 | Urban |
| AnnualCrop           |   354 |  7.90 % | 0.6097 | Agricultural |
| Highway              |   311 |  6.94 % | 0.7368 | Urban |
| Industrial           |   111 |  2.48 % | 0.6535 | Urban |
| Pasture              |    43 |  0.96 % | 0.4800 | Agricultural |
| River                |    42 |  0.94 % | 0.4998 | Water |
| SeaLake              |    41 |  0.91 % | 0.4849 | Water |
| Forest               |     0 |  0.00 % | — | Natural |
| **Total**            | 4,481 | 100 % | — | — |

The **top three** predicted classes — `PermanentCrop` (1,645),
`HerbaceousVegetation` (1,314), and `Residential` (620) — account for
nearly 80 % of all patches. **No patch was classified as `Forest`**, which
is consistent with the heavily cultivated peri-urban landscape around
Lahore where dense forest cover is rare.

### Limitations

- 64 × 64 patches at 10 m = **640 m ground footprint per patch**, which is
  coarse for heterogeneous urban areas.
- The EuroSAT 10-class taxonomy is **not Lahore-specific**; predictions are
  reinterpreted into 4 bins and should be read as a coarse LULC overview
  rather than ground-truth labels.
- 64 → 224 upsampling for ResNet50 introduces an additional distribution
  shift (documented in `lahore_analysis.py`).

### Lahore Artifacts

| File | Description |
|------|-------------|
| `results/lahore_analysis.xlsx` | 4 sheets: `Land_Cover_Summary`, `EuroSAT_Class_Breakdown`, `Per_Patch_Predictions` (4,481 rows), `Confidence_Statistics` |
| `results/lahore_land_cover_pie.png` | 2376 × 1906 px donut chart of the 4-bin distribution |
| `results/lahore_class_distribution.png` | 3270 × 1470 px per-EuroSAT-class bar chart (10 classes) |

---

## Installation

### Requirements

- **Python:** 3.10 (pinned in `.python-version`)
- **Package manager:** [`uv`](https://docs.astral.sh/uv/)
- **GPU (recommended):** CUDA-compatible. The reported numbers were produced
  on a 4 GB Quadro P2000; ResNet50 at 224 × 224 needs batch size ≤ 16.
- **CPU fallback:** supported — inference will be substantially slower.

### Steps

```bash
# 1. Clone the repository
git clone <this-repo>
cd Eurosatproject

# 2. Install all dependencies into a project-local .venv
uv sync

# 3. Pull pre-trained checkpoints via Git LFS (best_model.pth files)
git lfs install
git lfs pull
```

### Dependencies (high-level)

Core: `torch==1.13.1+cu117`, `torchvision==0.14.1+cu117`
Data: `numpy`, `pandas`, `Pillow`, `rasterio`, `openpyxl`
Modelling: `scikit-learn`, `tqdm`
Visualisation: `matplotlib`, `seaborn`
Efficiency: `thop`, `psutil`
Explainability: `pytorch-gradcam`, `grad-cam`
Geospatial: `earthengine-api`, `geemap`, `dotenv`
Notebooks: `jupyter`, `ipykernel`

The full pinned graph lives in `uv.lock`.

---

## Usage

> All commands assume the project root as the working directory.

### 1. Acquire the Lahore Sentinel-2 scene (one-time)

`notebooks/lahore_data_acquisition.ipynb` walks through authenticating to
Google Earth Engine, exporting the Lahore RGB composite to Google Drive, and
slicing the downloaded GeoTIFF into 64 × 64 patches. Requires a free GEE
account and a `GEE_PROJECT_ID` in `.env`.

### 2. Train both models

```bash
uv run python src/enhanced_training.py
```

This:

- trains the **Custom CNN** for 10 epochs at 64 × 64;
- trains **ResNet50** for 5 epochs at 224 × 224;
- saves the best-validation checkpoint to `models/<model>/best_model.pth`
  and a metadata bundle to `models/<model>/eurosat_<model>_final.pth`;
- regenerates `results/training_curves.png`, `confusion_matrix_*.png`,
  `per_class_metrics.png`, and `metrics_export.xlsx`.

### 3. Profile efficiency

```bash
uv run python src/efficiency.py
```

Produces `results/efficiency_bubble_chart.png` and `efficiency_metrics.xlsx`.

### 4. Generate Grad-CAM heatmaps

```bash
uv run python src/gradcam.py
```

Produces `results/gradcam_samples_grid.png` and 50 individual figures in
`results/gradcam_individual/`.

### 5. Run the Lahore inference

```bash
uv run python src/lahore_analysis.py
```

Produces `results/lahore_land_cover_pie.png`,
`results/lahore_class_distribution.png`, and `results/lahore_analysis.xlsx`.

### 6. Run a notebook

```bash
uv run jupyter notebook notebooks/patches_64_64.ipynb
```

> **Note:** `notebooks/patches_64_64.ipynb` reads from a hard-coded local
> path (`C:\Users\SHEHER SAAZ\Music\…\lahore_rgb_2023.tif`). Update the
> `TIF_PATH` and `OUTPUT_DIR` constants at the top of the notebook to match
> your machine before executing.

---

## Reproducibility Notes

- **Seeds:** `SEED = 42` is set in `random`, `numpy`, `torch` (CPU + CUDA),
  and the DataLoader shuffle is seeded via `np.random.RandomState(42)`.
- **cuDNN:** `benchmark = True` is enabled (faster) and `deterministic =
  False` is set. Re-runs are therefore expected to be **close** but not
  bit-identical across different CUDA / cuDNN versions. For a slower
  bit-exact run, flip `cudnn.deterministic = True` in `set_seed()`.
- **Data split:** The train/val/test indices are shuffled with a fixed seed
  *before* splitting, so the published test indices are deterministic.
- **Checkpoints:** `models/<model>/best_model.pth` reflects the best
  validation accuracy seen during training, **not** the final epoch. The
  published test numbers come from re-evaluating this checkpoint.
- **Two `latency_ms/img` columns:** the headline comparison table
  (1.82 / 10.44) reports **throughput** measured during `evaluate()` over
  the full 4,050-image test set, while the Efficiency Profile
  (1.79 / 14.85) uses a controlled **batch-1** measurement (median of 100
  timed runs after 10 warm-up passes), which is more representative of
  single-image serving latency.

---

## Project Artefacts

### Models (Git LFS)

| File | Description |
|------|-------------|
| `models/custom_cnn/best_model.pth`        | Best-val-accuracy Custom CNN weights |
| `models/custom_cnn/eurosat_cnn_final.pth` | Bundle: weights + metadata |
| `models/resnet50/best_model.pth`         | Best-val-accuracy ResNet50 weights |
| `models/resnet50/eurosat_resnet50_final.pth` | Bundle: weights + metadata |

### Results

| File | Description |
|------|-------------|
| `results/training_curves.png`           | 3564 × 1335 — Loss + accuracy curves (both models) |
| `results/confusion_matrix_cnn.png`      | 2593 × 2364 — Normalised confusion matrix, Custom CNN |
| `results/confusion_matrix_resnet50.png` | 2593 × 2364 — Normalised confusion matrix, ResNet50 |
| `results/per_class_metrics.png`         | 3864 × 1614 — Per-class precision / recall / F1 (both models) |
| `results/metrics_export.xlsx`           | 9.2 KB — 4 sheets: overall, per-class, history, comparison |
| `results/efficiency_bubble_chart.png`   | 2516 × 1769 — Params × latency bubble chart |
| `results/efficiency_metrics.xlsx`       | 5.8 KB — 2 sheets: efficiency + host snapshot |
| `results/gradcam_samples_grid.png`      | 4359 × 4424 — Grad-CAM overlay grid (ResNet50) |
| `results/gradcam_individual/`           | 50 individual 3-panel Grad-CAM figures (all 10 classes) |
| `results/lahore_analysis.xlsx`          | 142 KB — 4 sheets: summary, class breakdown, per-patch (4,481 rows), confidence |
| `results/lahore_land_cover_pie.png`     | 2376 × 1906 — Donut chart of the 4-bin Lahore distribution |
| `results/lahore_class_distribution.png` | 3270 × 1470 — Bar chart of the 10 EuroSAT class shares |

### Documentation

| File | Description |
|------|-------------|
| `eurosat_paper.tex`                     | LaTeX source for the project report |
| `proposals/ProjectProposal/Proposal_ShahidUsman_MSDS.pdf` | Original project proposal |
| `proposals/ProjectProposal/Analysis_Execution_Plan_MSU.pdf` | Analysis & execution plan |
| `Analysis_Execution_Plan_MSU.docx`      | Same plan in Word format |

---

## License & Acknowledgments

This project is developed for **educational purposes** as part of the
*Tools & Techniques in Data Science* course at Riphah University.

- **EuroSAT dataset** — Helber et al.,
  *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and
  Land Cover Classification*, IEEE Journal of Selected Topics in Applied
  Earth Observations and Remote Sensing, 2017.
  [github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT)
- **Sentinel-2 imagery** courtesy of the European Space Agency (ESA),
  accessed through Google Earth Engine.
- **ResNet50 backbone** — He, Zhang, Ren & Sun, *Deep Residual Learning for
  Image Recognition*, CVPR 2016.
- **Grad-CAM** — Selvaraju et al., *Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization*, ICCV 2017.
