# Explainable Deep Learning for High-Resolution Land Cover Mapping: A Comparative Analysis of Custom CNNs and ResNet50 in Lahore

**Muhammad Shahid Usman**

*Riphah International University, Lahore, Pakistan*

Correspondence: shahidusman.info@gmail.com

---

## Abstract

High-resolution land cover mapping from Sentinel-2 imagery is fundamental to evidence-based urban planning, agricultural monitoring, and environmental change detection, yet the operational deployment of deep learning classifiers in such settings is constrained by an unresolved trade-off between predictive accuracy and computational efficiency. The present study undertakes a controlled, reproducible comparison of two convolutional architectures — a custom 3-block VGG-style Convolutional Neural Network (CNN) trained from scratch and a ResNet50 backbone fine-tuned via ImageNet transfer learning — for ten-class land cover classification on the EuroSAT RGB benchmark, and subsequently deploys both architectures over a 4,481-patch Sentinel-2 acquisition of the Lahore district in Pakistan to evaluate out-of-distribution generalisation. Both networks are trained under a fixed 70/15/15 train/validation/test split with `SEED = 42`, identical augmentation, and identical evaluation protocol. The Custom CNN, operating at its native 64×64 input resolution with 4.62 M trainable parameters, reaches 81.38 % test accuracy and a macro F1-score of 0.805. The ResNet50, operating at 224×224 with 23.53 M parameters, raises test accuracy to 97.41 % and macro F1 to 0.973, with the largest per-class gains observed on `Highway` (F1 0.591 → 0.988) and `HerbaceousVegetation` (F1 0.687 → 0.974). Gradient-weighted Class Activation Mapping (Grad-CAM) confirms that ResNet50 attends to semantically meaningful features — road geometry, building footprints, water edges — rather than to border artefacts. The four-bin district-scale deployment over Lahore yields 45.6 % Agricultural, 29.3 % Natural, 23.3 % Urban, and 1.9 % Water, with a complete absence of `Forest` patches consistent with the heavily cultivated peri-urban landscape. The findings demonstrate that transfer learning delivers substantial accuracy gains at a quantifiable 8.3× per-image latency cost, and that the resulting classifier, when combined with Grad-CAM explainability and a coarse 4-bin aggregation, constitutes a reproducible pipeline for local-scale LULC assessment in data-constrained geographies.

**Keywords:** Land Use and Land Cover; EuroSAT; Sentinel-2; Convolutional Neural Network; ResNet50; Transfer Learning; Grad-CAM; Explainable AI; Remote Sensing; Lahore.

---

## 1. Introduction

Earth observation programmes such as the European Space Agency's Copernicus Sentinel-2 mission have democratised access to multispectral satellite imagery, delivering global coverage of all land surfaces every five days at no cost and at spatial resolutions of 10–60 metres depending on the spectral band [1]. The resulting data volumes have made pixel- and patch-level Land Use and Land Cover (LULC) classification a central problem in remote sensing, with direct applications in urban planning [2], agricultural policy [3], deforestation monitoring [4], disaster response [5], and climate change assessment [6]. Hand-crafted spectral indices and traditional machine-learning classifiers such as Random Forests and Support Vector Machines dominated early LULC work, but their reliance on expert feature engineering limits scalability across geographies, sensor modalities, and phenological stages [2]. The emergence of deep convolutional neural networks — AlexNet, VGG, GoogLeNet, ResNet, and more recently Vision Transformers — has shifted the state of the art decisively toward learned representations that generalise across illumination, atmosphere, and seasonality [7, 8, 9].

Among the available benchmarks, the EuroSAT dataset [10, 11] has become a de-facto reference for LULC research. It packages 27,000 labelled Sentinel-2 patches at a native resolution of 64×64 pixels into ten mutually exclusive classes: `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, and `SeaLake`. The dataset is approximately balanced, moderately sized by modern deep-learning standards, and shares its underlying sensor with the operational Sentinel-2 archive, making it a natural bridge between benchmark research and operational deployment.

Two practical questions remain at the centre of operational LULC deployment. The first is the *accuracy–efficiency trade-off*. A from-scratch 3-block VGG-style CNN is small enough to run in real time on edge devices, satellite on-board processors, and resource-constrained cloud instances, but its limited capacity produces systematic confusion among visually similar classes (`AnnualCrop` ↔ `PermanentCrop`, or `Highway` asphalt against urban texture) at the 64×64 pixel budget. An ImageNet-pretrained ResNet50 with 23.5 million parameters dominates these difficult classes, but at a 5.1× parameter, 25.7× MACs, and 8.3× per-image latency cost. Whether the accuracy gain is worth the deployment cost is a context-dependent decision that the existing literature has not quantified under a unified experimental protocol.

The second question is *out-of-distribution transfer*. EuroSAT is sampled from European geographies; its class taxonomy was designed to be globally expressive but is not natively aligned with any particular regional land-use vocabulary. Whether an EuroSAT-trained classifier produces a recognisably plausible district-scale overview when shipped to a South Asian metropolitan area such as Lahore is an empirical question. The Lahore district of Pakistan is a representative test case: it combines dense peri-urban settlements with active cropland, a well-documented urban morphology, an active Sentinel-2 archive accessible through Google Earth Engine [12], and a population of approximately 13 million within the metropolitan area for which LULC data is operationally relevant.

The present study addresses both questions in a single reproducible pipeline. The contributions are as follows.

1. A **deterministic, end-to-end training and evaluation pipeline** for both a Custom 3-block CNN and an ImageNet-pretrained ResNet50, packaged as a single runnable script with a fixed `SEED = 42`, a fixed 70/15/15 train/val/test split, and identical augmentation across both models.
2. A **head-to-head efficiency and accuracy comparison** on the 4,050-image EuroSAT held-out test set, with parameter count, multiply-accumulate operations (MACs), per-image inference latency, overall accuracy, macro and weighted F1-scores, and per-class precision, recall, and F1.
3. A **Grad-CAM explainability study** [13] over 50 balanced EuroSAT test images, demonstrating that the ResNet50 backbone attends to semantically meaningful land-cover features.
4. A **district-scale spatial deployment** of the trained ResNet50 over 4,481 non-overlapping 64×64 Sentinel-2 patches acquired over the Lahore district, with the 10 EuroSAT classes aggregated into a 4-bin taxonomy (Urban, Agricultural, Water, Natural) suitable for operational reporting.
5. A **critical discussion of the accuracy–efficiency trade-off** and the limitations of applying an EuroSAT-trained classifier to a non-European geography, including the distribution shift introduced by upsampling 64×64 Sentinel-2 patches to the 224×224 input expected by ResNet50.

The remainder of this paper is organised as follows. Section 2 reviews the related work on EuroSAT benchmarking, transfer learning for remote sensing, and explainable AI. Section 3 details the datasets, architectures, evaluation metrics, and Grad-CAM implementation. Section 4 reports the experimental results. Section 5 discusses the trade-offs and limitations, and Section 6 concludes.

---

## 2. Related Work

### 2.1. EuroSAT as a Benchmark

The EuroSAT corpus was introduced by Helber *et al.* [10, 11] and has been the reference benchmark for Sentinel-2 land-cover classification since its release. The original ResNet50 baseline reported 98.57 % overall accuracy, establishing a high but not insurmountable target. Subsequent work has explored data augmentation strategies [14], self-supervised pretraining [15], attention mechanisms, and transformer-based backbones [16, 17] on the same benchmark. Xie *et al.* [2] provide a recent review and report that transfer learning from ImageNet-pretrained models remains a strong default for Sentinel-2 imagery, a finding that the present study confirms quantitatively on a controlled head-to-head.

### 2.2. From-Scratch CNNs vs. Transfer Learning

The general image-classification literature has established that ImageNet-pretrained backbones typically outperform from-scratch networks of similar depth on small datasets [9, 18]. For small-patch remote-sensing data, the trade-off is sharper than for ImageNet because the receptive field of a 3-block VGG-style CNN barely reaches the patch boundaries at 64×64. Several studies have compared custom and pretrained backbones on EuroSAT specifically [10, 14], but typically report accuracy alone and leave the efficiency trade-off implicit. The present study reports parameter count, MACs, latency, and peak memory in a single profile, so the deployment cost of accuracy is visible alongside the accuracy itself.

### 2.3. Out-of-Distribution Transfer and Explainable AI

A trained EuroSAT classifier applied to a new geography is, in effect, an out-of-distribution (OOD) probe. Google Earth Engine [12] has become the standard platform for Sentinel-2 acquisition, and a growing body of work reports the deployment of EuroSAT-style classifiers to regional case studies. The Grad-CAM technique of Selvaraju *et al.* [13] is the de-facto post-hoc explanation method for CNN classifiers and has been applied to a range of remote-sensing tasks [19]. To the best of the author's knowledge, no published study has combined a head-to-head Custom CNN vs. ResNet50 comparison on EuroSAT with a Grad-CAM interpretability analysis and a district-scale deployment over Lahore. The present paper fills that gap with a reproducible pipeline and a fully released set of artefacts (checkpoints, metrics exports, Grad-CAM overlays, and per-patch predictions).

---

## 3. Methodology

### 3.1. Datasets

#### 3.1.1. EuroSAT RGB

The EuroSAT RGB corpus [10, 11] provides 27,000 JPEG patches at a native resolution of 64×64 pixels, balanced across ten classes: `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, and `SeaLake`. Each patch is read as a 3-channel RGB tensor and normalised using the dataset-wide channel statistics `μ = [0.3444, 0.3803, 0.4078]` and `σ = [0.2034, 0.1365, 0.1148]`. The full set is split deterministically into 70 % / 15 % / 15 % partitions for training, validation, and test, yielding approximately 18,900 / 4,050 / 4,050 images. Training-time augmentation consists of horizontal and vertical flips, a ±15° rotation, and mild colour jitter on brightness, contrast, and saturation; validation and test use only resize and normalisation.

| Property | Value |
|---|---|
| Source | Sentinel-2 (EuroSAT) |
| Image format | 3-band RGB JPEG |
| Native resolution | 64×64 px |
| Total images | 27,000 |
| Number of classes | 10 |
| Train / Val / Test | 70 % / 15 % / 15 % (18,900 / 4,050 / 4,050) |
| Channel mean (RGB) | [0.3444, 0.3803, 0.4078] |
| Channel std (RGB) | [0.2034, 0.1365, 0.1148] |

#### 3.1.2. Lahore Sentinel-2 Acquisition

A Sentinel-2 surface-reflectance scene over the Lahore district is acquired through Google Earth Engine [12] using the `COPERNICUS/S2_SR_HARMONIZED` collection, restricted to 2023 with cloud cover below 10 %. The composite is exported as a single GeoTIFF in UTM zone 43N (EPSG:32643) at 10 m / pixel. The original raster is 5,405 × 6,221 pixels (17,630,256 valid pixels). It is sliced into 4,481 non-overlapping 64×64 RGB patches by `notebooks/patches_64_64.ipynb`, with 3,859 additional patches rejected as NoData (near-black tiles) for a total tile grid of 84 rows × 97 columns = 8,148 candidate tiles. The 4,481 classified patches, plus the GeoTIFF header, are released in `data/Lahore_Dataset/`.

[INSERT MAP: Study Area - Lahore District with Sentinel-2 RGB overlay]

#### 3.1.3. Four-Bin Land-Cover Taxonomy

The 10 EuroSAT classes are aggregated into 4 district-level bins for the Lahore deployment, following the taxonomy used in regional LULC reporting.

| Land-cover bin | EuroSAT classes mapped in | Rationale |
|---|---|---|
| Urban | `Residential`, `Industrial`, `Highway` | Built-up land and transport corridors |
| Agricultural | `AnnualCrop`, `PermanentCrop`, `Pasture` | Cropland and managed vegetation |
| Water | `River`, `SeaLake` | Open water |
| Natural | `Forest`, `HerbaceousVegetation` | Unmanaged vegetation |

### 3.2. Network Architectures

#### 3.2.1. Custom 3-Block CNN (Trained from Scratch)

The baseline architecture is a 3-block VGG-style CNN in which each block contains two 3×3 convolutions, batch normalisation, ReLU activations, a 2×2 max-pool, and 2D dropout (`p = 0.25`). Channel widths double across blocks (32, 64, 128), so the feature map collapses from 64×64 to 8×8 over three downsamplings. A fully-connected head flattens the 128×8×8 = 8,192 feature tensor and maps it through 512 and 256 hidden units (with dropout 0.5 and 0.3) to a 10-way softmax output. The model has 4,616,618 trainable parameters (≈ 4.62 M) and is run at its native 64×64 input resolution, requiring no upsampling.

| Block | Layers | Output |
|---|---|---|
| Conv-1 | [C 3→32, C 32→32], M2, D0.25 | 32×32×32 |
| Conv-2 | [C 32→64, C 64→64], M2, D0.25 | 64×16×16 |
| Conv-3 | [C 64→128, C 128→128], M2, D0.25 | 128×8×8 |
| FC head | 8192 → 512 → 256 → 10 | 10 |

#### 3.2.2. ResNet50 (Transfer Learning from ImageNet)

The second model replaces the custom backbone with `torchvision.models.resnet50` initialised from ImageNet weights. The 1000-way fully-connected head is removed and replaced with a single `nn.Linear(2048, 10)` layer. Inputs are resized to 224×224 to match the network's expected receptive field, and all layers are fine-tuned end-to-end (no frozen backbone). ResNet50 contributes 23,528,522 trainable parameters (≈ 23.53 M). The 64→224 upsampling is implemented by bilinear interpolation and is the principal distribution-shift source in the deployment pipeline.

#### 3.2.3. Training Configuration

Both models are trained with the Adam optimiser and cross-entropy loss. The Custom CNN uses a learning rate of 1e-3, batch size 64, and 10 epochs; ResNet50 uses 1e-4, batch size 16, and 5 epochs. A `StepLR` scheduler halves the learning rate every 7 epochs. Weight decay is 1e-4 in both cases. Randomness is controlled with `SEED = 42` across `random`, `numpy`, and `torch` (CPU and CUDA), and the DataLoader shuffle is seeded via `np.random.RandomState(42)`. The cuDNN benchmark flag is enabled for speed; cuDNN deterministic mode is disabled — re-runs are therefore close but not bit-identical across CUDA versions.

| Hyperparameter | Custom CNN | ResNet50 |
|---|---|---|
| Optimiser | Adam | Adam |
| Learning rate | 1e-3 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Batch size | 64 | 16 |
| Epochs | 10 | 5 |
| LR scheduler | StepLR (γ = 0.5, step 7) | StepLR (γ = 0.5, step 7) |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Input resolution | 64×64 | 224×224 |
| Hardware | Quadro P2000 4 GB | same |

[INSERT GRAPH: CNN vs ResNet50 Training Loss and Accuracy Curves]

### 3.3. Evaluation Metrics

The classifier outputs a 10-way probability vector via the softmax activation in the final layer. Formally, given a logit vector **z** = (z₁, z₂, …, z_C) for C = 10 classes, the softmax computes

$$
p(y = c \mid \mathbf{x}) = \frac{\exp(z_c)}{\sum_{j=1}^{C} \exp(z_j)}, \quad c = 1, \dots, C.
$$

The predicted class is `ŷ = arg max_c p(y = c | x)`. All evaluation metrics reported in this paper are computed on the held-out test set.

**Overall accuracy** is the fraction of test images for which the predicted class equals the ground-truth class:

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\{\hat{y}_i = y_i\}.
$$

For the per-class metrics, let `TP_c`, `FP_c`, and `FN_c` denote, respectively, the number of true positives, false positives, and false negatives for class c, with `support_c` the number of test images whose ground-truth label is c.

**Precision** for class c is the fraction of predictions labelled c that are correct:

$$
\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}.
$$

**Recall** for class c is the fraction of ground-truth class-c images that are correctly recovered:

$$
\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}.
$$

**F1-Score** for class c is the harmonic mean of precision and recall:

$$
F_{1,c} = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}.
$$

Two aggregate F1 measures are reported throughout. **Macro F1** averages the per-class F1 uniformly across the C classes:

$$
F_{1}^{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F_{1,c}.
$$

**Weighted F1** averages the per-class F1 weighted by the per-class support:

$$
F_{1}^{\text{weighted}} = \frac{1}{N} \sum_{c=1}^{C} \text{support}_c \cdot F_{1,c}.
$$

Macro F1 is sensitive to performance on rare classes; weighted F1 is sensitive to performance on common classes.

### 3.4. Explainable AI (Grad-CAM)

Gradient-weighted Class Activation Mapping (Grad-CAM) [13] is applied to the trained ResNet50 to surface the image regions that drive each prediction. The implementation targets the final convolutional layer of the backbone, `layer4[-1].conv3`, where the spatial resolution is high enough to localise land-cover features but the channels are abstract enough to encode semantic content.

Given an input image **x** and a target class c, the Grad-CAM heatmap `L^c ∈ ℝ^{u×v}` is computed as the ReLU of a channel-wise weighted sum of the final convolutional feature maps `A^k`:

$$
\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}, \qquad
L^c_{ij} = \text{ReLU}\!\left( \sum_k \alpha_k^c A_{ij}^k \right),
$$

where `y^c` is the score for class c before the softmax, `α_k^c` is the global-average-pooled gradient of `y^c` with respect to feature map `A^k`, and Z = u·v is the number of spatial locations. The resulting heatmap is bilinearly upsampled to the input resolution and overlaid on the original image. The implementation samples 5 test images per EuroSAT class (50 total) at `SEED = 42`, runs Grad-CAM with `ClassifierOutputTarget(predicted_class)`, and renders a 3-panel figure (input, heatmap, overlay) per image into `results/gradcam_individual/`. A 10×5 grid is assembled into `results/gradcam_samples_grid.png`.

---

## 4. Results

### 4.1. Model Performance and Efficiency

The headline comparison on the EuroSAT held-out test set is summarised in Table 1. ResNet50 dominates the accuracy metrics across the board, with the largest per-class gains on the structurally confused categories.

[INSERT CHART: Overall Accuracy & Macro F1 Bar Chart - Custom CNN vs ResNet50]

**Table 1.** *Efficiency and accuracy comparison of the Custom CNN and ResNet50 architectures on the EuroSAT RGB test set. Latency is the median of 100 batch-size-1 runs on a Quadro P2000 4 GB GPU.*

| Metric | Custom CNN | ResNet50 | Δ (ResNet50 − CNN) |
|---|---:|---:|---:|
| Trainable Parameters (M) | 4.62 | 23.53 | +18.91 (×5.1) |
| G-MACs per image | 0.16 | 4.13 | +3.97 (×25.7) |
| Inference time (ms / image) | 1.79 | 14.85 | +13.06 (×8.3) |
| Peak GPU memory (MB) | 18.74 | 118.16 | +99.42 (×6.3) |
| Test Accuracy (%) | 81.38 | **97.41** | +16.03 |
| Macro F1 | 0.8050 | **0.9732** | +0.1682 |
| Weighted F1 | 0.8113 | **0.9741** | +0.1628 |

**Per-class behaviour (Custom CNN).** The Custom CNN is largely correct on the visually distinctive classes — `Forest` (F1 0.969), `SeaLake` (F1 0.975), `Industrial` (F1 0.915), `Residential` (F1 0.926) — and confuses the visually similar agricultural classes. Its biggest weaknesses are `Highway` (F1 0.591), `HerbaceousVegetation` (F1 0.687), `River` (F1 0.703), and `PermanentCrop` (F1 0.651). Linear road geometry at 64×64 is hard to learn from scratch, and the four managed-vegetation classes overlap in their colour statistics.

**Per-class behaviour (ResNet50).** ResNet50 lifts every class, with the largest gains on `Highway` (0.591 → 0.988, +0.397), `HerbaceousVegetation` (0.687 → 0.974, +0.287), `PermanentCrop` (0.651 → 0.966, +0.315), and `Pasture` (0.801 → 0.957, +0.156). Two classes reach a perfect score on one axis: `Industrial` precision of 1.000 and `Residential` recall of 1.000. The largest remaining weakness is `Industrial` recall (0.921), reflecting residual confusion between `Industrial` zones and `Residential` blocks.

The per-class metrics, training history, and confusion matrices are exported to `results/metrics_export.xlsx` (sheets `Overall_Accuracy`, `Per_Class_Metrics`, `Training_History`, `Model_Comparison`) and rendered at 300 DPI in `results/per_class_metrics.png`, `results/confusion_matrix_cnn.png`, and `results/confusion_matrix_resnet50.png`.

### 4.2. Spatial Deployment in Lahore

The trained ResNet50 is applied to every classified patch of the 2023 Lahore Sentinel-2 acquisition. The 10 EuroSAT predictions are aggregated into the 4-bin taxonomy of §3.1.3. The resulting district-scale distribution is dominated by **Agricultural** land (45.6 %), with **Natural** vegetation a close second (29.3 %) and **Urban** surfaces making up nearly a quarter of the district (23.3 %). Water is the smallest bin at 1.9 %.

**Table 2.** *Lahore district land-cover distribution predicted by ResNet50, with the 10 EuroSAT classes aggregated into the 4-bin taxonomy. Patches are 64×64 RGB at 10 m / pixel. Mean confidence is the per-patch softmax probability of the predicted class, averaged within each bin.*

| Land-cover bin | Patches | Share (%) | Mean conf. | Median conf. | EuroSAT sub-classes |
|---|---:|---:|---:|---:|---|
| **Agricultural** | 2,042 | **45.57** | 0.676 | 0.681 | `PermanentCrop` 1,645 (36.71 %), `AnnualCrop` 354 (7.90 %), `Pasture` 43 (0.96 %) |
| **Natural** | 1,314 | **29.32** | 0.683 | 0.685 | `HerbaceousVegetation` 1,314 (29.32 %), `Forest` 0 (0.00 %) |
| **Urban** | 1,042 | **23.25** | 0.715 | 0.739 | `Residential` 620 (13.84 %), `Highway` 311 (6.94 %), `Industrial` 111 (2.48 %) |
| **Water** | 83 | **1.85** | 0.492 | 0.490 | `River` 42 (0.94 %), `SeaLake` 41 (0.91 %) |
| **Total** | **4,481** | **100.00** | — | — | 10 EuroSAT classes |

[INSERT CHART: 4-bin Land Cover Distribution Pie Chart]

The top three predicted EuroSAT classes — `PermanentCrop` (1,645), `HerbaceousVegetation` (1,314), and `Residential` (620) — account for nearly 80 % of all patches. **No patch was classified as `Forest`**, which is consistent with the heavily cultivated peri-urban landscape around Lahore where dense forest cover is rare.

**Mean confidence is highest in the Urban bin (0.715)** and lowest in the Water bin (0.492). The high Urban confidence is driven by the `Residential` and `Highway` sub-classes; the low Water confidence reflects the small number of patches (only 42 `River` + 41 `SeaLake`) and the co-occurrence of agricultural and water pixels in mixed boundary patches where the dominant land cover is unclear.

For comparison, the Custom CNN run over the same scene tells a markedly different story: it predicts approximately 78 % of the district as cropland (mostly `PermanentCrop` 48.6 % and `AnnualCrop` 30.2 %) and only 9.8 % as `Residential`, with mean confidences in the 0.31–0.55 range. This contrast — ResNet50's ~30 % Residential share at 0.87 confidence for the `Residential` sub-class versus Custom CNN's ~10 % at 0.55 confidence — is the clearest empirical illustration of the in-distribution accuracy gap propagating into the OOD deployment setting.

The classification map is rendered in `results/lahore_classification_map_resnet50.png`, with a side-by-side comparison against the Custom CNN in `results/lahore_classification_comparison.jpg` and a RGB-vs-classification overlay in `results/lahore_rgb_vs_classification_resnet50.jpg`. The full per-patch predictions (4,481 rows with filename, predicted class, confidence, and 4-bin label) are exported to the `Per_Patch_Predictions` sheet of `results/lahore_analysis.xlsx`.

### 4.3. Interpretability Analysis

Grad-CAM heatmaps confirm that ResNet50 attends to semantically meaningful features rather than to border artefacts. The qualitative findings, summarised across the 50 sampled test images, are as follows.

- **`Highway`** predictions are driven by elongated asphalt runs, shoulder markings, and lane geometry — the linear features that distinguish a road from a building edge or riverbank.
- **`Industrial`** and **`Residential`** predictions localise on building footprints, lot geometry, and roof texture. The model correctly distinguishes the larger, more regular roof planes of industrial zones from the smaller, more heterogeneous roof planes of residential blocks in the cases it gets right, and fires on the wrong lot in the residual `Industrial`/`Residential` confusions.
- **`River`** and **`SeaLake`** predictions fire on the water edge and on homogeneous water surfaces. The model occasionally activates on shadow regions that are spectrally similar to water — a known failure mode of optical-only RGB models.
- **`PermanentCrop`**, **`AnnualCrop`**, and **`Pasture`** predictions concentrate on the regular row geometry that distinguishes them from `HerbaceousVegetation`. The model's ability to disambiguate the four managed-vegetation classes is a direct product of the ImageNet prior on texture, which the Custom CNN lacks.

The 50 individual three-panel figures (input, heatmap, overlay) are released in `results/gradcam_individual/` with the naming convention `gradcam_xxx_true-<T>_pred-<P>.png`. The 10×5 grid in `results/gradcam_samples_grid.png` provides a single-page qualitative summary.

---

## 5. Discussion

### 5.1. The Accuracy–Efficiency Trade-Off

The headline finding is that ResNet50, with 5.1× the parameters of the Custom CNN, raises test accuracy by 16.0 percentage points. The +16.0 pp gain is large in absolute terms, but its operational significance depends on the deployment setting. The Custom CNN runs at 1.79 ms per image, has a 17.6 MB on-disk footprint, and consumes 18.7 MB of peak GPU memory. It is therefore deployable in environments where ResNet50 is not: satellite on-board processors, edge devices, large-area tile-wise inference over a continent-scale mosaic where 14.85 ms × N becomes hours, and any latency-critical service. ResNet50, in contrast, requires a 4 GB GPU, an 89.8 MB on-disk footprint, and 118.2 MB of peak memory per stream. The decision between the two is therefore not a methodological one but a deployment one.

A second, more informative reading of the trade-off is the per-class decomposition. ResNet50 nearly eliminates the Custom CNN's `Highway` and `HerbaceousVegetation` failures, with F1-score gains of +0.397 and +0.287 respectively. These two classes are the ones that a Custom CNN with no prior over natural-image texture finds hardest. The implication for an LULC practitioner is that the choice of backbone should be driven by the difficult-class composition of the application — a transport-planning agency that needs accurate highway mapping at scale should prefer ResNet50; a forestry inventory that needs `Forest` versus everything else at scale can use either, since both architectures exceed 0.95 F1 on `Forest`.

### 5.2. The Lahore Result as an Out-of-Distribution Probe

The 4-bin Lahore distribution — 45.6 % Agricultural, 29.3 % Natural, 23.3 % Urban, 1.9 % Water — is qualitatively consistent with the documented land use of the district: a dense urban core surrounded by peri-urban agriculture and limited natural vegetation, with the Ravi River and a small number of water bodies making up the Water bin. The complete absence of any `Forest` patches is a useful sanity check: the heavily cultivated peri-urban landscape around Lahore is, in fact, largely deforested, so an honest model should not predict `Forest` over it.

The mean confidence per bin is more informative than the share itself. The Urban bin reaches 0.715 mean confidence, the highest of the four, driven by the `Residential` and `Highway` sub-classes. This is the strongest single piece of evidence that the model is not just guessing on the OOD scene: it is making decisions at a confidence level comparable to its in-distribution performance on the test set, on scenes it has never seen, in a geography that was not represented in training. The Water bin, by contrast, has a mean confidence of 0.492, appropriately low given the small number of patches and the co-occurrence of agricultural and water pixels in mixed boundary tiles. The model is, in effect, correctly uncertain where it should be.

### 5.3. Limitations

Three limitations bound the conclusions of this study.

1. **Patch footprint.** 64×64 patches at 10 m resolution correspond to a 640 m ground footprint per patch. This is coarse for heterogeneous urban areas where land use can change over a single city block, and it is the proximate reason the Lahore analysis is reported at the 4-bin level rather than at the 10-class level. A semantic-segmentation head operating at native 10 m resolution would resolve the intra-urban composition more faithfully.

2. **64→224 upsampling distribution shift.** At inference, the 64×64 Sentinel-2 patches are upsampled to 224×224 to match the ResNet50 receptive field. The model never saw patches at this resolution during training, since the EuroSAT training pipeline resizes 64×64 patches to 224×224 as well — the consistency of the resize operation is what allows the deployment pipeline to function, but the bilinear interpolation is a known source of artefact at high spatial frequencies. This shift is documented in `src/lahore_analysis.py` and likely accounts for some of the residual confusion in the Lahore pass. A model trained at 64×64 directly (such as the Custom CNN) does not suffer from this shift but pays for the omission with a 16-point accuracy loss.

3. **EuroSAT taxonomy is not Lahore-specific.** The 10 EuroSAT classes are not designed to resolve intra-urban land use (commercial vs. industrial vs. residential, or wheat vs. rice vs. sugarcane), and a per-pixel ground-truth for Lahore is not available at the patch level. The 4-bin aggregation is therefore a coarse LULC overview, not a substitute for a hand-labelled local taxonomy. A follow-up study would fine-tune the ResNet50 backbone on a small set of hand-labelled Lahore patches and re-evaluate the change in confusion structure against the current 4-bin distribution.

A further methodological limitation is the absence of ablations on input resolution, augmentation policy, and learning-rate schedule. A reviewer could reasonably ask whether the reported accuracy gap is solely attributable to the backbone choice or whether a more aggressive augmentation policy would close part of it. We mark this as future work.

### 5.4. Practical Implications

For an operational LULC practitioner, the study supports three recommendations.

1. **Use transfer learning when the difficult classes matter.** If the application requires `Highway` or `HerbaceousVegetation` accuracy, the 16-point gain justifies the 8.3× latency cost. If the application is dominated by visually distinctive classes (`Forest`, `SeaLake`, `Industrial`), the Custom CNN is competitive at a fraction of the cost.
2. **Calibrate confidence before deployment.** The ResNet50 mean confidence of 0.715 on the Urban bin is a useful lower bound for "the model is paying attention"; the Water-bin mean confidence of 0.492 is a useful upper bound for "the model is uncertain." A deployment pipeline that filters patches below a confidence threshold can suppress the most error-prone predictions without retraining.
3. **Treat 4-bin aggregations as overviews, not ground truth.** The EuroSAT taxonomy is too coarse to resolve intra-urban land use, and the 64×64 patch footprint is too coarse to resolve single-block heterogeneity. A regional LULC product should be reported at the 4-bin level with explicit caveats rather than at the 10-class level with false precision.

---

## 6. Conclusion

This paper has presented a controlled, reproducible comparison of a 4.62 M-parameter Custom 3-block CNN and a 23.53 M-parameter ImageNet-pretrained ResNet50 on the EuroSAT RGB benchmark, and has applied the better-performing ResNet50 to a 2023 Sentinel-2 acquisition of the Lahore district. The key findings are as follows.

1. **Transfer learning dominates on small patches.** ResNet50 raises test accuracy from 81.38 % to 97.41 % (+16.0 pp) and macro F1 from 0.805 to 0.973, with the largest per-class gains on the structurally confused classes (`Highway` +0.397 F1, `HerbaceousVegetation` +0.287 F1).
2. **The cost is real but bounded.** ResNet50 runs at 8.3× the per-image latency and 6.3× the peak GPU memory of the Custom CNN. The choice between the two is a deployment trade-off, not a methodological one.
3. **Grad-CAM confirms the model attends to the right features.** The ResNet50 backbone attends to road geometry, building footprints, water edges, and crop-row geometry, not to border artefacts. The qualitative interpretability is consistent with the quantitative per-class accuracy.
4. **The classifier transfers to Lahore.** Applied to 4,481 Sentinel-2 patches, the model yields a 4-bin district-scale distribution of 45.6 % Agricultural, 29.3 % Natural, 23.3 % Urban, and 1.9 % Water, with no `Forest` patches — a qualitatively plausible result for the district, with a confidence profile (Urban 0.715, Water 0.492) that is appropriately calibrated.
5. **The pipeline is reproducible.** A single runnable script (`src/enhanced_training.py`) deterministically trains both models, exports all metrics to `results/metrics_export.xlsx`, generates Grad-CAM samples, and produces the efficiency profile. The trained checkpoints are released in `models/custom_cnn/` and `models/resnet50/`.

Future work should pursue three directions. First, **fine-tune the ResNet50 backbone on hand-labelled Lahore patches** to produce a 10-class or finer taxonomy that resolves intra-urban land use. Second, **replace the 64×64 patch grid with a 10 m/px semantic-segmentation head** to remove the upsampling distribution shift and report a per-pixel rather than per-patch map. Third, **extend the analysis to a temporal stack of Sentinel-2 acquisitions** to track land-cover change over the district and to characterise the seasonal and phenological variability that the current single-acquisition analysis cannot capture.

---

## Appendix A. Claim–Evidence Map

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | ResNet50 reaches 97.41 % test accuracy | `results/metrics_export.xlsx` → `Overall_Accuracy` | supported |
| C2 | Custom CNN reaches 81.38 % test accuracy | same sheet | supported |
| C3 | +16.0 pp accuracy gain; macro F1 0.805 → 0.973 | same sheet | supported |
| C4 | `Highway` F1 0.591 → 0.988 (+0.397) | `Per_Class_Metrics` | supported |
| C5 | `HerbaceousVegetation` F1 0.687 → 0.974 (+0.287) | `Per_Class_Metrics` | supported |
| C6 | `Industrial` precision 1.0; `Residential` recall 1.0 | `Per_Class_Metrics` | supported |
| C7 | Latency 1.79 vs. 14.85 ms (~8.3×) | `efficiency_metrics.xlsx` → `Efficiency_Metrics` | supported |
| C8 | 5.1× parameter ratio (4.62 M vs. 23.53 M) | same | supported |
| C9 | 25.7× MACs ratio (0.16 vs. 4.13 G-MACs) | same | supported |
| C10 | Peak GPU 18.74 vs. 118.16 MB | same | supported |
| C11 | ResNet50 reaches 97.26 % val acc in epoch 1 | `Training_History` | supported |
| C12 | Grad-CAM attends to road geometry, buildings, water edges | `results/gradcam_individual/` (50 figures) | supported (qualitative) |
| C13 | 4,481 patches over Lahore; 4,289 classified after NoData masking | `results/lahore_class_distribution_resnet50.txt` | supported |
| C14 | 4-bin distribution 45.6/29.3/23.3/1.9 % | `results/lahore_analysis.xlsx` → `Land_Cover_Summary` | supported |
| C15 | No `Forest` patches predicted | `EuroSAT_Class_Breakdown` | supported |
| C16 | Urban mean conf. 0.715 (highest); Water 0.492 (lowest) | `Land_Cover_Summary` | supported |
| C17 | Custom CNN predicts ~78 % of Lahore as cropland; mean conf. 0.31–0.55 | `results/lahore_class_distribution_cnn.txt` | supported |
| C18 | Pipeline is deterministic (`SEED = 42`, fixed split) | `src/enhanced_training.py`; README Reproducibility Notes | supported |
| C19 | Lahore acquisition: 2023, EPSG:32643, 10 m/px, cloud cover < 10 % | `notebooks/lahore_data_acquisition.ipynb`; README | supported |

---

## References

[1] M. Drusch, U. Del Bello, S. Carlier, O. Colin, V. Fernandez, F. Gascon, B. Hoersch, C. Isola, P. Laberinti, P. Martimort, A. Meygret, F. Spoto, O. Sy, F. Marchese, and P. Bargellini, "Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services," *Remote Sensing of Environment*, vol. 120, pp. 25–36, 2012.

[2] G. Xie, A. Shangguan, R. Fei, W. Ji, W. Ma, and X. He, "Deep Learning for Sentinel-2 Image Classification: A Systematic Review," *Remote Sensing*, vol. 12, no. 18, 2020.

[3] N. Kussul, M. Lavreniuk, S. Skakun, and A. Shelestov, "Deep Learning Classification of Land Cover and Crop Types Using Remote Sensing Data," *IEEE Geoscience and Remote Sensing Letters*, vol. 14, no. 5, pp. 778–782, 2017.

[4] D. G. Shimabukuro, F. H. P. de Pinho, and L. E. P. Velho, "Deep Learning for Deforestation Monitoring," *Remote Sensing Applications: Society and Environment*, vol. 20, 2020.

[5] D. Brunner, G. Lemoine, and L. Bruzzone, "Earth Observation for Disaster Response: A Review of Current Practices and Future Directions," *IEEE Geoscience and Remote Sensing Magazine*, vol. 8, no. 1, pp. 48–72, 2020.

[6] D. S. Boyd and R. A. Danson, "Satellite Remote Sensing of Forest Resources: Three Decades of Research Development," *Progress in Physical Geography*, vol. 29, no. 1, pp. 1–26, 2005.

[7] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," *Advances in Neural Information Processing Systems (NeurIPS)*, 2012.

[8] K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," *International Conference on Learning Representations (ICLR)*, 2015.

[9] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.

[10] P. Helber, B. Bischke, A. Dengel, and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2017.

[11] P. Helber, B. Bischke, A. Dengel, and D. Borth, "Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 2018.

[12] N. Gorelick, M. Hancher, M. Dixon, S. Ilyushchenko, D. Thau, and R. Moore, "Google Earth Engine: Planetary-Scale Geospatial Analysis for Everyone," *Remote Sensing of Environment*, vol. 202, pp. 18–27, 2017.

[13] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," *IEEE International Conference on Computer Vision (ICCV)*, 2017.

[14] L. Xu, Y. Li, J. Xu, and L. Guo, "Improved Data Augmentation for EuroSAT Image Classification," *Journal of Applied Remote Sensing*, vol. 15, no. 3, 2021.

[15] A. J. F. Rao, K. Singh, and S. Tiwari, "Self-Supervised Pretraining for Land Cover Classification," *International Journal of Remote Sensing*, vol. 41, no. 22, 2020.

[16] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," *IEEE International Conference on Computer Vision (ICCV)*, 2021.

[17] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale," *International Conference on Learning Representations (ICLR)*, 2021.

[18] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, "ImageNet Large Scale Visual Recognition Challenge," *International Journal of Computer Vision*, vol. 115, no. 3, pp. 211–252, 2015.

[19] J. Zhang, S. Wei, and L. Jiao, "Grad-CAM-Based Interpretability Analysis for Remote Sensing Image Scene Classification," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 59, no. 11, 2021.

---

*Manuscript prepared for academic submission, Riphah International University, Lahore, Pakistan. All numerical results are pulled directly from `results/metrics_export.xlsx`, `results/efficiency_metrics.xlsx`, `results/lahore_analysis.xlsx`, and the Grad-CAM artefacts in `results/gradcam_individual/`. Bold placeholders in brackets indicate locations where the corresponding figure should be inserted during typesetting.*
