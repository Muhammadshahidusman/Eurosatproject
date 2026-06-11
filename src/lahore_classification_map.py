"""
Lahore district pixel-wise land-cover classification map (CNN vs ResNet50).

Loads the full-resolution Lahore Sentinel-2 acquisition, runs both trained
EuroSAT classifiers (custom 3-block CNN at native 64x64, and ResNet50 at
224x224) over every 64x64 tile, and assembles colour-coded per-pixel
classification maps. Each of the 10 EuroSAT classes is rendered in a
distinct colour; tiles outside the scene (NaN in the source TIF) are
masked in white.

Outputs (in `results/`):
    lahore_classification_map_resnet50.jpg  # ResNet50 colour map
    lahore_classification_map_resnet50.png  # lossless copy
    lahore_classification_map_cnn.jpg       # Custom CNN colour map
    lahore_classification_map_cnn.png       # lossless copy
    lahore_rgb_vs_classification.jpg        # side-by-side RGB vs ResNet50
    lahore_classification_comparison.jpg    # CNN vs ResNet50 side-by-side
    lahore_class_distribution_resnet50.txt  # per-class report (ResNet50)
    lahore_class_distribution_cnn.txt       # per-class report (CNN)

Run from the project root:
    uv run python src/lahore_classification_map.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIF_PATH = PROJECT_ROOT / "data" / "Lahore_Dataset" / "lahore_rgb_2023.tif"
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "custom_cnn" / "eurosat_cnn_final.pth"
RESNET_MODEL_PATH = PROJECT_ROOT / "models" / "resnet50" / "eurosat_resnet50_final.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EuroSAT 10-class taxonomy (must match the training-time order).
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Distinct, qualitative colour for every EuroSAT class. The NODATA entry
# is reserved for the masked-out boundary of the Lahore district polygon.
CLASS_COLORS = {
    "AnnualCrop":           "#F0E68C",  # khaki
    "Forest":               "#228B22",  # forest green
    "HerbaceousVegetation": "#90EE90",  # light green
    "Highway":              "#404040",  # dark grey
    "Industrial":           "#B22222",  # firebrick
    "Pasture":              "#9ACD32",  # yellow-green
    "PermanentCrop":        "#FF8C00",  # dark orange
    "Residential":          "#9370DB",  # medium purple
    "River":                "#1E90FF",  # dodger blue
    "SeaLake":              "#000080",  # navy
}
NODATA_COLOR = "#FFFFFF"

# 10-m Sentinel-2 acquisition. Tiles match the pre-extracted patches.
PATCH_SIZE = 64
# Sentinel-2 surface reflectance 0..~3000; we clip to 0..3000 then rescale.
REFL_MAX = 3000.0

# Per-model preprocessing. ResNet50 was trained at 224x224, so 64x64 tiles
# are upsampled (a known distribution shift). The custom CNN was trained
# at the native 64x64 tile size and must NOT be upsampled -- otherwise the
# classifier's 128*8*8 = 8192-dim input would not match.
MEAN = [0.3444, 0.3803, 0.4078]
STD = [0.2034, 0.1365, 0.1148]
RESNET_INFER_SIZE = (224, 224)
CNN_INFER_SIZE = (64, 64)
BATCH_SIZE = 64

# Sentinel value for the nodata class index in the integer class map.
NODATA_LABEL = 255

# Publication-quality matplotlib defaults.
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Custom CNN architecture (must match enhanced_training.py:EuroSATCNN)
# ---------------------------------------------------------------------------
class EuroSATCNN(nn.Module):
    """Custom 3-block CNN: Conv-BN-ReLU x2 -> MaxPool -> Dropout2d, x3.

    Trained at 64x64 (see src/enhanced_training.py).  Re-instantiated here
    rather than imported so the script is self-contained.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(in_ch: int, out_ch: int, p: float = 0.25) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(p),
            )

        self.features = nn.Sequential(
            block(3, 32),    # 64x64 -> 32x32
            block(32, 64),   # 32x32 -> 16x16
            block(64, 128),  # 16x16 ->  8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256),         nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# TIF loading
# ---------------------------------------------------------------------------
def load_tif_as_uint8(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the 3-band Lahore TIF and return (rgb_uint8 HxWx3, valid_mask HxW)."""
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32)        # (3, H, W)
    # NaN regions in the source TIF are outside the Lahore district polygon.
    finite = np.isfinite(arr).all(axis=0)                    # (H, W) bool
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr / REFL_MAX * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb = np.transpose(arr, (1, 2, 0))                       # (H, W, 3)
    return rgb, finite


# ---------------------------------------------------------------------------
# Tiled dataset
# ---------------------------------------------------------------------------
class TiledGeoTifDataset(Dataset):
    """Yield every 64x64 tile from a TIF, in raster-row-major order.

    Tiles whose mean intensity is below ``NODATA_MEAN_THRESHOLD`` (i.e. the
    scene boundary) are kept in the dataset but flagged as ``is_nodata`` so
    the caller can record their position without running inference on them.

    The model-specific ``infer_size`` controls the resize applied to the
    tile before tensorisation.  64x64 (custom CNN) keeps the native
    Sentinel-2 resolution; 224x224 (ResNet50) upsamples.
    """

    def __init__(self, rgb: np.ndarray, valid_mask: np.ndarray,
                 patch_size: int = PATCH_SIZE,
                 infer_size: tuple[int, int] = CNN_INFER_SIZE,
                 transform=None):
        self.rgb = rgb
        self.valid_mask = valid_mask
        self.patch_size = patch_size
        self.infer_size = infer_size
        self.transform = transform
        h, w, _ = rgb.shape
        self.n_rows = h // patch_size
        self.n_cols = w // patch_size
        self.length = self.n_rows * self.n_cols

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        i, j = divmod(idx, self.n_cols)
        y0, x0 = i * self.patch_size, j * self.patch_size
        tile = self.rgb[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
        mask = self.valid_mask[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
        # Tile-level nodata: more than half of the tile is outside the scene.
        is_nodata = bool((~mask).sum() > (self.patch_size * self.patch_size) // 2)
        if is_nodata:
            # Return a zero tensor at the expected inference size; the
            # caller will skip model inference for this entry.
            tensor = torch.zeros(3, *self.infer_size, dtype=torch.float32)
        else:
            from PIL import Image
            tensor = self.transform(Image.fromarray(tile))
        return tensor, idx, is_nodata


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_resnet50(ckpt_path: Path, num_classes: int) -> nn.Module:
    """Build ResNet50 with a fresh 10-class head and load the trained weights."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def load_custom_cnn(ckpt_path: Path, num_classes: int) -> nn.Module:
    """Build the custom EuroSATCNN and load the trained weights."""
    model = EuroSATCNN(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_tiled_inference(model: nn.Module, dataset: TiledGeoTifDataset
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_map HxW uint8, confidence HxW float32) at TILE resolution.

    ``class_map`` has one value per 64x64 tile block, so its shape is
    ``(n_rows, n_cols)``; multiply both axes by ``PATCH_SIZE`` to map back
    to the original raster pixels.
    """
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            torch.tensor([x[1] for x in b], dtype=torch.long),
            torch.tensor([x[2] for x in b], dtype=torch.bool),
        ),
    )
    softmax = nn.Softmax(dim=1)
    class_map = np.full((dataset.n_rows, dataset.n_cols), NODATA_LABEL, dtype=np.uint8)
    conf_map = np.zeros((dataset.n_rows, dataset.n_cols), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs, is_nodata in tqdm(loader, desc="lahore-tiled-inference"):
            valid_mask = ~is_nodata
            if valid_mask.any():
                imgs_v = imgs[valid_mask].to(DEVICE)
                logits = model(imgs_v)
                probs = softmax(logits)
                confs, preds = probs.max(dim=1)
                preds = preds.cpu().numpy().astype(np.uint8)
                confs = confs.cpu().numpy().astype(np.float32)
                # Map predictions back to their tile indices in the batch.
                v_pos = 0
                for k, idx in enumerate(idxs.tolist()):
                    if is_nodata[k]:
                        continue
                    i, j = divmod(idx, dataset.n_cols)
                    class_map[i, j] = preds[v_pos]
                    conf_map[i, j] = confs[v_pos]
                    v_pos += 1
    return class_map, conf_map


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_class_map(class_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Build a (H, W, 3) uint8 RGB image of the classification, with nodata white.

    ``class_map`` is at tile resolution; it is tiled up to pixel resolution
    so each 64x64 block shares one colour.  The nodata regions are painted
    white at tile resolution (any tile that contains more than half
    nodata pixels) and the kron upsample preserves that mask.
    """
    h, w = valid_mask.shape
    tile_h, tile_w = class_map.shape
    tile_rgb = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    for cls_idx, cls_name in enumerate(EUROSAT_CLASSES):
        hexcol = CLASS_COLORS[cls_name].lstrip("#")
        rgb = tuple(int(hexcol[k:k + 2], 16) for k in (0, 2, 4))
        sel = class_map == cls_idx
        tile_rgb[sel] = rgb
    tile_rgb[class_map == NODATA_LABEL] = (255, 255, 255)
    pixel_rgb = np.kron(tile_rgb, np.ones((PATCH_SIZE, PATCH_SIZE, 1), dtype=np.uint8))
    pixel_rgb = pixel_rgb[:h, :w]
    return pixel_rgb


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def build_legend_handles() -> list[Patch]:
    return [Patch(facecolor=CLASS_COLORS[c], edgecolor="black", label=c)
            for c in EUROSAT_CLASSES] + [
        Patch(facecolor=NODATA_COLOR, edgecolor="black", label="No-data / outside scene")
    ]


def save_class_map_jpg(class_rgb: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(class_rgb)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Column (px)")
    ax.set_ylabel("Row (px)")
    ax.legend(handles=build_legend_handles(),
              loc="lower left", bbox_to_anchor=(1.01, 0.0),
              frameon=False, fontsize=9, title="Land-cover class")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_side_by_side(rgb_uint8: np.ndarray, class_rgb: np.ndarray,
                      out_path: Path, right_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(rgb_uint8)
    axes[0].set_title("Original RGB (Sentinel-2, 2023)", fontweight="bold")
    axes[0].set_axis_off()

    axes[1].imshow(class_rgb)
    axes[1].set_title(right_title, fontweight="bold")
    axes[1].set_axis_off()

    fig.suptitle("Lahore District — RGB vs. Pixel-wise Classification",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.legend(handles=build_legend_handles(),
               loc="lower center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_two_model_comparison(rgb_uint8: np.ndarray,
                              cnn_rgb: np.ndarray, resnet_rgb: np.ndarray,
                              out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    axes[0].imshow(rgb_uint8)
    axes[0].set_title("Original RGB (Sentinel-2, 2023)", fontweight="bold")
    axes[0].set_axis_off()

    axes[1].imshow(cnn_rgb)
    axes[1].set_title("Custom 3-block CNN (64x64)", fontweight="bold")
    axes[1].set_axis_off()

    axes[2].imshow(resnet_rgb)
    axes[2].set_title("ResNet50 (224x224, transfer learning)", fontweight="bold")
    axes[2].set_axis_off()

    fig.suptitle("Lahore District — CNN vs ResNet50 Land-Cover Classification",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.legend(handles=build_legend_handles(),
               loc="lower center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_distribution_report(class_map: np.ndarray, conf_map: np.ndarray,
                              valid_mask: np.ndarray, model_name: str,
                              out_path: Path) -> dict:
    """Write a human-readable distribution report and return a summary dict."""
    tile_total = class_map.size
    nodata_tiles = int((class_map == NODATA_LABEL).sum())
    classified_tiles = tile_total - nodata_tiles
    full_h, full_w = valid_mask.shape
    full_pixels = int(valid_mask.sum())

    lines = [f"Lahore District Classification — {model_name} / EuroSAT (10 classes)",
             "=" * 68, ""]
    lines.append(f"Source TIF          : {TIF_PATH.name}")
    lines.append(f"Tile size           : {PATCH_SIZE}x{PATCH_SIZE} px @ 10 m")
    lines.append(f"Tile grid           : {class_map.shape[0]} rows x {class_map.shape[1]} cols "
                 f"({tile_total:,} tiles)")
    lines.append(f"  - nodata tiles    : {nodata_tiles:,}")
    lines.append(f"  - classified tiles: {classified_tiles:,}")
    lines.append(f"Original raster     : {full_h} x {full_w} px "
                 f"({full_pixels:,} valid pixels)")
    lines.append("")
    lines.append("Per-class distribution (pixel share of the valid Lahore scene):")
    lines.append("-" * 68)
    lines.append(f"  {'Class':<25} {'Tiles':>8} {'Share %':>10} {'Mean conf.':>12}")
    summary = {}
    for cls_idx, cls_name in enumerate(EUROSAT_CLASSES):
        sel = class_map == cls_idx
        n_tiles = int(sel.sum())
        n_pixels = n_tiles * PATCH_SIZE * PATCH_SIZE
        n_pixels = min(n_pixels, full_pixels)
        share = n_pixels / full_pixels * 100.0 if full_pixels else 0.0
        mean_conf = float(conf_map[sel].mean()) if n_tiles else 0.0
        summary[cls_name] = {"tiles": n_tiles, "pixels": n_pixels,
                             "share_percent": share, "mean_confidence": mean_conf}
        lines.append(f"  {cls_name:<25} {n_tiles:>8d} {share:>9.2f}% {mean_conf:>12.3f}")
    lines.append("-" * 68)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def print_model_summary(name: str, summary: dict, elapsed: float, n_classified: int) -> None:
    print(f"\n--- {name} (per-class summary, top 5 by tile count) ---")
    top5 = sorted(summary.items(), key=lambda kv: -kv[1]["tiles"])[:5]
    for cname, s in top5:
        print(f"  {cname:<25} {s['tiles']:>5d} tiles  "
              f"{s['share_percent']:>5.2f}%  conf={s['mean_confidence']:.3f}")
    print(f"  Wall-clock       : {elapsed:.1f}s "
          f"({elapsed / max(n_classified, 1) * 1000:.2f} ms / classified tile)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_one_model(model_name: str, model: nn.Module, model_pretty: str,
                  rgb: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """Run tiled inference for one model and return everything we need to plot/report."""
    infer_size = CNN_INFER_SIZE if model_name == "cnn" else RESNET_INFER_SIZE
    transform = transforms.Compose([
        transforms.Resize(infer_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    dataset = TiledGeoTifDataset(rgb, valid_mask, PATCH_SIZE, infer_size, transform)
    print(f"  Tile grid     : {dataset.n_rows} rows x {dataset.n_cols} cols "
          f"({dataset.length:,} tiles)")
    print(f"  Input size    : {infer_size[0]}x{infer_size[1]}")

    t0 = time.time()
    class_map, conf_map = run_tiled_inference(model, dataset)
    elapsed = time.time() - t0
    n_classified = int((class_map != NODATA_LABEL).sum())
    print(f"  Inference     : {elapsed:.1f}s "
          f"({elapsed / max(n_classified, 1) * 1000:.2f} ms / classified tile, "
          f"{n_classified:,}/{dataset.length:,} tiles)")

    class_rgb = render_class_map(class_map, valid_mask)
    # Per-model colour map.
    title = (f"Lahore District — {model_pretty} Classification Map\n"
             f"Sentinel-2 (2023), tile size {PATCH_SIZE}x{PATCH_SIZE} px @ 10 m, "
             f"inference at {infer_size[0]}x{infer_size[1]}")
    save_class_map_jpg(class_rgb, RESULTS_DIR / f"lahore_classification_map_{model_name}.jpg", title)
    plt.imsave(RESULTS_DIR / f"lahore_classification_map_{model_name}.png", class_rgb, dpi=300)
    summary = write_distribution_report(
        class_map, conf_map, valid_mask, model_pretty,
        RESULTS_DIR / f"lahore_class_distribution_{model_name}.txt",
    )
    print_model_summary(model_pretty, summary, elapsed, n_classified)
    return class_rgb, class_map, summary, elapsed


def main() -> None:
    if not TIF_PATH.exists():
        raise FileNotFoundError(f"Lahore TIF not found: {TIF_PATH}")
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Custom CNN checkpoint not found: {CNN_MODEL_PATH}")
    if not RESNET_MODEL_PATH.exists():
        raise FileNotFoundError(f"ResNet50 checkpoint not found: {RESNET_MODEL_PATH}")

    print(f"Device        : {DEVICE}")
    print(f"Source TIF    : {TIF_PATH.name}")

    rgb, valid_mask = load_tif_as_uint8(TIF_PATH)
    print(f"Raster shape  : {rgb.shape[0]} x {rgb.shape[1]} px "
          f"({valid_mask.sum() / valid_mask.size * 100:.1f}% valid)")

    # ----- ResNet50 -----
    print(f"\n[1/2] ResNet50 — {RESNET_MODEL_PATH.name}")
    resnet_model = load_resnet50(RESNET_MODEL_PATH, len(EUROSAT_CLASSES))
    print(f"  Model params: {sum(p.numel() for p in resnet_model.parameters()):,}")
    resnet_rgb, resnet_class_map, resnet_summary, resnet_t = run_one_model(
        "resnet50", resnet_model, "ResNet50", rgb, valid_mask,
    )
    # Per-model side-by-side.
    save_side_by_side(rgb, resnet_rgb,
                      RESULTS_DIR / "lahore_rgb_vs_classification_resnet50.jpg",
                      "ResNet50 Classification")

    # ----- Custom CNN -----
    print(f"\n[2/2] Custom CNN — {CNN_MODEL_PATH.name}")
    cnn_model = load_custom_cnn(CNN_MODEL_PATH, len(EUROSAT_CLASSES))
    print(f"  Model params: {sum(p.numel() for p in cnn_model.parameters()):,}")
    cnn_rgb, cnn_class_map, cnn_summary, cnn_t = run_one_model(
        "cnn", cnn_model, "Custom 3-block CNN", rgb, valid_mask,
    )
    # Per-model side-by-side.
    save_side_by_side(rgb, cnn_rgb,
                      RESULTS_DIR / "lahore_rgb_vs_classification_cnn.jpg",
                      "Custom CNN Classification")

    # ----- Three-way comparison (RGB / CNN / ResNet50) -----
    save_two_model_comparison(
        rgb, cnn_rgb, resnet_rgb,
        RESULTS_DIR / "lahore_classification_comparison.jpg",
    )

    # ----- Cross-model agreement summary -----
    both = (resnet_class_map != NODATA_LABEL) & (cnn_class_map != NODATA_LABEL)
    n_both = int(both.sum())
    if n_both:
        agree = int((resnet_class_map[both] == cnn_class_map[both]).sum())
        print("\n--- Cross-model agreement ---")
        print(f"  Tiles classified by both  : {n_both:,}")
        print(f"  Tiles with same prediction: {agree:,} ({agree / n_both * 100:.1f}%)")

    print("\nSaved artefacts:")
    for fname in (
        "lahore_classification_map_resnet50.jpg",
        "lahore_classification_map_resnet50.png",
        "lahore_classification_map_cnn.jpg",
        "lahore_classification_map_cnn.png",
        "lahore_rgb_vs_classification_resnet50.jpg",
        "lahore_rgb_vs_classification_cnn.jpg",
        "lahore_classification_comparison.jpg",
        "lahore_class_distribution_resnet50.txt",
        "lahore_class_distribution_cnn.txt",
    ):
        f = RESULTS_DIR / fname
        size = f"{f.stat().st_size / 1e6:.2f} MB" if f.exists() else "MISSING"
        print(f"  results/{fname}  ({size})")


if __name__ == "__main__":
    main()
