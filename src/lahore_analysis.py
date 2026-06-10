"""
Lahore district land-cover analysis.

Applies the trained ResNet50 EuroSAT model to the 4,481 64x64 RGB
patches extracted from the Lahore Sentinel-2 acquisition and
summarises the predictions in a four-bin land-cover taxonomy.

Outputs:
    results/lahore_land_cover_pie.png   # 300 DPI pie chart
    results/lahore_analysis.xlsx        # Multi-sheet workbook

Run from the project root:
    uv run python src/lahore_analysis.py
"""

from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATCHES_DIR = PROJECT_ROOT / "data" / "Lahore_Dataset" / "patches"
MODEL_PATH = PROJECT_ROOT / "models" / "resnet50" / "eurosat_resnet50_final.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EuroSAT 10-class taxonomy (must match the training-time order).
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Mapping from the 10 EuroSAT classes to a coarser 4-bin land-cover taxonomy.
# This is the level at which the model can be expected to be meaningful
# because the EuroSAT classes themselves were never defined for Lahore.
LAHORE_BINS = {
    # Urban: built-up land and transport corridors
    "Urban":         {"Residential", "Industrial", "Highway"},
    # Agricultural: cropland and managed vegetation
    "Agricultural":  {"AnnualCrop", "PermanentCrop", "Pasture"},
    # Water: rivers and standing water
    "Water":         {"River", "SeaLake"},
    # Natural: unmanaged vegetation
    "Natural":       {"Forest", "HerbaceousVegetation"},
}
BIN_ORDER = ["Urban", "Agricultural", "Water", "Natural"]
BIN_COLORS = {"Urban": "#d62728", "Agricultural": "#2ca02c", "Water": "#1f77b4", "Natural": "#9467bd"}

# Inference preprocessing. ResNet50 was trained at 224x224, so the 64x64
# patches are upsampled. This is a known distribution shift documented
# in the paper (see Discussion, "limitations").
INFER_SIZE = (224, 224)
MEAN = [0.3444, 0.3803, 0.4078]
STD = [0.2034, 0.1365, 0.1148]
BATCH_SIZE = 64

# Publication-quality defaults.
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Reads all *.jpg patches in a directory."""

    def __init__(self, patch_dir: Path, transform):
        self.transform = transform
        self.files = sorted(patch_dir.glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(f"No .jpg patches found in {patch_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path.name


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model() -> nn.Module:
    """Load the trained ResNet50 weights produced by enhanced_training.py."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EUROSAT_CLASSES))
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(model: nn.Module, loader: DataLoader) -> pd.DataFrame:
    """Run the model over every patch and return a per-patch prediction table."""
    rows: list[dict] = []
    softmax = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for imgs, names in tqdm(loader, desc="lahore-inference"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = softmax(logits)
            confs, preds = probs.max(dim=1)
            for name, pred, conf in zip(names, preds.cpu().numpy(), confs.cpu().numpy()):
                rows.append({
                    "filename": name,
                    "predicted_class": EUROSAT_CLASSES[int(pred)],
                    "confidence": float(conf),
                })
    return pd.DataFrame(rows)


def map_to_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 10 EuroSAT classes into the 4-bin district taxonomy."""
    cls_to_bin = {c: b for b, members in LAHORE_BINS.items() for c in members}
    df = df.copy()
    df["land_cover_bin"] = df["predicted_class"].map(cls_to_bin)
    unknown = df["land_cover_bin"].isna().sum()
    if unknown:
        print(f"  Warning: {unknown} patches mapped to no bin (unexpected).")
        df["land_cover_bin"] = df["land_cover_bin"].fillna("Unknown")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pie(df: pd.DataFrame, path: Path) -> None:
    """Donut chart of the 4-bin land-cover distribution (annotated %)."""
    counts = df["land_cover_bin"].value_counts().reindex(BIN_ORDER, fill_value=0)
    total = int(counts.sum())
    pcts = counts / total * 100
    labels = [f"{name}\n{c:,} patches ({p:.1f}%)"
              for name, c, p in zip(counts.index, counts.values, pcts.values)]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(
        counts.values,
        colors=[BIN_COLORS[b] for b in counts.index],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    # Annotate percentages on the wedges themselves.
    for w, pct in zip(wedges, pcts.values):
        ang = (w.theta2 + w.theta1) / 2.0
        x = 0.85 * np.cos(np.deg2rad(ang))
        y = 0.85 * np.sin(np.deg2rad(ang))
        ax.text(x, y, f"{pct:.1f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    ax.set_title(
        f"Lahore District Land Cover\nResNet50 Predictions on {total:,} Sentinel-2 Patches",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_class_bar(df: pd.DataFrame, path: Path) -> None:
    """Bar chart of the 10 EuroSAT class shares (finer-grained view)."""
    counts = df["predicted_class"].value_counts().reindex(EUROSAT_CLASSES, fill_value=0)
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    bin_color = []
    for c in counts.index:
        for b, members in LAHORE_BINS.items():
            if c in members:
                bin_color.append(BIN_COLORS[b])
                break
    bars = ax.bar(counts.index, pcts.values, color=bin_color, edgecolor="white")
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Share of patches (%)")
    ax.set_title("Lahore District — EuroSAT Class Distribution", fontweight="bold")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, max(pcts.values) * 1.15)

    from matplotlib.patches import Patch
    legend = [Patch(color=BIN_COLORS[b], label=b) for b in BIN_ORDER]
    ax.legend(handles=legend, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------
def export_excel(per_patch: pd.DataFrame, out_path: Path) -> None:
    """Write a 4-sheet workbook: patches, summary, bins, confidence."""
    bin_df = (per_patch.groupby("land_cover_bin")
              .agg(patches=("filename", "count"),
                   mean_confidence=("confidence", "mean"),
                   median_confidence=("confidence", "median"))
              .reindex(BIN_ORDER, fill_value=0)
              .reset_index()
              .rename(columns={"land_cover_bin": "Land_Cover_Bin"}))
    bin_df["share_percent"] = bin_df["patches"] / bin_df["patches"].sum() * 100

    class_df = (per_patch.groupby("predicted_class")
                .agg(patches=("filename", "count"),
                     mean_confidence=("confidence", "mean"))
                .reindex(EUROSAT_CLASSES, fill_value=0)
                .reset_index()
                .rename(columns={"predicted_class": "EuroSAT_Class"}))
    class_df["land_cover_bin"] = class_df["EuroSAT_Class"].map(
        {c: b for b, members in LAHORE_BINS.items() for c in members}
    )
    class_df["share_percent"] = class_df["patches"] / class_df["patches"].sum() * 100

    conf_df = per_patch["confidence"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame("confidence")

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        bin_df.to_excel(xw, sheet_name="Land_Cover_Summary", index=False)
        class_df.to_excel(xw, sheet_name="EuroSAT_Class_Breakdown", index=False)
        per_patch.to_excel(xw, sheet_name="Per_Patch_Predictions", index=False)
        conf_df.to_excel(xw, sheet_name="Confidence_Statistics")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not PATCHES_DIR.exists():
        raise FileNotFoundError(f"Lahore patches directory not found: {PATCHES_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained ResNet50 checkpoint not found at {MODEL_PATH}. "
            "Run `uv run python src/enhanced_training.py` first."
        )

    print(f"Device       : {DEVICE}")
    print(f"Model        : {MODEL_PATH.name}")
    print(f"Patches dir  : {PATCHES_DIR}")

    transform = transforms.Compose([
        transforms.Resize(INFER_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    dataset = PatchDataset(PATCHES_DIR, transform)
    print(f"Patch count  : {len(dataset):,}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=DEVICE.type == "cuda", collate_fn=lambda batch: (
                            torch.stack([b[0] for b in batch]),
                            [b[1] for b in batch],
                        ))

    model = load_model()
    print(f"Model params : {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    predictions = run_inference(model, loader)
    elapsed = time.time() - t0
    print(f"Inference    : {elapsed:.1f}s ({elapsed / len(dataset) * 1000:.2f} ms / patch)")

    predictions = map_to_bins(predictions)

    # ---- Plots ----
    plot_pie(predictions, RESULTS_DIR / "lahore_land_cover_pie.png")
    plot_class_bar(predictions, RESULTS_DIR / "lahore_class_distribution.png")
    print(f"Saved plots  : {RESULTS_DIR}/lahore_land_cover_pie.png, lahore_class_distribution.png")

    # ---- Excel ----
    export_excel(predictions, RESULTS_DIR / "lahore_analysis.xlsx")
    print(f"Saved Excel  : {RESULTS_DIR}/lahore_analysis.xlsx")

    # ---- Console summary ----
    print("\n--- Land-Cover Distribution (4-bin) ---")
    counts = predictions["land_cover_bin"].value_counts().reindex(BIN_ORDER, fill_value=0)
    for b in BIN_ORDER:
        n = int(counts[b])
        pct = n / counts.sum() * 100
        print(f"  {b:<14} {n:5d} patches  ({pct:5.2f}%)")

    print("\n--- Top EuroSAT Classes ---")
    for c, n in predictions["predicted_class"].value_counts().head(5).items():
        print(f"  {c:<25} {n:5d}  ({n / len(predictions) * 100:5.2f}%)")


if __name__ == "__main__":
    main()
