"""
Enhanced training pipeline for EuroSAT RGB classification.

Trains two architectures (custom 3-block CNN and ResNet50 with transfer
learning) and produces publication-ready artifacts:

    results/
        training_curves.png          # Loss + accuracy, both models
        confusion_matrix_cnn.png
        confusion_matrix_resnet50.png
        per_class_metrics.png
        metrics_export.xlsx          # 4 sheets
if th
    models/
        custom_cnn/eurosat_cnn_final.pth
        resnet50/eurosat_resnet50_final.pth

Run from the project root:
    uv run python src/enhanced_training.py
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Eurosat_Dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "custom_cnn").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "resnet50").mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_R  IO
NUM_EPOCHS_CNN = 10   # Custom CNN at 64x64
NUM_EPOCHS_RESNET = 5  # ResNet50 at 224x224 — heavy; 5 epochs is sufficient
LR_CNN = 1e-3
LR_RESNET = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE_CNN = 64
BATCH_SIZE_RESNET = 16  # smaller to fit on 4 GB Quadro P2000 at 224x224

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# EuroSAT RGB channel statistics (pre-computed from the full dataset).
MEAN = [0.3444, 0.3803, 0.4078]
STD = [0.2034, 0.1365, 0.1148]

# Publication-quality matplotlib defaults.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: deterministic cuDNN can be 5-10x slower on ConvNets because it
    # disables the autotuner. We seed carefully but allow benchmark mode
    # so cuDNN picks the fastest kernels.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FlatImageDataset(Dataset):
    """Reads a flat folder of JPEG files where the class is the filename prefix."""

    def __init__(self, data_dir: Path, class_names: list[str], transform=None):
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.samples: list[tuple[str, int]] = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            # AnnualCrop_123.jpg -> AnnualCrop
            prefix = "_".join(fname.replace(".jpg", "").split("_")[:-1])
            if prefix in self.class_to_idx:
                self.samples.append(
                    (str(data_dir / fname), self.class_to_idx[prefix])
                )
        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loaders(img_size: int, batch_size: int):
    """Build train/val/test DataLoaders at the requested resolution."""
    train_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    full_train = FlatImageDataset(DATA_DIR, CLASS_NAMES, transform=train_t)
    full_eval = FlatImageDataset(DATA_DIR, CLASS_NAMES, transform=eval_t)

    total = len(full_train)
    n_train = int(TRAIN_RATIO * total)
    n_val = int(VAL_RATIO * total)
    indices = list(range(total))
    rng = np.random.RandomState(SEED)
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # num_workers=0 on Windows: the multiprocessing spawn loader can
    # collide with CUDA DLL initialisation on machines with a small
    # paging file, surfacing as WinError 1455 in worker subprocesses.
    num_workers = 0 if os.name == "nt" else 2
    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=DEVICE.type == "cuda",
    )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    loaders = {
        "train": DataLoader(Subset(full_train, train_idx),
                            shuffle=True,  **loader_kwargs),
        "val":   DataLoader(Subset(full_eval,  val_idx),
                            shuffle=False, **loader_kwargs),
        "test":  DataLoader(Subset(full_eval,  test_idx),
                            shuffle=False, **loader_kwargs),
    }
    sizes = {"train": n_train, "val": n_val, "test": total - n_train - n_val}
    return loaders, sizes


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class EuroSATCNN(nn.Module):
    """Custom 3-block CNN: Conv-BN-ReLU x2 -> MaxPool -> Dropout2d, x3."""

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


def build_resnet50(num_classes: int, weights_local: Path | None = None) -> nn.Module:
    """ResNet50 with the final FC replaced for 10-way classification."""
    if weights_local and weights_local.exists():
        model = models.resnet50(weights=None)
        model.load_state_dict(torch.load(weights_local, map_location="cpu"))
        print(f"  Loaded ImageNet weights from {weights_local}")
    else:
        # Fall back to the (slow) download if no local weights are present.
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, phase: str):
    """Run a single epoch in train or eval mode. Returns (loss, accuracy)."""
    is_train = phase == "train"
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=phase, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    loaders: dict,
    num_epochs: int,
    lr: float,
    name: str,
    ckpt_path: Path,
) -> dict:
    """Full training loop with best-checkpoint tracking and LR scheduler."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n=== Training {name} on {DEVICE} ===")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, "train")
        vl_loss, vl_acc = run_epoch(model, loaders["val"], criterion, None, "val")
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path.with_name("best_model.pth"))
            flag = " *"
        print(
            f"  Epoch {epoch:02d}/{num_epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:5.2f}%  "
            f"val loss={vl_loss:.4f} acc={vl_acc*100:5.2f}%{flag}  "
            f"({time.time()-t0:.1f}s)"
        )

    # Restore the best weights for downstream evaluation.
    model.load_state_dict(torch.load(ckpt_path.with_name("best_model.pth"), map_location=DEVICE))

    # Persist a final checkpoint that bundles weights + metadata.
    torch.save({
        "architecture": name,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "epochs_trained": num_epochs,
        "best_val_acc": best_val_acc,
        "model_state": model.state_dict(),
    }, ckpt_path)
    print(f"  Best val acc: {best_val_acc*100:.2f}%  ->  {ckpt_path.name}")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) for the given loader."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="predict", leave=False):
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)


def per_class_metrics(y_true, y_pred) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0
    )
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support,
    })
    df["Accuracy"] = df["Recall"]  # per-class accuracy == recall
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(histories: dict[str, dict]) -> None:
    """Two-panel figure: loss and accuracy for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_cnn = range(1, len(histories["Custom CNN"]["train_loss"]) + 1)
    epochs_res = range(1, len(histories["ResNet50"]["train_loss"]) + 1)

    axes[0].plot(epochs_cnn, histories["Custom CNN"]["train_loss"],
                 "-o", color="#1f77b4", markersize=4, label="Custom CNN (train)")
    axes[0].plot(epochs_cnn, histories["Custom CNN"]["val_loss"],
                 "--s", color="#1f77b4", markersize=4, label="Custom CNN (val)")
    axes[0].plot(epochs_res,  histories["ResNet50"]["train_loss"],
                 "-o", color="#d62728", markersize=4, label="ResNet50 (train)")
    axes[0].plot(epochs_res,  histories["ResNet50"]["val_loss"],
                 "--s", color="#d62728", markersize=4, label="ResNet50 (val)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("(a) Loss vs. Epoch"); axes[0].legend(loc="upper right")

    axes[1].plot(epochs_cnn, [a * 100 for a in histories["Custom CNN"]["train_acc"]],
                 "-o", color="#1f77b4", markersize=4, label="Custom CNN (train)")
    axes[1].plot(epochs_cnn, [a * 100 for a in histories["Custom CNN"]["val_acc"]],
                 "--s", color="#1f77b4", markersize=4, label="Custom CNN (val)")
    axes[1].plot(epochs_res,  [a * 100 for a in histories["ResNet50"]["train_acc"]],
                 "-o", color="#d62728", markersize=4, label="ResNet50 (train)")
    axes[1].plot(epochs_res,  [a * 100 for a in histories["ResNet50"]["val_acc"]],
                 "--s", color="#d62728", markersize=4, label="ResNet50 (val)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("(b) Accuracy vs. Epoch"); axes[1].legend(loc="lower right")
    axes[1].set_ylim(0, 105)

    fig.suptitle("Training Dynamics on EuroSAT RGB", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png")
    plt.close(fig)


def plot_confusion(y_true, y_pred, name: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Normalized frequency"}, ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_per_class_metrics(per_class_dfs: dict[str, pd.DataFrame]) -> None:
    """Grouped bar chart of precision/recall/F1 across both models."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(NUM_CLASSES)
    n_models = len(per_class_dfs)
    metrics = ["Precision", "Recall", "F1-Score"]
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    n_bars = n_models * len(metrics)
    width = 0.8 / n_bars

    for mi, metric in enumerate(metrics):
        for mj, (name, df) in enumerate(per_class_dfs.items()):
            offset = (mi * n_models + mj - (n_bars - 1) / 2) * width
            color = palette[mj % len(palette)]
            alpha = 0.95 if mi == 0 else 0.7 + 0.15 * mi
            label = f"{name} — {metric}" if mi == 0 else None
            ax.bar(x + offset, df[metric].values, width=width,
                   color=color, alpha=alpha, label=label, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Precision, Recall, and F1-Score", fontweight="bold")
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "per_class_metrics.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------
def export_excel(
    overall_rows: list[dict],
    per_class_dfs: dict[str, pd.DataFrame],
    histories: dict[str, dict],
    out_path: Path,
) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame(overall_rows).to_excel(xw, sheet_name="Overall_Accuracy", index=False)
        combined_pc = pd.concat(
            [df.assign(Model=name) for name, df in per_class_dfs.items()],
            ignore_index=True,
        )[["Model", "Class", "Precision", "Recall", "F1-Score", "Support", "Accuracy"]]
        combined_pc.to_excel(xw, sheet_name="Per_Class_Metrics", index=False)

        hist_rows = []
        for name, h in histories.items():
            for i in range(len(h["train_loss"])):
                hist_rows.append({
                    "Model": name,
                    "Epoch": i + 1,
                    "Train_Loss": h["train_loss"][i],
                    "Val_Loss":   h["val_loss"][i],
                    "Train_Acc":  h["train_acc"][i],
                    "Val_Acc":    h["val_acc"][i],
                })
        pd.DataFrame(hist_rows).to_excel(xw, sheet_name="Training_History", index=False)

        comp = pd.DataFrame(overall_rows)[[
            "Model", "Overall_Accuracy", "Macro_F1", "Weighted_F1",
            "Params_M", "Epochs", "Inference_Time_ms_per_Image",
        ]]
        comp.to_excel(xw, sheet_name="Model_Comparison", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    print(f"Device: {DEVICE}  |  torch {torch.__version__}")
    print(f"Classes: {CLASS_NAMES}")

    overall_rows: list[dict] = []
    histories: dict[str, dict] = {}
    per_class_dfs: dict[str, pd.DataFrame] = {}
    inference_times: dict[str, float] = {}

    # ---- Custom CNN @ 64x64 ----
    print("\n>>> Custom CNN <<<")
    cnn = EuroSATCNN(NUM_CLASSES).to(DEVICE)
    cnn_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"  Trainable params: {cnn_params:,}")
    cnn_loaders, cnn_sizes = make_loaders(img_size=64, batch_size=BATCH_SIZE_CNN)
    print(f"  Splits: {cnn_sizes}")
    cnn_ckpt = MODELS_DIR / "custom_cnn" / "eurosat_cnn_final.pth"
    h_cnn = train_model(cnn, cnn_loaders, NUM_EPOCHS_CNN, LR_CNN, "Custom CNN", cnn_ckpt)
    histories["Custom CNN"] = h_cnn

    y_true_c, y_pred_c, inf_ms_c = evaluate(cnn, cnn_loaders["test"])
    overall_rows.append(build_overall_row("Custom CNN", y_true_c, y_pred_c, cnn_params, NUM_EPOCHS_CNN, inf_ms_c))
    per_class_dfs["Custom CNN"] = per_class_metrics(y_true_c, y_pred_c)
    plot_confusion(y_true_c, y_pred_c, "Custom CNN", RESULTS_DIR / "confusion_matrix_cnn.png")
    inference_times["Custom CNN"] = inf_ms_c

    # ---- ResNet50 @ 224x224 ----
    print("\n>>> ResNet50 <<<")
    rn50 = build_resnet50(NUM_CLASSES).to(DEVICE)
    rn50_params = sum(p.numel() for p in rn50.parameters() if p.requires_grad)
    print(f"  Trainable params: {rn50_params:,}")
    rn_loaders, rn_sizes = make_loaders(img_size=224, batch_size=BATCH_SIZE_RESNET)
    print(f"  Splits: {rn_sizes}")
    rn_ckpt = MODELS_DIR / "resnet50" / "eurosat_resnet50_final.pth"
    h_rn = train_model(rn50, rn_loaders, NUM_EPOCHS_RESNET, LR_RESNET, "ResNet50", rn_ckpt)
    histories["ResNet50"] = h_rn

    y_true_r, y_pred_r, inf_ms_r = evaluate(rn50, rn_loaders["test"])
    overall_rows.append(build_overall_row("ResNet50", y_true_r, y_pred_r, rn50_params, NUM_EPOCHS_RESNET, inf_ms_r))
    per_class_dfs["ResNet50"] = per_class_metrics(y_true_r, y_pred_r)
    plot_confusion(y_true_r, y_pred_r, "ResNet50", RESULTS_DIR / "confusion_matrix_resnet50.png")
    inference_times["ResNet50"] = inf_ms_r

    # ---- Plots & Excel ----
    plot_training_curves(histories)
    plot_per_class_metrics(per_class_dfs)
    export_excel(overall_rows, per_class_dfs, histories, RESULTS_DIR / "metrics_export.xlsx")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINAL TEST ACCURACY")
    print("=" * 60)
    for row in overall_rows:
        print(f"  {row['Model']:<12}  {row['Overall_Accuracy']*100:6.2f}%   "
              f"F1={row['Macro_F1']:.4f}   params={row['Params_M']:.2f}M   "
              f"infer={row['Inference_Time_ms_per_Image']:.2f} ms/img")
    print(f"\nArtifacts written to {RESULTS_DIR}")
    print(f"Models written to {MODELS_DIR}")


def evaluate(model, loader) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the test set, returning (y_true, y_pred, ms_per_image)."""
    model.eval()
    y_true, y_pred = [], []
    n_imgs = 0
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="test", leave=False):
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
            n_imgs += imgs.size(0)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return np.array(y_true), np.array(y_pred), (elapsed / n_imgs) * 1000.0


def build_overall_row(
    name: str, y_true, y_pred, params: int, epochs: int, inf_ms: float
) -> dict:
    return {
        "Model": name,
        "Overall_Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1":         f1_score(y_true, y_pred, average="macro"),
        "Weighted_F1":      f1_score(y_true, y_pred, average="weighted"),
        "Params_M":         params / 1e6,
        "Epochs":           epochs,
        "Inference_Time_ms_per_Image": inf_ms,
    }


if __name__ == "__main__":
    main()
