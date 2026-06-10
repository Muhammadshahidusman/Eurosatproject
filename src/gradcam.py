"""
Grad-CAM visualisations for the trained ResNet50 model.

Applies Grad-CAM to the final convolutional layer of the ResNet50
backbone (layer4[-1].conv3) on a random sample of EuroSAT test
images and produces publication-quality overlays.

Outputs:
    results/gradcam_samples_grid.png      # 5x4 figure
    results/gradcam_individual/xxx.png     # 50 individual heatmaps

Run from the project root:
    uv run python src/visualize_gradcam.py
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM                                  # FIX 1: correct import
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # FIX 2: target wrapper
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Eurosat_Dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "resnet50" / "eurosat_resnet50_final.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
INDIVIDUAL_DIR = RESULTS_DIR / "gradcam_individual"
INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 50
GRID_COLS = 4
GRID_ROWS = 5
SEED = 42

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
INFER_SIZE = (224, 224)
MEAN = [0.3444, 0.3803, 0.4078]
STD = [0.2034, 0.1365, 0.1148]

# Publication-quality defaults.
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model() -> nn.Module:
    """Load the trained ResNet50 with the FC replaced for 10 classes."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EUROSAT_CLASSES))
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------
def denormalise(t: torch.Tensor) -> np.ndarray:
    """Undo the channel normalisation for display."""
    img = t.detach().cpu().permute(1, 2, 0).numpy()
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


def overlay(heatmap: np.ndarray, rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a CAM heatmap over an RGB image."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap("jet")
    heat = cmap(heatmap)[..., :3]  # drop alpha channel
    return np.clip(alpha * heat + (1 - alpha) * rgb, 0, 1)


def gradcam_for_image(model: nn.Module, pil_image: Image.Image, target_class: int) -> np.ndarray:
    """Return a normalised [0,1] Grad-CAM heatmap for one image."""
    transform = transforms.Compose([
        transforms.Resize(INFER_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    target_layers = [model.layer4[-1].conv3]              # FIX 3: list + plural kwarg
    targets = [ClassifierOutputTarget(target_class)]       # FIX 4: explicit target object

    with GradCAM(model=model, target_layers=target_layers) as cam:   # FIX 5: context manager
        grayscale = cam(input_tensor=tensor, targets=targets)

    return grayscale[0]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_test_images(n: int) -> list[tuple[Path, int]]:
    """Pick n (path, true_label) pairs from the EuroSAT folder, balancing classes."""
    random.seed(SEED)
    by_class: dict[str, list[Path]] = {c: [] for c in EUROSAT_CLASSES}
    for path in DATA_DIR.glob("*.jpg"):
        prefix = "_".join(path.stem.split("_")[:-1])
        if prefix in by_class:
            by_class[prefix].append(path)

    per_class = max(1, n // len(EUROSAT_CLASSES))
    pool: list[Path] = []
    for cls, files in by_class.items():
        random.shuffle(files)
        pool.extend(files[:per_class])
    random.shuffle(pool)
    pool = pool[:n]

    return [(p, EUROSAT_CLASSES.index("_".join(p.stem.split("_")[:-1]))) for p in pool]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_individual(rgb: np.ndarray, overlay_img: np.ndarray,
                    true_label: str, pred_label: str, conf: float,
                    save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(rgb); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(overlay_img); axes[1].set_title("Grad-CAM overlay"); axes[1].axis("off")
    axes[2].imshow(rgb); axes[2].imshow(overlay_img, alpha=0.4)
    axes[2].set_title(f"True: {true_label}\nPred: {pred_label} ({conf*100:.1f}%)"); axes[2].axis("off")
    fig.suptitle(f"Grad-CAM — {save_path.stem}", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_grid(samples: list[dict], save_path: Path, n_rows: int, n_cols: int) -> None:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)
    for ax, s in zip(axes.flat, samples):
        ax.imshow(s["overlay"])
        ax.set_title(f"Pred: {s['pred']}\nTrue: {s['true']} ({s['conf']*100:.0f}%)",
                     fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(samples):]:
        ax.axis("off")
    fig.suptitle("Grad-CAM Heatmaps — ResNet50 on EuroSAT RGB", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained checkpoint missing: {MODEL_PATH}")

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Device        : {DEVICE}")
    print(f"Model         : {MODEL_PATH.name}")
    print(f"Source images : {DATA_DIR}")

    model = load_model()
    print(f"Model params  : {sum(p.numel() for p in model.parameters()):,}")

    samples_meta = sample_test_images(NUM_SAMPLES)
    print(f"Sampled images: {len(samples_meta)}")

    softmax = nn.Softmax(dim=1)
    transform = transforms.Compose([
        transforms.Resize(INFER_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    samples: list[dict] = []
    for i, (path, true_idx) in enumerate(samples_meta):
        pil = Image.open(path).convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs = softmax(logits)
            conf, pred = probs.max(dim=1)

        pred_idx = int(pred.item())
        rgb = denormalise(tensor[0])
        heat = gradcam_for_image(model, pil, pred_idx)
        ovl = overlay(heat, rgb)

        out = INDIVIDUAL_DIR / f"gradcam_{i:03d}_true-{EUROSAT_CLASSES[true_idx]}_pred-{EUROSAT_CLASSES[pred_idx]}.png"
        plot_individual(rgb, ovl, EUROSAT_CLASSES[true_idx], EUROSAT_CLASSES[pred_idx],
                        float(conf.item()), out)

        samples.append({
            "overlay": ovl,
            "true": EUROSAT_CLASSES[true_idx],
            "pred": EUROSAT_CLASSES[pred_idx],
            "conf": float(conf.item()),
        })
        print(f"  {i+1:02d}/{len(samples_meta)}  true={EUROSAT_CLASSES[true_idx]:<22}  "
              f"pred={EUROSAT_CLASSES[pred_idx]:<22}  conf={conf.item()*100:5.1f}%")

    plot_grid(samples, RESULTS_DIR / "gradcam_samples_grid.png", GRID_ROWS, GRID_ROWS)
    print(f"\nSaved grid    : {RESULTS_DIR / 'gradcam_samples_grid.png'}")
    print(f"Saved {len(samples)} individuals to {INDIVIDUAL_DIR}/")


if __name__ == "__main__":
    main()