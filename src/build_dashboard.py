"""
Build the static portfolio dashboard in `dashboard/`.

Re-creates the exact 70/15/15 split used by enhanced_training.py, re-evaluates
both checkpoints on the held-out test set, exports them to ONNX (int8
quantised for the browser), verifies the ONNX models against PyTorch,
generates Grad-CAM overlays on *test-split* images only, and packages every
number and image the dashboard needs.

    dashboard/
        data/metrics.json        # all metrics, curves, confusion matrices
        data/lahore.json         # per-tile class grids for both models
        models/*.onnx            # browser models (onnxruntime-web)
        img/samples/*.jpg        # original 64x64 EuroSAT test images
        img/gradcam/*.jpg        # ResNet50 Grad-CAM overlays
        img/lahore_rgb.jpg       # Sentinel-2 RGB backdrop for the map
        img/figures/*.webp       # web copies of the result figures in results/
        data/downloads/*.xlsx    # result workbooks offered for download

Run from the project root (needs torch, torchvision, onnx, onnxruntime,
openpyxl, Pillow):
    python src/build_dashboard.py --data-dir data/Eurosat_Dataset

`--data-dir` may be the flat EuroSAT folder or the official class-subfolder
layout; files are ordered by filename either way, matching the training split.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import openpyxl
import torch
import torch.nn as nn
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration (mirrors src/enhanced_training.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = PROJECT_ROOT / "dashboard"

SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
MEAN = [0.3444, 0.3803, 0.4078]
STD = [0.2034, 0.1365, 0.1148]
INPUT_SIZE = {"cnn": 64, "resnet50": 224}

# Lahore class-map colours (must match src/lahore_classification_map.py).
CLASS_COLORS = {
    "AnnualCrop": "#F0E68C", "Forest": "#228B22", "HerbaceousVegetation": "#90EE90",
    "Highway": "#404040", "Industrial": "#B22222", "Pasture": "#9ACD32",
    "PermanentCrop": "#FF8C00", "Residential": "#9370DB", "River": "#1E90FF",
    "SeaLake": "#000080",
}
PATCH = 64

SAMPLES_PER_CLASS = 3
MAX_MISCLASSIFIED = 8
N_CALIBRATION = 256


# ---------------------------------------------------------------------------
# Models (architecture identical to enhanced_training.EuroSATCNN)
# ---------------------------------------------------------------------------
class EuroSATCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(in_ch: int, out_ch: int, p: float = 0.25) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(p),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_state(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt


def load_models() -> dict[str, nn.Module]:
    cnn = EuroSATCNN(len(CLASS_NAMES))
    cnn.load_state_dict(load_state(MODELS_DIR / "custom_cnn" / "eurosat_cnn_final.pth"))
    rn = models.resnet50(weights=None)
    rn.fc = nn.Linear(rn.fc.in_features, len(CLASS_NAMES))
    rn.load_state_dict(load_state(MODELS_DIR / "resnet50" / "eurosat_resnet50_final.pth"))
    return {"cnn": cnn.eval(), "resnet50": rn.eval()}


# ---------------------------------------------------------------------------
# Data split (identical to enhanced_training.make_loaders)
# ---------------------------------------------------------------------------
def list_samples(data_dir: Path) -> list[tuple[Path, int]]:
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    files = {p.name: p for p in data_dir.rglob("*.jpg")}
    samples = []
    for name in sorted(files):
        prefix = "_".join(name.replace(".jpg", "").split("_")[:-1])
        if prefix in class_to_idx:
            samples.append((files[name], class_to_idx[prefix]))
    return samples


def split_indices(total: int) -> tuple[list[int], list[int], list[int]]:
    n_train, n_val = int(TRAIN_RATIO * total), int(VAL_RATIO * total)
    indices = list(range(total))
    np.random.RandomState(SEED).shuffle(indices)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def batches(samples, idx, size, bs=64):
    t = eval_transform(size)
    for k in range(0, len(idx), bs):
        chunk = idx[k:k + bs]
        x = torch.stack([t(Image.open(samples[i][0]).convert("RGB")) for i in chunk])
        y = np.array([samples[i][1] for i in chunk])
        yield chunk, x, y


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def summarise(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n), where=(precision + recall) > 0)
    support = cm.sum(1)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "per_class": [
            {"class": c, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(CLASS_NAMES)
        ],
        "confusion": cm.tolist(),
    }


def eval_torch(model, samples, idx, size):
    preds, probs = [], []
    with torch.no_grad():
        for _, x, _ in batches(samples, idx, size):
            p = torch.softmax(model(x), 1)
            probs.append(p.numpy())
            preds.append(p.argmax(1).numpy())
    return np.concatenate(preds), np.concatenate(probs)


def eval_onnx(path: Path, samples, idx, size):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    preds = []
    for _, x, _ in batches(samples, idx, size):
        preds.append(sess.run(None, {name: x.numpy()})[0].argmax(1))
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# ONNX export + int8 quantisation
# ---------------------------------------------------------------------------
class Calibration(CalibrationDataReader):
    def __init__(self, samples, idx, size, input_name):
        self.it = (({input_name: x.numpy()}) for _, x, _ in batches(samples, idx, size, bs=16))

    def get_next(self):
        return next(self.it, None)


def export_onnx(key: str, model: nn.Module, samples, calib_idx, work: Path) -> Path:
    size = INPUT_SIZE[key]
    fp32 = work / f"{key}_fp32.onnx"
    torch.onnx.export(
        model, torch.randn(1, 3, size, size), str(fp32),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    prep = work / f"{key}_prep.onnx"
    quant_pre_process(str(fp32), str(prep))
    int8 = work / f"{key}_int8.onnx"
    quantize_static(
        str(prep), str(int8), Calibration(samples, calib_idx, size, "input"),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
    )
    return int8


# ---------------------------------------------------------------------------
# Grad-CAM (ResNet50, last bottleneck of layer4)
# ---------------------------------------------------------------------------
def jet(v: np.ndarray) -> np.ndarray:
    """Matplotlib-style jet colormap for v in [0, 1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def gradcam_overlay(model: nn.Module, img: Image.Image, target: int) -> Image.Image:
    acts, grads = {}, {}
    layer = model.layer4[-1]

    def keep(module, inputs, output):
        acts["a"] = output
        output.register_hook(lambda g: grads.update(g=g))

    handle = layer.register_forward_hook(keep)
    x = eval_transform(224)(img).unsqueeze(0)
    model.zero_grad()
    model(x)[0, target].backward()
    handle.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    base = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32)
    out = 0.55 * base + 0.45 * jet(cam).astype(np.float32)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Existing result workbooks
# ---------------------------------------------------------------------------
def read_sheet(path: Path, sheet: str) -> list[dict]:
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet]
    rows = list(ws.iter_rows(values_only=True))
    return [dict(zip(rows[0], r)) for r in rows[1:]]


# ---------------------------------------------------------------------------
# Lahore tile grids, recovered losslessly from the class-map PNGs
# ---------------------------------------------------------------------------
def lahore_grid(model_key: str) -> tuple[str, int, int]:
    arr = np.asarray(Image.open(RESULTS_DIR / f"lahore_classification_map_{model_key}.png").convert("RGB"))
    rows, cols = arr.shape[0] // PATCH, arr.shape[1] // PATCH
    centres = arr[PATCH // 2::PATCH, PATCH // 2::PATCH][:rows, :cols]
    lut = {tuple(int(h[k:k + 2], 16) for k in (1, 3, 5)): str(i)
           for i, h in enumerate(CLASS_COLORS[c] for c in CLASS_NAMES)}
    chars = []
    for px in centres.reshape(-1, 3):
        t = tuple(int(v) for v in px)
        if t == (255, 255, 255):
            chars.append(".")
        elif t in lut:
            chars.append(lut[t])
        else:
            raise ValueError(f"Unexpected colour {t} in {model_key} class map")
    return "".join(chars), rows, cols


def lahore_rgb_backdrop(rows: int, cols: int, out: Path) -> None:
    """Crop the Sentinel-2 RGB panel out of the side-by-side figure."""
    im = Image.open(RESULTS_DIR / "lahore_rgb_vs_classification_resnet50.jpg").convert("RGB")
    a = np.asarray(im).astype(int)
    half = a[:, : a.shape[1] // 2]
    dark = half.sum(-1) < 60                       # panel background is black
    # Rows/columns that are mostly black belong to the image panel; titles and
    # legend text are only sparsely dark.
    rows_in = np.where(dark.mean(1) > 0.3)[0]
    y0, y1 = rows_in.min(), rows_in.max() + 1
    cols_in = np.where(dark[y0:y1].mean(0) > 0.3)[0]
    x0, x1 = cols_in.min(), cols_in.max() + 1
    panel = im.crop((x0, y0, x1, y1))
    # Panel spans the full 5405x6221 raster; the tile grid covers the top-left
    # rows*64 x cols*64 pixels of it.
    H, W = 5405, 6221
    pw, ph = panel.size
    panel = panel.crop((0, 0, round(pw * cols * PATCH / W), round(ph * rows * PATCH / H)))
    panel.resize((cols * 12, rows * 12), Image.LANCZOS).save(out, quality=88)


# ---------------------------------------------------------------------------
# Result figures + downloadable workbooks
# ---------------------------------------------------------------------------
# results/ file -> published name. Figures are re-encoded as WebP at <= 2000 px.
FIGURES = {
    "Study_AreaMAp.png": "study_area",
    "methodology_flowchart (2).png": "methodology",
    "training_curves.png": "training_curves",
    "confusion_matrix_cnn.png": "confusion_cnn",
    "confusion_matrix_resnet50.png": "confusion_resnet50",
    "per_class_metrics.png": "per_class_metrics",
    "efficiency_bubble_chart.png": "efficiency_bubble",
    "gradcam_samples_grid.png": "gradcam_grid",
    "lahore_classification_comparison.jpg": "lahore_comparison",
    "lahore_rgb_vs_classification_resnet50.jpg": "lahore_rgb_vs_resnet50",
    "lahore_rgb_vs_classification_cnn.jpg": "lahore_rgb_vs_cnn",
    "lahore_land_cover_pie.png": "lahore_pie",
    "lahore_class_distribution.png": "lahore_class_distribution",
}
DOWNLOADS = ["metrics_export.xlsx", "efficiency_metrics.xlsx", "lahore_analysis.xlsx"]
MAX_FIGURE_PX = 2000


def export_figures() -> None:
    fig_dir = OUT_DIR / "img" / "figures"
    dl_dir = OUT_DIR / "data" / "downloads"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dl_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for src, name in FIGURES.items():
        im = Image.open(RESULTS_DIR / src)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        im = im.convert("RGB")
        im.thumbnail((MAX_FIGURE_PX, MAX_FIGURE_PX), Image.LANCZOS)
        im.save(fig_dir / f"{name}.webp", quality=90, method=6)
        sizes[name] = im.size
    for name in DOWNLOADS:
        shutil.copy(RESULTS_DIR / name, dl_dir / name)
    (OUT_DIR / "data" / "figures.json").write_text(json.dumps(sizes))
    print(f"Figures: {len(FIGURES)} -> {fig_dir}, downloads: {len(DOWNLOADS)} -> {dl_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "Eurosat_Dataset")
    ap.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "dashboard" / ".build")
    ap.add_argument("--figures-only", action="store_true",
                    help="only refresh the result figures and downloadable workbooks")
    args = ap.parse_args()
    export_figures()
    if args.figures_only:
        return
    torch.manual_seed(SEED)

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "models", "img/samples", "img/gradcam"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    samples = list_samples(args.data_dir)
    assert len(samples) == 27000, f"expected 27,000 EuroSAT images, found {len(samples)}"
    train_idx, val_idx, test_idx = split_indices(len(samples))
    y_test = np.array([samples[i][1] for i in test_idx])
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    nets = load_models()
    results, probs_test = {}, {}
    rng = np.random.RandomState(SEED)
    calib_idx = list(rng.choice(train_idx, N_CALIBRATION, replace=False))

    for key, net in nets.items():
        size = INPUT_SIZE[key]
        t0 = time.time()
        pred, prob = eval_torch(net, samples, test_idx, size)
        probs_test[key] = prob
        results[key] = summarise(y_test, pred)
        print(f"[{key}] PyTorch test acc = {results[key]['accuracy']*100:.2f}%  ({time.time()-t0:.0f}s)")

        int8 = export_onnx(key, net, samples, calib_idx, work)
        q_pred = eval_onnx(int8, samples, test_idx, size)
        q = summarise(y_test, q_pred)
        agree = float((q_pred == pred).mean())
        results[key]["onnx_int8"] = {
            "accuracy": q["accuracy"], "macro_f1": q["macro_f1"],
            "agreement_with_pytorch": agree,
            "size_mb": int8.stat().st_size / 1e6,
        }
        print(f"[{key}] ONNX int8 test acc = {q['accuracy']*100:.2f}%  agreement={agree*100:.2f}%  "
              f"size={int8.stat().st_size/1e6:.1f} MB")
        shutil.copy(int8, OUT_DIR / "models" / f"{key}.onnx")

    # ---- Gallery samples + Grad-CAM (test split only) ----
    rn = nets["resnet50"]
    rn_pred = probs_test["resnet50"].argmax(1)
    cnn_pred = probs_test["cnn"].argmax(1)
    chosen: list[int] = []
    for c in range(len(CLASS_NAMES)):
        pos = [k for k in range(len(test_idx)) if y_test[k] == c and rn_pred[k] == c]
        chosen += list(rng.choice(pos, SAMPLES_PER_CLASS, replace=False))
    wrong = [k for k in range(len(test_idx)) if rn_pred[k] != y_test[k]]
    chosen += list(rng.choice(wrong, min(MAX_MISCLASSIFIED, len(wrong)), replace=False))

    gallery = []
    for k in chosen:
        path, label = samples[test_idx[k]]
        img = Image.open(path).convert("RGB")
        shutil.copy(path, OUT_DIR / "img" / "samples" / path.name)
        gradcam_overlay(rn, img, int(rn_pred[k])).save(
            OUT_DIR / "img" / "gradcam" / path.name, quality=90)
        gallery.append({
            "file": path.name, "true": CLASS_NAMES[label],
            "resnet50": {"pred": CLASS_NAMES[rn_pred[k]], "conf": float(probs_test["resnet50"][k].max())},
            "cnn": {"pred": CLASS_NAMES[cnn_pred[k]], "conf": float(probs_test["cnn"][k].max())},
        })
    print(f"Gallery: {len(gallery)} test images ({len(wrong)} ResNet50 errors in test set)")

    # ---- Training history + efficiency from existing workbooks ----
    history = {"cnn": [], "resnet50": []}
    for r in read_sheet(RESULTS_DIR / "metrics_export.xlsx", "Training_History"):
        key = "cnn" if "CNN" in r["Model"] else "resnet50"
        history[key].append({k: float(r[k]) for k in ("Train_Loss", "Val_Loss", "Train_Acc", "Val_Acc")})
    efficiency = read_sheet(RESULTS_DIR / "efficiency_metrics.xlsx", "Efficiency_Metrics")
    host = read_sheet(RESULTS_DIR / "efficiency_metrics.xlsx", "Host_Snapshot")

    params = {k: sum(p.numel() for p in m.parameters()) for k, m in nets.items()}
    metrics = {
        "classes": CLASS_NAMES, "colors": CLASS_COLORS,
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "models": {k: {**results[k], "params": params[k], "input_size": INPUT_SIZE[k],
                       "history": history[k]} for k in nets},
        "efficiency": efficiency, "host": host,
        "normalisation": {"mean": MEAN, "std": STD},
        "gallery": gallery,
    }
    (OUT_DIR / "data" / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))

    # ---- Lahore ----
    grids = {}
    for key in ("cnn", "resnet50"):
        grid, rows, cols = lahore_grid(key)
        grids[key] = grid
    both = [(a, b) for a, b in zip(grids["cnn"], grids["resnet50"]) if a != "." and b != "."]
    lahore = {
        "rows": rows, "cols": cols, "tile_m": PATCH * 10,
        "acquisition": "Sentinel-2 L2A (S2_SR_HARMONIZED) median composite, 2026-01-01 to 2026-03-30",
        "grids": grids,
        "agreement": sum(a == b for a, b in both) / len(both),
    }
    (OUT_DIR / "data" / "lahore.json").write_text(json.dumps(lahore))
    lahore_rgb_backdrop(rows, cols, OUT_DIR / "img" / "lahore_rgb.jpg")
    print(f"Lahore: {rows}x{cols} tiles, CNN/ResNet50 agreement {lahore['agreement']*100:.1f}%")
    print(f"Dashboard assets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
