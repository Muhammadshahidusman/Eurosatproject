"""
Efficiency profile for the trained models.

Measures trainable parameter count, FLOPs, peak memory, and
wall-clock inference latency for the Custom CNN and ResNet50
backbones. Saves a publication-quality bubble chart and a
multi-sheet Excel workbook.

Outputs:
    results/efficiency_bubble_chart.png
    results/efficiency_metrics.xlsx

Run from the project root:
    uv run python src/efficiency.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from thop import profile
from torchvision import models, transforms

# Reuse the Custom CNN definition from enhanced_training without importing
# the file (which would execute its main()).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.3444, 0.3803, 0.4078]
STD = [0.2034, 0.1365, 0.1148]
NUM_CLASSES = 10
NUM_WARMUP = 10
NUM_TIMED = 100

# Publication-quality defaults.
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class EuroSATCNN(nn.Module):
    """Custom 3-block CNN (same as enhanced_training.py)."""

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
            nn.Linear(512, 256),         nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def model_size_mb(model: nn.Module) -> float:
    """Approximate in-memory size of the model parameters (float32)."""
    n = sum(p.numel() for p in model.parameters())
    return n * 4 / (1024 ** 2)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model: nn.Module, input_size: tuple[int, ...]) -> float:
    """Return MACs (multiply-accumulates) using `thop`."""
    dummy = torch.randn(1, *input_size, device=DEVICE)
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(macs)


def measure_latency(model: nn.Module, input_size: tuple[int, ...],
                    warmup: int, n_timed: int) -> float:
    """Return per-image wall-clock latency in milliseconds (CPU/GPU synced)."""
    model.eval()
    timings = []
    with torch.no_grad():
        # Warm-up runs (also let cuDNN pick algorithms).
        for _ in range(warmup):
            _ = model(torch.randn(1, *input_size, device=DEVICE))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(n_timed):
            x = torch.randn(1, *input_size, device=DEVICE)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
    return float(np.median(timings)) * 1000.0


def measure_peak_memory(model: nn.Module, input_size: tuple[int, ...]) -> float:
    """Return peak GPU memory in MB; 0 if CUDA is unavailable."""
    if DEVICE.type != "cuda":
        return 0.0
    model = model.to(DEVICE)
    model.eval()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(torch.randn(1, *input_size, device=DEVICE))
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024 ** 2)


def system_snapshot() -> dict:
    """Snapshot the host CPU/RAM state (informational)."""
    vm = psutil.virtual_memory()
    return {
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_gb": vm.total / (1024 ** 3),
        "ram_used_gb": vm.used / (1024 ** 3),
        "ram_percent": vm.percent,
    }


# ---------------------------------------------------------------------------
# Excel + chart
# ---------------------------------------------------------------------------
def export_excel(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    sys_df = pd.DataFrame([system_snapshot()])

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Efficiency_Metrics", index=False)
        sys_df.to_excel(xw, sheet_name="Host_Snapshot", index=False)


def plot_bubble(rows: list[dict], out_path: Path) -> None:
    """Bubble chart: x=latency, y=params, bubble size=MACs, color=model."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    palette = {"Custom CNN": "#1f77b4", "ResNet50": "#d62728"}
    for row in rows:
        ax.scatter(
            row["latency_ms_per_image"],
            row["params_millions"],
            s=row["macs_giga"] * 18,           # bubble area scales with G-MACs
            color=palette[row["model"]],
            alpha=0.65, edgecolor="black", linewidth=0.8,
            label=row["model"],
        )
        ax.annotate(
            row["model"],
            (row["latency_ms_per_image"], row["params_millions"]),
            textcoords="offset points", xytext=(10, 8), fontsize=10, fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median inference latency (ms / image, log scale)")
    ax.set_ylabel("Trainable parameters (millions, log scale)")
    ax.set_title("Efficiency Profile — Custom CNN vs. ResNet50\n"
                 "Bubble area ∝ G-MACs (multiply-accumulates)", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Device        : {DEVICE}")
    print(f"warmup={NUM_WARMUP}, timed runs={NUM_TIMED}")

    candidates = [
        ("Custom CNN", EuroSATCNN(NUM_CLASSES), (3, 64, 64)),
        ("ResNet50",   build_resnet50(NUM_CLASSES), (3, 224, 224)),
    ]

    rows: list[dict] = []
    for name, model, input_size in candidates:
        print(f"\n>>> {name} (input {tuple(input_size)}) <<<")
        model = model.to(DEVICE).eval()

        total, trainable = count_params(model)
        macs = measure_flops(model, input_size)
        latency = measure_latency(model, input_size, NUM_WARMUP, NUM_TIMED)
        peak = measure_peak_memory(model, input_size)
        size_mb = model_size_mb(model)

        row = {
            "model": name,
            "input_resolution": f"{input_size[1]}x{input_size[2]}",
            "params_total": total,
            "trainable_params": trainable,
            "params_millions": trainable / 1e6,
            "size_mb": size_mb,
            "macs": macs,
            "macs_giga": macs / 1e9,
            "gflops": 2 * macs / 1e9,
            "latency_ms_per_image": latency,
            "peak_gpu_memory_mb": peak,
            "device": DEVICE.type,
        }
        rows.append(row)

        print(f"  Total params     : {total:>12,}")
        print(f"  Trainable params : {trainable:>12,}  ({row['params_millions']:.2f} M)")
        print(f"  Size             : {size_mb:.2f} MB")
        print(f"  MACs             : {macs/1e9:.2f} G")
        print(f"  Latency          : {latency:.2f} ms / image")
        print(f"  Peak GPU memory  : {peak:.2f} MB")

    # Outputs
    plot_bubble(rows, RESULTS_DIR / "efficiency_bubble_chart.png")
    export_excel(rows, RESULTS_DIR / "efficiency_metrics.xlsx")
    print(f"\nSaved plot  : {RESULTS_DIR / 'efficiency_bubble_chart.png'}")
    print(f"Saved Excel : {RESULTS_DIR / 'efficiency_metrics.xlsx'}")


if __name__ == "__main__":
    main()
