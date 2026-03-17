"""
utils.py
========
Shared utilities for the MedVLM-AD project.

Sections
--------
1. Metrics    – accuracy, F1, confusion matrix
2. Plots      – distributions, sample grid, curves, confusion matrix
3. Checkpoint – save / load helpers
4. Misc       – set_seed, logger setup, dataset summary
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

# ── 1. Metrics ────────────────────────────────────────────────────────────────

def compute_accuracy(preds, labels) -> float:
    """Top-1 accuracy."""
    return float((np.asarray(preds) == np.asarray(labels)).mean())


def compute_metrics(
    preds, labels,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Full classification metrics suite.

    Returns
    -------
    dict  keys: accuracy, macro_f1, weighted_f1, macro_precision,
                macro_recall, f1_<ClassName> × 4
    """
    try:
        from sklearn.metrics import (
            accuracy_score, f1_score,
            precision_score, recall_score,
        )
    except ImportError as e:
        raise ImportError("pip install scikit-learn") from e

    p, l = np.asarray(preds), np.asarray(labels)
    out: Dict[str, float] = {
        "accuracy":        accuracy_score(l, p),
        "macro_f1":        f1_score(l, p, average="macro",    zero_division=0),
        "weighted_f1":     f1_score(l, p, average="weighted", zero_division=0),
        "macro_precision": precision_score(l, p, average="macro", zero_division=0),
        "macro_recall":    recall_score(l, p, average="macro",    zero_division=0),
    }
    names = class_names or CLASS_NAMES
    for i, f1 in enumerate(f1_score(l, p, average=None, zero_division=0)):
        out[f"f1_{names[i]}"] = float(f1)
    return out


def compute_confusion_matrix(preds, labels, num_classes: int = 4) -> np.ndarray:
    """Return (K×K) confusion matrix."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(
        np.asarray(labels), np.asarray(preds),
        labels=list(range(num_classes)),
    )


# ── 2. Plots ──────────────────────────────────────────────────────────────────

def plot_class_distribution(dataset, save_path=None) -> None:
    """Bar chart of class sample counts."""
    import matplotlib.pyplot as plt
    from collections import Counter
    counts = Counter(dataset.labels)
    names  = [IDX_TO_CLASS[i] for i in sorted(counts)]
    values = [counts[i]       for i in sorted(counts)]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(f"Class Distribution — {dataset.split.capitalize()} Split",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Samples"); ax.set_xlabel("Class")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_sample_grid(dataset, n_per_class: int = 4, save_path=None) -> None:
    """Grid showing n_per_class MRI samples for each class."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    cls_idx: Dict[int, List[int]] = defaultdict(list)
    for i, l in enumerate(dataset.labels):
        cls_idx[l].append(i)

    fig, axes = plt.subplots(4, n_per_class, figsize=(n_per_class * 2.5, 11))
    for row in range(4):
        for col in range(n_per_class):
            ax = axes[row][col]
            idxs = cls_idx[row]
            if col < len(idxs):
                item = dataset[idxs[col]]
                t = item[0].squeeze(0).numpy()
                ax.imshow(t, cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[row], fontsize=8, rotation=90, labelpad=4)
    fig.suptitle("Sample MRI Images by Dementia Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path=None,
    normalise: bool = True,
) -> None:
    """Heatmap of a confusion matrix."""
    import matplotlib.pyplot as plt
    names = class_names or CLASS_NAMES
    if normalise:
        rs = cm.sum(1, keepdims=True).astype(float)
        cm_p = np.where(rs > 0, cm / rs * 100, 0.0)
        fmt, sfx = ".1f", "%"
    else:
        cm_p, fmt, sfx = cm.astype(float), ".0f", ""

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_p, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
           xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix" + (" (%)" if normalise else ""))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    th = cm_p.max() / 2
    for i in range(cm_p.shape[0]):
        for j in range(cm_p.shape[1]):
            ax.text(j, i, format(cm_p[i, j], fmt) + sfx, ha="center", va="center",
                    color="white" if cm_p[i, j] > th else "black", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses, val_losses,
    train_accs, val_accs,
    save_path=None,
) -> None:
    """Side-by-side loss and accuracy curves."""
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, train_losses, "b-o", ms=3, label="Train")
    ax1.plot(epochs, val_losses,   "r-o", ms=3, label="Val")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [a * 100 for a in train_accs], "b-o", ms=3, label="Train")
    ax2.plot(epochs, [a * 100 for a in val_accs],   "r-o", ms=3, label="Val")
    ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("MedVLM-AD Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


def plot_intensity_histograms(dataset, save_path=None, max_samples: int = 1200) -> None:
    """Per-class pixel intensity histograms."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    buckets: Dict[str, list] = defaultdict(list)
    for idx in range(min(len(dataset), max_samples)):
        item  = dataset[idx]
        label = item[1]
        buckets[IDX_TO_CLASS[label]].extend(item[0].numpy().flatten().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    for ax, cls, color in zip(axes.flat, CLASS_NAMES, colors):
        d = np.array(buckets[cls])
        ax.hist(d, bins=60, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(d.mean(), color="black", ls="--", lw=1.2,
                   label=f"μ={d.mean():.3f}")
        ax.set_title(cls, fontsize=10, fontweight="bold")
        ax.set_xlabel("Pixel value (normalised)"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pixel Intensity by Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    else:
        plt.show()


# ── 3. Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: Union[str, Path],
    filename: str = "checkpoint.pth",
) -> Path:
    """Save model + optimizer state dict with epoch and metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":              metrics,
    }, out)
    logging.getLogger(__name__).info("Checkpoint → %s", out)
    return out


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, Dict]:
    """Load a checkpoint.  Returns (model, optimizer, epoch, metrics)."""
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ck["optimizer_state_dict"])
    return model, optimizer, ck["epoch"], ck.get("metrics", {})


# ── 4. Misc ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set Python / NumPy / PyTorch seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


def setup_logger(
    name: str = "medvlm_ad",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configured logger with console (+ optional file) handler."""
    lgr = logging.getLogger(name)
    lgr.setLevel(level)
    lgr.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); lgr.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt); lgr.addHandler(fh)
    return lgr


def summarise_dataset(root_dir: Union[str, Path]) -> Dict:
    """Count images per split/class without loading tensors."""
    root = Path(root_dir)
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    out: Dict = {}
    total = 0
    for split in ("train", "test"):
        if not (root / split).exists():
            continue
        out[split] = {}
        for cls in CLASS_NAMES:
            d = root / split / cls
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in EXTS) \
                if d.exists() else 0
            out[split][cls] = n
            total += n
    out["total"] = total
    return out


def print_dataset_summary(root_dir: Union[str, Path]) -> None:
    """Pretty-print the dataset file summary."""
    s = summarise_dataset(root_dir)
    print("\n" + "=" * 52)
    print("  Alzheimer's MRI Dataset — Summary")
    print("=" * 52)
    for split in ("train", "test"):
        if split not in s:
            continue
        print(f"\n  {split.upper()}")
        print(f"  {'Class':<24} {'Count':>7}")
        print("  " + "-" * 31)
        for cls in CLASS_NAMES:
            print(f"  {cls:<24} {s[split].get(cls, 0):>7,}")
        print(f"  {'TOTAL':<24} {sum(s[split].values()):>7,}")
    print(f"\n  Grand total: {s.get('total', 0):,} images")
    print("=" * 52 + "\n")
