"""
alzheimer_dataset.py
====================
Core PyTorch Dataset + DataLoader factory for the Kaggle
Alzheimer's MRI Dataset (4-class classification).

Expected directory layout
-------------------------
data/
├── train/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
└── test/
    ├── NonDemented/
    ├── VeryMildDemented/
    ├── MildDemented/
    └── ModerateDemented/
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Class registry ────────────────────────────────────────────────────────────
CLASS_NAMES: List[str] = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}

# Clinical language descriptions for the VLM language branch
CLINICAL_TEXTS: Dict[str, str] = {
    "NonDemented": (
        "Normal hippocampal volume with no cortical atrophy. "
        "Intact sulcal morphology and normal ventricular size. "
        "No white-matter hyperintensities detected."
    ),
    "VeryMildDemented": (
        "Minimal hippocampal volume loss with early-stage entorhinal "
        "cortex thinning. Minor sulcal widening and subtle cortical "
        "changes consistent with very mild cognitive impairment."
    ),
    "MildDemented": (
        "Moderate hippocampal atrophy with parietal cortex thinning. "
        "Enlarged lateral ventricles and progressive sulcal widening. "
        "Reduced entorhinal cortex thickness consistent with mild dementia."
    ),
    "ModerateDemented": (
        "Severe global cortical atrophy with markedly enlarged ventricles. "
        "Diffuse white-matter changes and significant hippocampal volume loss. "
        "Widespread cortical thinning consistent with moderate Alzheimer's disease."
    ),
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Dataset ───────────────────────────────────────────────────────────────────
class AlzheimerMRIDataset(Dataset):
    """
    PyTorch Dataset for Alzheimer's MRI 4-class classification.

    Parameters
    ----------
    root_dir : str | Path
        Root directory that contains ``train/`` and ``test/`` sub-folders.
    split : str
        One of ``'train'``, ``'val'``, or ``'test'``.
    transform : callable, optional
        Torchvision transform applied to every image.
    return_text : bool
        When True each ``__getitem__`` returns ``(image, label, text)``
        instead of ``(image, label)``.  Required for the VLM branch.
    val_fraction : float
        Fraction of the training pool reserved for validation (default 0.15).
    seed : int
        Random seed for the stratified train/val split.
    cache_images : bool
        Pre-load all images into RAM as PIL objects for faster iteration.

    Examples
    --------
    >>> from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
    >>> train_tf, val_tf = get_transforms(img_size=224)
    >>> ds = AlzheimerMRIDataset("data/processed", split="train", transform=train_tf)
    >>> img, label, text = ds[0]
    >>> img.shape, label, text[:40]
    (torch.Size([1, 224, 224]), 0, 'Normal hippocampal volume with no cor')
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[object] = None,
        return_text: bool = True,
        val_fraction: float = 0.15,
        seed: int = 42,
        cache_images: bool = False,
    ) -> None:
        assert split in {"train", "val", "test"}, \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.root_dir     = Path(root_dir)
        self.split        = split
        self.transform    = transform
        self.return_text  = return_text
        self.cache_images = cache_images
        self._img_cache: Dict[int, Image.Image] = {}

        disk_split  = "test" if split == "test" else "train"
        self.split_dir = self.root_dir / disk_split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}\n"
                "Run  scripts/download_dataset.py  first."
            )

        all_paths, all_labels = self._scan_directory(self.split_dir)

        if split in {"train", "val"}:
            all_paths, all_labels = self._stratified_split(
                all_paths, all_labels, val_fraction, seed, split
            )

        self.image_paths: List[Path] = all_paths
        self.labels:      List[int]  = all_labels

        logger.info(
            "AlzheimerMRIDataset | split=%-5s | n=%d | %s",
            split, len(self), self._count_str(),
        )

        if self.cache_images:
            self._fill_cache()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _scan_directory(self, directory: Path) -> Tuple[List[Path], List[int]]:
        paths, labels = [], []
        for cls_name in CLASS_NAMES:
            cls_dir = directory / cls_name
            if not cls_dir.exists():
                logger.warning("Class directory missing: %s", cls_dir)
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in VALID_EXTS:
                    paths.append(p)
                    labels.append(CLASS_TO_IDX[cls_name])
        if not paths:
            raise RuntimeError(f"No images found under {directory}")
        return paths, labels

    @staticmethod
    def _stratified_split(
        paths: List[Path], labels: List[int],
        val_frac: float, seed: int, target: str,
    ) -> Tuple[List[Path], List[int]]:
        rng        = np.random.default_rng(seed)
        paths_arr  = np.array(paths, dtype=object)
        labels_arr = np.array(labels)
        tr_idx, va_idx = [], []
        for c in range(len(CLASS_NAMES)):
            pos = np.where(labels_arr == c)[0]
            rng.shuffle(pos)
            n_val = max(1, int(len(pos) * val_frac))
            va_idx.extend(pos[:n_val].tolist())
            tr_idx.extend(pos[n_val:].tolist())
        chosen = va_idx if target == "val" else tr_idx
        return paths_arr[chosen].tolist(), labels_arr[chosen].tolist()

    def _fill_cache(self) -> None:
        logger.info("Pre-loading %d images into RAM …", len(self))
        for i in range(len(self)):
            self._img_cache[i] = Image.open(self.image_paths[i]).convert("L")

    def _load(self, idx: int) -> Image.Image:
        if self.cache_images and idx in self._img_cache:
            return self._img_cache[idx]
        return Image.open(self.image_paths[idx]).convert("L")

    def _count_str(self) -> str:
        from collections import Counter
        c = Counter(self.labels)
        return " | ".join(f"{IDX_TO_CLASS[k]}={v}" for k, v in sorted(c.items()))

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights — shape (4,)."""
        counts = np.array(
            [self.labels.count(i) for i in range(len(CLASS_NAMES))],
            dtype=np.float32,
        )
        w = 1.0 / (counts + 1e-6)
        return torch.tensor(w / w.sum(), dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for ``WeightedRandomSampler``."""
        cw = self.get_class_weights()
        return torch.tensor([cw[l].item() for l in self.labels], dtype=torch.float32)

    def to_dataframe(self) -> pd.DataFrame:
        """Export metadata as a DataFrame."""
        return pd.DataFrame({
            "path":          [str(p) for p in self.image_paths],
            "label":         self.labels,
            "class_name":    [IDX_TO_CLASS[l] for l in self.labels],
            "clinical_text": [CLINICAL_TEXTS[IDX_TO_CLASS[l]] for l in self.labels],
        })

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = self._load(idx)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        if self.return_text:
            return img, label, CLINICAL_TEXTS[IDX_TO_CLASS[label]]
        return img, label

    def __repr__(self) -> str:
        return (f"AlzheimerMRIDataset(split={self.split!r}, "
                f"n={len(self)}, counts={{{self._count_str()}}})")


# ── Transform factory ─────────────────────────────────────────────────────────

def get_transforms(
    img_size: int = 224,
    mean: Tuple[float, ...] = (0.5,),
    std:  Tuple[float, ...] = (0.5,),
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train and val/test transform pipelines.

    Returns
    -------
    train_transform, val_transform
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.90, 1.10)),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    root_dir: Union[str, Path],
    img_size: int  = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    val_fraction: float = 0.15,
    seed: int = 42,
    cache_images: bool = False,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders in one call.

    Returns
    -------
    dict
        ``{'train': DataLoader, 'val': DataLoader, 'test': DataLoader}``

    Examples
    --------
    >>> loaders = build_dataloaders("data/processed", batch_size=32)
    >>> imgs, labels, texts = next(iter(loaders["train"]))
    >>> imgs.shape
    torch.Size([32, 1, 224, 224])
    """
    train_tf, val_tf = get_transforms(img_size=img_size)
    datasets = {
        "train": AlzheimerMRIDataset(root_dir, "train", train_tf,
                                     val_fraction=val_fraction, seed=seed,
                                     cache_images=cache_images),
        "val":   AlzheimerMRIDataset(root_dir, "val",   val_tf,
                                     val_fraction=val_fraction, seed=seed,
                                     cache_images=cache_images),
        "test":  AlzheimerMRIDataset(root_dir, "test",  val_tf,
                                     cache_images=cache_images),
    }
    loaders: Dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        if split == "train" and use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=ds.get_sample_weights(),
                num_samples=len(ds),
                replacement=True,
            )
            loaders[split] = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                        num_workers=num_workers, pin_memory=True,
                                        drop_last=True)
        else:
            loaders[split] = DataLoader(ds, batch_size=batch_size,
                                        shuffle=(split == "train"),
                                        num_workers=num_workers, pin_memory=True,
                                        drop_last=(split == "train"))
    return loaders
