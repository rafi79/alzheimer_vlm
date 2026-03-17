"""
preprocessor.py
===============
Full MRI preprocessing pipeline for the Alzheimer's MRI Dataset.

Pipeline steps (applied in order)
----------------------------------
1. load_grayscale      – open image, convert to float32 NumPy array
2. resize              – bicubic resize to target (W × H)
3. skull_strip         – threshold-based background removal
4. apply_clahe         – Contrast-Limited Adaptive Histogram Equalisation
5. z_score_normalise   – brain-tissue Z-score normalisation
6. check_artifact      – heuristic blank/corrupt-image detection
7. save_preprocessed   – rescale to uint8 and save as PNG
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_grayscale(path: Union[str, Path]) -> np.ndarray:
    """
    Open an image file and return a float32 grayscale array.

    Returns
    -------
    np.ndarray
        Shape (H, W), dtype float32, values in [0, 255].
    """
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


# ── Step 2: Resize ────────────────────────────────────────────────────────────

def resize(
    arr: np.ndarray,
    size: Tuple[int, int] = (176, 208),
    resample: int = Image.BICUBIC,
) -> np.ndarray:
    """
    Resize a 2-D float32 array to *size* = (width, height).

    Parameters
    ----------
    arr    : float32 array, shape (H, W)
    size   : target (width, height)
    resample : PIL resampling filter

    Returns
    -------
    np.ndarray  same dtype, shape (height, width)
    """
    img = Image.fromarray(arr.astype(np.uint8))
    return np.array(img.resize(size, resample=resample), dtype=np.float32)


# ── Step 3: Skull-strip ───────────────────────────────────────────────────────

def skull_strip(
    arr: np.ndarray,
    threshold_fraction: float = 0.05,
    closing_radius: int = 5,
) -> np.ndarray:
    """
    Approximate skull-stripping via intensity threshold + morphological closing.

    Parameters
    ----------
    arr                  : float32 array, values in [0, 255]
    threshold_fraction   : pixels below ``fraction * max`` set to 0
    closing_radius       : radius of morphological closing (pixels)

    Returns
    -------
    np.ndarray  background pixels zeroed out
    """
    mx = arr.max()
    if mx < 1e-6:
        return arr
    mask = (arr > threshold_fraction * mx).astype(np.uint8) * 255
    m = Image.fromarray(mask)
    for _ in range(closing_radius):
        m = m.filter(ImageFilter.MaxFilter(3))
    for _ in range(closing_radius):
        m = m.filter(ImageFilter.MinFilter(3))
    return arr * (np.array(m, dtype=np.float32) / 255.0)


# ── Step 4: CLAHE ─────────────────────────────────────────────────────────────

def apply_clahe(
    arr: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    Uses scikit-image when available; falls back to PIL equalise.

    Parameters
    ----------
    arr        : float32 array, arbitrary range
    clip_limit : normalised clip limit (scikit-image convention)
    grid_size  : tile grid (rows, cols)

    Returns
    -------
    np.ndarray  float32, rescaled to [0, 255]
    """
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return arr
    arr_01 = (arr - mn) / (mx - mn)

    try:
        from skimage import exposure
        out = exposure.equalize_adapthist(
            arr_01,
            kernel_size=grid_size,
            clip_limit=clip_limit / 10.0,
        )
    except ImportError:
        logger.warning("scikit-image missing — using PIL equalise fallback.")
        pil = Image.fromarray((arr_01 * 255).astype(np.uint8))
        out = np.array(pil.convert("L"), dtype=np.float32) / 255.0

    return (out * 255.0).astype(np.float32)


# ── Step 5: Z-score ───────────────────────────────────────────────────────────

def z_score_normalise(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalise over brain-tissue pixels (non-zero).

    Parameters
    ----------
    arr : float32 array
    eps : prevents division by zero

    Returns
    -------
    np.ndarray  brain pixels have zero mean, unit std
    """
    mask = arr > 0
    if mask.sum() < 10:
        return arr
    out = arr.copy()
    out[mask] = (arr[mask] - arr[mask].mean()) / (arr[mask].std() + eps)
    return out


# ── Step 6: Artifact check ────────────────────────────────────────────────────

def check_artifact(
    arr: np.ndarray,
    min_mean: float = 5.0,
    min_nonzero_frac: float = 0.05,
) -> Tuple[bool, str]:
    """
    Heuristic blank / corrupt image detection.

    Returns
    -------
    (is_valid: bool, reason: str)
        ``is_valid=True`` means the image passed all checks.
    """
    if arr.size == 0:
        return False, "empty array"
    if arr.mean() < min_mean:
        return False, f"mean {arr.mean():.2f} < {min_mean}"
    if np.count_nonzero(arr) / arr.size < min_nonzero_frac:
        return False, "too many zero pixels"
    return True, "ok"


# ── Step 7: Save ──────────────────────────────────────────────────────────────

def save_preprocessed(arr: np.ndarray, out_path: Path) -> None:
    """Rescale float32 array to uint8 [0, 255] and save as PNG."""
    mn, mx = arr.min(), arr.max()
    uint8 = ((arr - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8) if mx > mn \
            else np.zeros_like(arr, dtype=np.uint8)
    Image.fromarray(uint8).save(out_path)


# ── Full single-image pipeline ────────────────────────────────────────────────

def preprocess_image(
    path: Union[str, Path],
    output_size: Tuple[int, int] = (176, 208),
    do_skull_strip: bool = True,
    do_clahe: bool = True,
    do_zscore: bool = True,
) -> Optional[np.ndarray]:
    """
    Run the complete preprocessing pipeline on a single MRI image.

    Parameters
    ----------
    path         : input image file
    output_size  : (width, height)
    do_skull_strip, do_clahe, do_zscore : toggle individual steps

    Returns
    -------
    np.ndarray or None
        Preprocessed float32 array, or None when artifact check fails.
    """
    arr = load_grayscale(path)
    arr = resize(arr, size=output_size)
    if do_skull_strip:
        arr = skull_strip(arr)
    if do_clahe:
        arr = apply_clahe(arr)
    if do_zscore:
        arr = z_score_normalise(arr)
    ok, reason = check_artifact(arr)
    if not ok:
        logger.warning("Artifact in %s: %s", Path(path).name, reason)
        return None
    return arr


# ── Batch processor ───────────────────────────────────────────────────────────

class DatasetPreprocessor:
    """
    Batch preprocessor: walks the entire raw dataset and writes to an
    output directory that mirrors the original train/test/class structure.

    Parameters
    ----------
    raw_root       : root of the raw Kaggle dataset
    processed_root : output root (created if missing)
    output_size    : (width, height) of processed images
    do_skull_strip, do_clahe, do_zscore : toggle pipeline steps

    Examples
    --------
    >>> proc = DatasetPreprocessor("data/raw", "data/processed")
    >>> stats = proc.run()
    >>> stats["total"], stats["processed"], stats["skipped"]
    (6400, 6392, 8)
    """

    def __init__(
        self,
        raw_root: Union[str, Path],
        processed_root: Union[str, Path],
        output_size: Tuple[int, int] = (176, 208),
        do_skull_strip: bool = True,
        do_clahe: bool = True,
        do_zscore: bool = True,
    ) -> None:
        self.raw_root       = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.output_size    = output_size
        self.do_skull_strip = do_skull_strip
        self.do_clahe       = do_clahe
        self.do_zscore      = do_zscore

    def run(self) -> Dict:
        """
        Process every image and write results to ``processed_root``.

        Returns
        -------
        dict  summary with keys: total, processed, skipped, per_class
        """
        stats: Dict = {"total": 0, "processed": 0, "skipped": 0, "per_class": {}}

        for split in ("train", "test"):
            split_dir = self.raw_root / split
            if not split_dir.exists():
                logger.warning("Split dir missing: %s", split_dir)
                continue
            for cls in CLASS_NAMES:
                in_dir  = split_dir / cls
                out_dir = self.processed_root / split / cls
                if not in_dir.exists():
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                key = f"{split}/{cls}"
                stats["per_class"][key] = {"processed": 0, "skipped": 0}

                for p in sorted(in_dir.iterdir()):
                    if p.suffix.lower() not in VALID_EXTS:
                        continue
                    stats["total"] += 1
                    out_path = out_dir / (p.stem + ".png")
                    try:
                        arr = preprocess_image(
                            p, self.output_size,
                            self.do_skull_strip, self.do_clahe, self.do_zscore,
                        )
                        if arr is None:
                            stats["skipped"] += 1
                            stats["per_class"][key]["skipped"] += 1
                        else:
                            save_preprocessed(arr, out_path)
                            stats["processed"] += 1
                            stats["per_class"][key]["processed"] += 1
                    except Exception as e:
                        logger.error("Error %s: %s", p, e)
                        stats["skipped"] += 1
                        stats["per_class"][key]["skipped"] += 1

        logger.info(
            "Done — total=%d  processed=%d  skipped=%d",
            stats["total"], stats["processed"], stats["skipped"],
        )
        return stats

    def compute_normalisation_stats(self) -> Dict[str, float]:
        """
        Compute pixel mean and std across all processed training images.

        Returns
        -------
        dict  with keys ``mean`` and ``std`` (float, range [0,255])
        """
        all_px: List[np.ndarray] = []
        for cls in CLASS_NAMES:
            cls_dir = self.processed_root / "train" / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix == ".png":
                    all_px.append(load_grayscale(p).flatten())
        if not all_px:
            raise RuntimeError("No processed training images found.")
        flat = np.concatenate(all_px)
        return {"mean": float(flat.mean()), "std": float(flat.std())}
