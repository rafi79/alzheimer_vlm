#!/usr/bin/env python3
"""
scripts/download_dataset.py
============================
Download and extract the Alzheimer's MRI Dataset from Kaggle.

Usage
-----
    # Kaggle API (recommended)
    python scripts/download_dataset.py --method kaggle

    # Already-downloaded zip
    python scripts/download_dataset.py --method manual --zip_path ~/Downloads/archive.zip

Prerequisites
-------------
    pip install kaggle
    Place ~/.kaggle/kaggle.json  (chmod 600)
    OR: export KAGGLE_USERNAME=... KAGGLE_KEY=...

Dataset
-------
    tourist55/alzheimers-dataset-4-class-of-images
    https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images
"""

import argparse, logging, shutil, sys, zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KAGGLE_DATASET  = "tourist55/alzheimers-dataset-4-class-of-images"
DEFAULT_OUT     = Path("data/raw")
CLASS_NAMES     = ["NonDemented","VeryMildDemented","MildDemented","ModerateDemented"]
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def download_kaggle(out_dir: Path) -> Path:
    try:
        import kaggle
    except ImportError:
        log.error("kaggle not installed — run: pip install kaggle"); sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading '%s' …", KAGGLE_DATASET)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(out_dir),
                                       unzip=False, quiet=False)
    zips = list(out_dir.glob("*.zip"))
    if not zips:
        log.error("No zip found after download."); sys.exit(1)
    log.info("Downloaded: %s", zips[0]); return zips[0]


def extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s → %s …", zip_path.name, dest)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    # Flatten single nested folder if present
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "train").exists():
        inner = children[0]
        log.info("Flattening nested dir: %s", inner.name)
        for item in inner.iterdir():
            if not (dest / item.name).exists():
                shutil.move(str(item), str(dest / item.name))
        inner.rmdir()
    _verify(dest)


def _verify(root: Path) -> None:
    missing = [str(root / s / c)
               for s in ("train","test") for c in CLASS_NAMES
               if not (root / s / c).exists()]
    if missing:
        log.warning("Missing dirs:\n  %s", "\n  ".join(missing))
    else:
        log.info("✓ Dataset layout verified.")


def _summary(root: Path) -> None:
    print("\n" + "=" * 48)
    print("  Dataset Download Summary")
    print("=" * 48)
    total = 0
    for split in ("train","test"):
        st = root / split
        if not st.exists(): continue
        print(f"\n  {split.upper()}")
        print(f"  {'Class':<22} {'Images':>8}")
        print("  " + "-" * 30)
        for cls in CLASS_NAMES:
            n = sum(1 for f in (st/cls).iterdir()
                    if f.suffix.lower() in VALID_EXTS) \
                if (st/cls).exists() else 0
            print(f"  {cls:<22} {n:>8,}"); total += n
    print(f"\n  Grand total: {total:,}\n" + "=" * 48 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Download Kaggle Alzheimer's MRI dataset.")
    ap.add_argument("--method", choices=["kaggle","manual"], default="kaggle")
    ap.add_argument("--zip_path", default=None, help="Path for --method manual")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--keep_zip", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if args.method == "kaggle":
        zip_path = download_kaggle(out / "_tmp")
    else:
        if not args.zip_path:
            ap.error("--zip_path required with --method manual")
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            log.error("zip not found: %s", zip_path); sys.exit(1)

    extract(zip_path, out)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        tmp = out / "_tmp"
        if tmp.exists(): shutil.rmtree(tmp)

    _summary(out)


if __name__ == "__main__":
    main()
