#!/usr/bin/env python3
"""
scripts/run_eda.py
==================
Exploratory Data Analysis — generates all EDA plots.

Outputs (results/eda/)
-----------------------
  class_distribution_train.png
  class_distribution_val.png
  class_distribution_test.png
  sample_grid_train.png
  pixel_intensity_histograms.png

Usage
-----
    python scripts/run_eda.py --data_root data/processed
"""

import argparse, sys
import matplotlib; matplotlib.use("Agg")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
from src.utils.utils import (
    plot_class_distribution, plot_sample_grid,
    plot_intensity_histograms, print_dataset_summary, setup_logger,
)


def main():
    ap = argparse.ArgumentParser(description="Generate EDA plots.")
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--out_dir",   default="results/eda")
    ap.add_argument("--n_samples", type=int, default=4)
    args = ap.parse_args()

    log = setup_logger("eda")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print_dataset_summary(args.data_root)
    _, val_tf = get_transforms(img_size=224)

    for split in ("train", "val", "test"):
        log.info("Building %s dataset …", split)
        ds = AlzheimerMRIDataset(args.data_root, split, val_tf)
        plot_class_distribution(ds, save_path=out / f"class_distribution_{split}.png")
        log.info("  → class_distribution_%s.png", split)

    train_ds = AlzheimerMRIDataset(args.data_root, "train", val_tf)
    plot_sample_grid(train_ds, n_per_class=args.n_samples,
                     save_path=out / "sample_grid_train.png")
    log.info("  → sample_grid_train.png")

    plot_intensity_histograms(train_ds,
                              save_path=out / "pixel_intensity_histograms.png")
    log.info("  → pixel_intensity_histograms.png")
    log.info("EDA complete. All outputs in: %s", out)


if __name__ == "__main__":
    main()
