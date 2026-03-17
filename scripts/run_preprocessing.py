#!/usr/bin/env python3
"""
scripts/run_preprocessing.py
=============================
CLI for the full MRI preprocessing pipeline.

Usage
-----
    python scripts/run_preprocessing.py                        # all defaults
    python scripts/run_preprocessing.py --no_skull --no_clahe  # minimal
    python scripts/run_preprocessing.py --stats                # compute mean/std
"""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.preprocessor import DatasetPreprocessor
from src.utils.utils import setup_logger, print_dataset_summary


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess the Alzheimer's MRI Dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw_root",  default="data/raw")
    ap.add_argument("--out_root",  default="data/processed")
    ap.add_argument("--width",     type=int, default=176)
    ap.add_argument("--height",    type=int, default=208)
    ap.add_argument("--no_skull",  action="store_true", help="Skip skull-stripping")
    ap.add_argument("--no_clahe",  action="store_true", help="Skip CLAHE")
    ap.add_argument("--no_zscore", action="store_true", help="Skip Z-score normalisation")
    ap.add_argument("--stats",     action="store_true", help="Compute pixel mean/std")
    ap.add_argument("--log_file",  default="logs/preprocess.log")
    args = ap.parse_args()

    log = setup_logger("preprocess", log_file=args.log_file)
    log.info("=" * 58)
    log.info("  MedVLM-AD — Preprocessing Pipeline")
    log.info("=" * 58)
    log.info("  raw   : %s", args.raw_root)
    log.info("  out   : %s", args.out_root)
    log.info("  size  : %d × %d", args.width, args.height)
    log.info("  skull : %s | clahe : %s | zscore : %s",
             not args.no_skull, not args.no_clahe, not args.no_zscore)

    print_dataset_summary(args.raw_root)

    proc = DatasetPreprocessor(
        raw_root=args.raw_root,
        processed_root=args.out_root,
        output_size=(args.width, args.height),
        do_skull_strip=not args.no_skull,
        do_clahe=not args.no_clahe,
        do_zscore=not args.no_zscore,
    )

    stats = proc.run()
    out_dir = Path(args.out_root)
    (out_dir / "preprocessing_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    log.info("Stats saved → %s/preprocessing_stats.json", out_dir)

    print_dataset_summary(args.out_root)

    if args.stats:
        log.info("Computing pixel mean/std …")
        ns = proc.compute_normalisation_stats()
        norm = {
            "mean_raw": ns["mean"], "std_raw": ns["std"],
            "mean_norm": ns["mean"] / 255.0,
            "std_norm":  ns["std"]  / 255.0,
        }
        log.info("  mean=%.4f  std=%.4f  (raw [0,255])", ns["mean"], ns["std"])
        log.info("  mean=%.4f  std=%.4f  (norm [0,1])", norm["mean_norm"], norm["std_norm"])
        (out_dir / "normalisation_stats.json").write_text(
            json.dumps(norm, indent=2), encoding="utf-8"
        )
        log.info("Norm stats saved → %s/normalisation_stats.json", out_dir)

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
