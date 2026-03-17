# MedVLM-AD — Alzheimer's MRI Dataset Processing Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)


---


## 🧠 Overview

This repository implements the **complete data processing pipeline** for **MedVLM-AD** — a Vision-Language Model that achieves **99.14 % accuracy** on the Kaggle Alzheimer's MRI Dataset by fusing MRI visual features with structured clinical language descriptions.

```
Raw Kaggle ZIP
    │
    ▼  scripts/download_dataset.py
Download & Extract
    │
    ▼  scripts/run_preprocessing.py
MRI Preprocessing
  ├─ Skull-stripping
  ├─ Resize → 176 × 208
  ├─ CLAHE contrast enhancement
  └─ Z-score intensity normalisation
    │
    ▼  scripts/run_eda.py
EDA & Visualisation
    │
    ▼  src/dataset/alzheimer_dataset.py
PyTorch Dataset
  ├─ 4-class integer labels
  ├─ Clinical text descriptions (VLM language branch)
  ├─ Stratified train / val split (85 / 15 %)
  └─ WeightedRandomSampler (handles class imbalance)
    │
    ▼  src/augmentation/augmentation.py
Augmentation
  ├─ Spatial  — flip, affine, blur
  ├─ Intensity — noise, gamma, bias-field shift
  ├─ MixUp / CutMix
  └─ Brain-aware random erasing
    │
    ▼
Model-Ready DataLoaders → Training / Evaluation
```

---

## 📊 Dataset

**Alzheimer's MRI Dataset** · [Kaggle link](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)

| Class | Train | Test | Total |
|---|---|---|---|
| NonDemented | 2 560 | 640 | 3 200 |
| VeryMildDemented | 1 792 | 448 | 2 240 |
| MildDemented | 717 | 179 | 896 |
| ModerateDemented | 52 | 12 | 64 |
| **Total** | **5 121** | **1 279** | **6 400** |

> ⚠️ **Class imbalance**: ModerateDemented has only 64 samples.
> The pipeline compensates via `WeightedRandomSampler` and prototype-based loss.

**Expected directory layout after download:**

```
data/raw/
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
```

---

## 📁 Project Structure

```
alzheimer_vlm/
│
├── configs/
│   └── default.yaml              # All hyperparameters & paths
│
├── data/
│   ├── raw/                      # Original Kaggle images  (git-ignored)
│   └── processed/                # Preprocessed PNGs       (git-ignored)
│
├── src/
│   ├── dataset/
│   │   └── alzheimer_dataset.py  # Dataset class + DataLoader factory
│   ├── preprocessing/
│   │   └── preprocessor.py       # Skull-strip · CLAHE · Z-score · batch runner
│   ├── augmentation/
│   │   └── augmentation.py       # MixUp · CutMix · TTA · custom transforms
│   └── utils/
│       └── utils.py              # Metrics · plots · checkpointing · logging
│
├── scripts/
│   ├── download_dataset.py       # Kaggle API download & extraction
│   ├── run_preprocessing.py      # CLI for the preprocessing pipeline
│   └── run_eda.py                # Generate all EDA plots
│
├── tests/
│   └── test_dataset.py           # pytest unit tests (40+ tests)
│
├── results/                      # Generated plots    (git-ignored)
├── checkpoints/                  # Model weights      (git-ignored)
├── logs/                         # Training logs      (git-ignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1 · Clone

```bash
git clone https://github.com/<your-username>/alzheimer_vlm.git
cd alzheimer_vlm
```

### 2 · Virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Kaggle API credentials

```bash
# Put your kaggle.json at:
#   Linux/macOS : ~/.kaggle/kaggle.json   (chmod 600)
#   Windows     : C:\Users\<user>\.kaggle\kaggle.json

# OR export environment variables:
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

> Get your key from [kaggle.com → Account → Create New API Token](https://www.kaggle.com/settings).

---

## 🚀 Quick Start

Four commands to go from zero to model-ready tensors:

```bash
# 1 — Download
python scripts/download_dataset.py --method kaggle --output_dir data/raw

# 2 — Preprocess
python scripts/run_preprocessing.py \
    --raw_root data/raw --out_root data/processed --stats

# 3 — EDA
python scripts/run_eda.py --data_root data/processed

# 4 — Use in your training script
python - << 'EOF'
from src.dataset.alzheimer_dataset import build_dataloaders

loaders = build_dataloaders("data/processed", img_size=224, batch_size=64)
imgs, labels, texts = next(iter(loaders["train"]))
print(imgs.shape)    # torch.Size([64, 1, 224, 224])
print(texts[0][:60])
EOF
```

---

## 🔬 Pipeline — Step by Step

### Step 1 · Download

```bash
# Kaggle API
python scripts/download_dataset.py --method kaggle

# Already-downloaded zip
python scripts/download_dataset.py --method manual --zip_path ~/Downloads/archive.zip

# Keep the zip after extraction
python scripts/download_dataset.py --keep_zip
```

Sample output:
```
================================================
  Dataset Download Summary
================================================
  TRAIN
  Class                    Images
  ──────────────────────────────
  NonDemented               2,560
  VeryMildDemented          1,792
  MildDemented                717
  ModerateDemented             52
  TOTAL                     5,121

  Grand total: 6,400
================================================
```

---

### Step 2 · Preprocessing

```bash
# Full recommended pipeline
python scripts/run_preprocessing.py \
    --raw_root data/raw --out_root data/processed --stats

# Skip skull-stripping (faster)
python scripts/run_preprocessing.py --no_skull

# Custom output resolution
python scripts/run_preprocessing.py --width 224 --height 224
```

**Pipeline steps in order:**

| # | Function | Description |
|---|---|---|
| 1 | `load_grayscale()` | Open image → float32 NumPy array |
| 2 | `resize()` | Bicubic resize to (W × H) |
| 3 | `skull_strip()` | Threshold + morphological closing |
| 4 | `apply_clahe()` | Contrast-limited adaptive histogram equalisation |
| 5 | `z_score_normalise()` | Brain-tissue pixels → μ=0, σ=1 |
| 6 | `check_artifact()` | Heuristic blank/corrupt detection |
| 7 | `save_preprocessed()` | Rescale to uint8, save as PNG |

**Python API:**

```python
from src.preprocessing.preprocessor import preprocess_image, DatasetPreprocessor

# ── Single image ──────────────────────────────────────────────────────────────
arr = preprocess_image(
    "data/raw/train/MildDemented/mildDem0.jpg",
    output_size=(176, 208),
    do_skull_strip=True,
    do_clahe=True,
    do_zscore=True,
)
# arr: float32 NumPy array (208, 176)  or  None if artifact detected

# ── Entire dataset ────────────────────────────────────────────────────────────
proc = DatasetPreprocessor("data/raw", "data/processed")
stats = proc.run()
print(stats)
# {'total': 6400, 'processed': 6392, 'skipped': 8, 'per_class': {...}}

# ── Pixel statistics ──────────────────────────────────────────────────────────
ns = proc.compute_normalisation_stats()
# {'mean': 118.4, 'std': 61.2}
```

---

### Step 3 · EDA

```bash
python scripts/run_eda.py --data_root data/processed --n_samples 4
```

**Outputs saved to `results/eda/`:**

| File | Contents |
|---|---|
| `class_distribution_train.png` | Training class count bar chart |
| `class_distribution_val.png` | Validation class count bar chart |
| `class_distribution_test.png` | Test class count bar chart |
| `sample_grid_train.png` | 4 × 4 MRI sample grid per class |
| `pixel_intensity_histograms.png` | Per-class pixel intensity distributions |

---

### Step 4 · Dataset & DataLoaders

```python
from src.dataset.alzheimer_dataset import (
    AlzheimerMRIDataset,
    build_dataloaders,
    get_transforms,
    CLASS_NAMES,       # ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    CLASS_TO_IDX,      # {"NonDemented": 0, ...}
    CLINICAL_TEXTS,    # radiology-style text description per class
)

# ── Individual splits ─────────────────────────────────────────────────────────
train_tf, val_tf = get_transforms(img_size=224)

train_ds = AlzheimerMRIDataset(
    root_dir="data/processed",
    split="train",
    transform=train_tf,
    return_text=True,        # return (img, label, clinical_text)
    val_fraction=0.15,
    seed=42,
)

img, label, text = train_ds[0]
# img   : torch.Tensor  (1, 224, 224)  — grayscale
# label : int           0 / 1 / 2 / 3
# text  : str           clinical description for the VLM language branch

print(train_ds)
# AlzheimerMRIDataset(split='train', n=4352, ...)

# ── All splits in one call ────────────────────────────────────────────────────
loaders = build_dataloaders(
    root_dir="data/processed",
    img_size=224,
    batch_size=64,
    num_workers=4,
    use_weighted_sampler=True,   # recommended — compensates for class imbalance
)
# loaders = {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}

# ── Class-balanced loss ───────────────────────────────────────────────────────
import torch
class_weights = train_ds.get_class_weights()
# tensor([0.1098, 0.1568, 0.3915, 0.3419])  ← rare classes get higher weight
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# ── Export metadata ───────────────────────────────────────────────────────────
df = train_ds.to_dataframe()
df.to_csv("results/train_metadata.csv", index=False)
```

**Class index → name → clinical text mapping:**

| Index | Class Name | Clinical Description (excerpt) |
|---|---|---|
| 0 | NonDemented | Normal hippocampal volume, no cortical atrophy… |
| 1 | VeryMildDemented | Minimal hippocampal volume loss, early entorhinal… |
| 2 | MildDemented | Moderate hippocampal atrophy, parietal cortex thinning… |
| 3 | ModerateDemented | Severe global cortical atrophy, enlarged ventricles… |

---

### Step 5 · Augmentation

```python
from src.augmentation.augmentation import (
    get_train_transform, get_val_transform,
    MixUp, CutMix,
    TTAWrapper, tta_transforms,
)

# ── Standard training pipeline ────────────────────────────────────────────────
train_tf = get_train_transform(img_size=224)

# ── Heavy pipeline (extra gamma, bias-field, brain-erasing) ──────────────────
heavy_tf = get_train_transform(img_size=224, heavy=True)

# ── MixUp ─────────────────────────────────────────────────────────────────────
mixup = MixUp(alpha=0.4, num_classes=4)
mixed_imgs, soft_labels = mixup(imgs, labels)
# soft_labels shape: (B, 4) — use with F.cross_entropy directly

loss = torch.nn.functional.cross_entropy(logits, soft_labels)

# ── CutMix ────────────────────────────────────────────────────────────────────
cutmix = CutMix(alpha=1.0, num_classes=4)
mixed_imgs, soft_labels = cutmix(imgs, labels)

# ── Test-Time Augmentation (5-view averaging) ─────────────────────────────────
tta = TTAWrapper(
    model=trained_model,
    transforms_list=tta_transforms(img_size=224),
    device="cuda",
)
probs = tta.predict(image_tensor, text="Moderate hippocampal atrophy…")
# probs: torch.Tensor (4,)  — averaged softmax over 5 views
predicted_class = probs.argmax().item()
```

**Custom transforms summary:**

| Class | Effect | Key params |
|---|---|---|
| `AddGaussianNoise` | Additive zero-mean noise | `sigma_range`, `p` |
| `RandomGamma` | Gamma intensity correction | `gamma_range`, `p` |
| `RandomIntensityShift` | Scanner bias-field simulation | `shift_range`, `scale_range`, `p` |
| `BrainAwareRandomErasing` | Noise-filled rectangular erasure | `scale`, `ratio`, `noise_std` |
| `MixUp` | Batch label-mixing | `alpha`, `num_classes` |
| `CutMix` | Batch patch-mixing | `alpha`, `num_classes` |
| `TTAWrapper` | 5-view inference averaging | `transforms_list`, `device` |

---

## ⚙️ Configuration

All hyper-parameters live in `configs/default.yaml`:

```yaml
data:
  img_size:      224
  val_fraction:  0.15

preprocessing:
  output_size:   [176, 208]
  do_skull_strip: true
  do_clahe:       true
  do_zscore:      true

training:
  epochs:        50
  batch_size:    64
  optimizer:
    lr:          3.0e-4

loss:
  lambda_haa:    0.4   # HAA total weight
  lambda_patch:  0.3   # L_P — patch-text alignment
  lambda_slice:  0.5   # L_S — slice-description InfoNCE
  lambda_class:  0.2   # L_C — prototype class alignment
  temperature:   0.07
```

Load in Python:

```python
import yaml
with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)
lr = cfg["training"]["optimizer"]["lr"]   # 3e-4
```

---

## 📖 Module Reference

### `src/dataset/alzheimer_dataset.py`

| Symbol | Type | Purpose |
|---|---|---|
| `AlzheimerMRIDataset` | `Dataset` | Main dataset class |
| `build_dataloaders()` | function | Returns `{train, val, test}` DataLoaders |
| `get_transforms()` | function | Returns `(train_tf, val_tf)` |
| `CLASS_NAMES` | `List[str]` | Ordered class names |
| `CLASS_TO_IDX` | `Dict[str,int]` | Name → index |
| `IDX_TO_CLASS` | `Dict[int,str]` | Index → name |
| `CLINICAL_TEXTS` | `Dict[str,str]` | Radiology text per class |

### `src/preprocessing/preprocessor.py`

| Symbol | Type | Purpose |
|---|---|---|
| `load_grayscale()` | function | Load → float32 NumPy |
| `resize()` | function | Bicubic resize |
| `skull_strip()` | function | Threshold + morphological mask |
| `apply_clahe()` | function | CLAHE enhancement |
| `z_score_normalise()` | function | Brain-tissue Z-score |
| `check_artifact()` | function | Corrupt-image detection |
| `preprocess_image()` | function | Full single-image pipeline |
| `save_preprocessed()` | function | Float32 → uint8 PNG |
| `DatasetPreprocessor` | class | Batch dataset processor |

### `src/augmentation/augmentation.py`

| Symbol | Type | Purpose |
|---|---|---|
| `get_train_transform()` | function | Standard / heavy train pipeline |
| `get_val_transform()` | function | Deterministic val/test pipeline |
| `AddGaussianNoise` | transform | Additive Gaussian noise |
| `RandomGamma` | transform | Gamma correction |
| `RandomIntensityShift` | transform | Bias-field simulation |
| `BrainAwareRandomErasing` | transform | Noise-filled rect erasing |
| `MixUp` | class | Batch MixUp with soft labels |
| `CutMix` | class | Batch CutMix with soft labels |
| `TTAWrapper` | class | 5-view TTA inference |
| `tta_transforms()` | function | List of 5 deterministic TTA pipelines |

### `src/utils/utils.py`

| Symbol | Type | Purpose |
|---|---|---|
| `compute_accuracy()` | function | Top-1 accuracy |
| `compute_metrics()` | function | Acc · F1 · precision · recall |
| `compute_confusion_matrix()` | function | (K×K) confusion matrix |
| `plot_class_distribution()` | function | Class bar chart |
| `plot_sample_grid()` | function | MRI sample grid |
| `plot_confusion_matrix()` | function | Normalised heatmap |
| `plot_training_curves()` | function | Loss + accuracy curves |
| `plot_intensity_histograms()` | function | Per-class pixel distributions |
| `save_checkpoint()` | function | Save model + optimizer |
| `load_checkpoint()` | function | Restore from checkpoint |
| `set_seed()` | function | Full reproducibility |
| `setup_logger()` | function | Console + file logger |
| `summarise_dataset()` | function | Count images per split/class |
| `print_dataset_summary()` | function | Pretty-print summary |

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific class
pytest tests/test_dataset.py::TestPreprocessor -v

# Specific test
pytest tests/test_dataset.py::TestAugmentation::test_mixup_labels_sum_to_one -v
```

**Coverage breakdown (40+ tests):**

| Area | Tests |
|---|---|
| Dataset loading (all 3 splits) | 4 |
| `__getitem__` return types & shapes | 3 |
| Class/sample weights | 2 |
| DataFrame export | 1 |
| Error handling | 2 |
| DataLoader batch shapes | 2 |
| Preprocessing functions | 8 |
| Batch `DatasetPreprocessor.run()` | 1 |
| Augmentation transforms | 8 |
| MixUp / CutMix label properties | 4 |
| TTA transform count | 1 |
| Utility functions | 5 |

---

## 📈 Results

Performance on the Kaggle Alzheimer's MRI Dataset test set (1 279 images):

| Method | Acc (%) | Macro F1 (%) | AUC |
|---|---|---|---|
| VGG-16 | 86.02 | 85.04 | 0.931 |
| ResNet-50 | 90.33 | 89.52 | 0.952 |
| DenseNet-121 | 92.10 | 91.76 | 0.963 |
| EfficientNet-B4 | 94.68 | 94.13 | 0.974 |
| Swin-Transformer | 95.86 | 95.30 | 0.981 |
| MedCLIP | 96.72 | 96.28 | 0.986 |
| BioViL | 96.87 | 96.52 | 0.988 |
| **MedVLM-AD (ours)** | **99.14** | **99.02** | **0.997** |



