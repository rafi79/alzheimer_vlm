"""
tests/test_dataset.py
=====================
Unit tests — run with:  pytest tests/ -v --tb=short
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASS_NAMES = ["NonDemented","VeryMildDemented","MildDemented","ModerateDemented"]


# ── Fixture: synthetic dataset ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ad_data")
    for split in ("train","test"):
        for cls in CLASS_NAMES:
            d = root / split / cls; d.mkdir(parents=True)
            for i in range(20):
                arr = np.random.randint(30, 200, (208, 176), dtype=np.uint8)
                Image.fromarray(arr, "L").save(d / f"img_{i:04d}.png")
    return root


# ── Dataset ───────────────────────────────────────────────────────────────────

class TestAlzheimerMRIDataset:

    def test_import(self):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset
        assert AlzheimerMRIDataset

    def test_train_len(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "train", v, val_fraction=0.2)
        assert len(ds) == 64   # 4 × 20 × 0.8

    def test_val_len(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "val", v, val_fraction=0.2)
        assert len(ds) == 16

    def test_test_len(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "test", v)
        assert len(ds) == 80

    def test_getitem_with_text(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "train", v, return_text=True)
        img, lbl, txt = ds[0]
        assert img.shape == (1, 64, 64)
        assert 0 <= int(lbl) <= 3
        assert isinstance(txt, str) and len(txt) > 10

    def test_getitem_without_text(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "train", v, return_text=False)
        item = ds[0]; assert len(item) == 2

    def test_class_weights(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "train", v)
        w = ds.get_class_weights()
        assert w.shape == (4,) and torch.all(w > 0)

    def test_sample_weights(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "train", v)
        assert ds.get_sample_weights().shape == (len(ds),)

    def test_dataframe(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "train", v)
        df = ds.to_dataframe()
        assert len(df) == len(ds)
        assert {"path","label","class_name","clinical_text"}.issubset(df.columns)

    def test_invalid_split(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset
        with pytest.raises(AssertionError):
            AlzheimerMRIDataset(fake_root, "bad_split")

    def test_missing_root(self, tmp_path):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset
        with pytest.raises(FileNotFoundError):
            AlzheimerMRIDataset(tmp_path / "nonexistent", "train")

    def test_repr(self, fake_root):
        from src.dataset.alzheimer_dataset import AlzheimerMRIDataset, get_transforms
        _, v = get_transforms(64)
        ds = AlzheimerMRIDataset(fake_root, "train", v)
        assert "split='train'" in repr(ds)


# ── DataLoaders ───────────────────────────────────────────────────────────────

class TestBuildDataloaders:

    def test_keys(self, fake_root):
        from src.dataset.alzheimer_dataset import build_dataloaders
        loaders = build_dataloaders(fake_root, img_size=64, batch_size=8, num_workers=0)
        assert set(loaders) == {"train","val","test"}

    def test_batch_shape(self, fake_root):
        from src.dataset.alzheimer_dataset import build_dataloaders
        loaders = build_dataloaders(fake_root, img_size=64, batch_size=8, num_workers=0)
        imgs, labels, texts = next(iter(loaders["test"]))
        assert imgs.ndim == 4 and imgs.shape[1] == 1 and imgs.shape[2] == 64
        assert len(labels) == len(texts)


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocessor:

    def test_load_grayscale(self, fake_root):
        from src.preprocessing.preprocessor import load_grayscale
        p = next((fake_root / "train" / CLASS_NAMES[0]).iterdir())
        a = load_grayscale(p)
        assert a.dtype == np.float32 and a.ndim == 2

    def test_resize_shape(self):
        from src.preprocessing.preprocessor import resize
        a = np.random.rand(200, 180).astype(np.float32) * 255
        assert resize(a, (176, 208)).shape == (208, 176)

    def test_skull_strip_nonneg(self):
        from src.preprocessing.preprocessor import skull_strip
        a = np.random.rand(100, 100).astype(np.float32) * 255
        assert skull_strip(a).min() >= 0.0

    def test_zscore_stats(self):
        from src.preprocessing.preprocessor import z_score_normalise
        a = np.random.rand(100, 100).astype(np.float32) * 200 + 30
        out = z_score_normalise(a)
        brain = out[out != 0]
        assert abs(brain.mean()) < 0.1 and abs(brain.std() - 1.0) < 0.1

    def test_artifact_blank(self):
        from src.preprocessing.preprocessor import check_artifact
        ok, _ = check_artifact(np.zeros((100,100), np.float32))
        assert not ok

    def test_artifact_valid(self):
        from src.preprocessing.preprocessor import check_artifact
        a = np.random.rand(100,100).astype(np.float32) * 200 + 10
        ok, reason = check_artifact(a)
        assert ok and reason == "ok"

    def test_full_pipeline_no_raise(self, fake_root):
        from src.preprocessing.preprocessor import preprocess_image
        p = next((fake_root / "train" / CLASS_NAMES[0]).iterdir())
        result = preprocess_image(p, (64, 64))
        assert result is None or result.shape == (64, 64)

    def test_batch_run(self, fake_root, tmp_path):
        from src.preprocessing.preprocessor import DatasetPreprocessor
        proc = DatasetPreprocessor(fake_root, tmp_path/"proc",
                                   output_size=(64,64), do_clahe=False)
        s = proc.run()
        assert s["total"] > 0
        assert s["processed"] + s["skipped"] == s["total"]


# ── Augmentation ──────────────────────────────────────────────────────────────

class TestAugmentation:

    def test_gaussian_noise_changes_tensor(self):
        from src.augmentation.augmentation import AddGaussianNoise
        t = AddGaussianNoise(p=1.0)
        x = torch.zeros(1, 32, 32)
        assert not torch.allclose(t(x), x)

    def test_random_gamma_shape(self):
        from src.augmentation.augmentation import RandomGamma
        t = RandomGamma(p=1.0)
        x = torch.rand(1, 32, 32)
        assert t(x).shape == x.shape

    def test_intensity_shift_shape(self):
        from src.augmentation.augmentation import RandomIntensityShift
        t = RandomIntensityShift(p=1.0)
        x = torch.rand(1, 32, 32)
        assert t(x).shape == x.shape

    def test_brain_erasing_shape(self):
        from src.augmentation.augmentation import BrainAwareRandomErasing
        t = BrainAwareRandomErasing(p=1.0)
        x = torch.rand(1, 64, 64)
        assert t(x).shape == x.shape

    def test_mixup_output_shapes(self):
        from src.augmentation.augmentation import MixUp
        m = MixUp(alpha=0.4, num_classes=4)
        imgs   = torch.rand(8, 1, 64, 64)
        labels = torch.randint(0, 4, (8,))
        mi, ml = m(imgs, labels)
        assert mi.shape == imgs.shape and ml.shape == (8, 4)

    def test_mixup_labels_sum_to_one(self):
        from src.augmentation.augmentation import MixUp
        m = MixUp(alpha=0.4, num_classes=4)
        imgs = torch.rand(8, 1, 32, 32)
        labels = torch.randint(0, 4, (8,))
        _, ml = m(imgs, labels)
        assert torch.allclose(ml.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_cutmix_output_shapes(self):
        from src.augmentation.augmentation import CutMix
        cm = CutMix(alpha=1.0, num_classes=4)
        imgs = torch.rand(8, 1, 64, 64)
        labels = torch.randint(0, 4, (8,))
        mi, ml = cm(imgs, labels)
        assert mi.shape == imgs.shape and ml.shape == (8, 4)

    def test_tta_transforms_count(self):
        from src.augmentation.augmentation import tta_transforms
        assert len(tta_transforms(64)) == 5


# ── Utils ─────────────────────────────────────────────────────────────────────

class TestUtils:

    def test_accuracy(self):
        from src.utils.utils import compute_accuracy
        assert abs(compute_accuracy([0,1,2,3,0,1],[0,1,2,3,1,0]) - 4/6) < 1e-6

    def test_seed_reproducibility(self):
        from src.utils.utils import set_seed
        set_seed(42); a = torch.rand(5)
        set_seed(42); b = torch.rand(5)
        assert torch.allclose(a, b)

    def test_summarise_dataset(self, fake_root):
        from src.utils.utils import summarise_dataset
        s = summarise_dataset(fake_root)
        for cls in CLASS_NAMES:
            assert s["train"][cls] == 20
            assert s["test"][cls]  == 20

    def test_checkpoint_round_trip(self, tmp_path):
        import torch.nn as nn
        from src.utils.utils import save_checkpoint, load_checkpoint
        model = nn.Linear(10, 4)
        opt   = torch.optim.Adam(model.parameters())
        p = save_checkpoint(model, opt, epoch=5,
                            metrics={"val_acc": 0.99}, save_dir=tmp_path)
        m2 = nn.Linear(10, 4)
        _, _, epoch, metrics = load_checkpoint(p, m2)
        assert epoch == 5 and metrics["val_acc"] == 0.99

    def test_compute_metrics_keys(self):
        from src.utils.utils import compute_metrics
        preds  = [0,1,2,3,0,1,2,3]
        labels = [0,1,2,3,1,0,3,2]
        m = compute_metrics(preds, labels)
        for k in ("accuracy","macro_f1","weighted_f1","macro_precision","macro_recall"):
            assert k in m

    def test_confusion_matrix_shape(self):
        from src.utils.utils import compute_confusion_matrix
        cm = compute_confusion_matrix([0,1,2,3],[0,1,2,3])
        assert cm.shape == (4, 4)
