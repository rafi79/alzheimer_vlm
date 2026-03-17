"""
augmentation.py
===============
MRI-specific data augmentation for the MedVLM-AD pipeline.

Contents
--------
AddGaussianNoise        – additive zero-mean Gaussian noise
RandomGamma             – random gamma intensity adjustment
RandomIntensityShift    – simulate scanner bias field
BrainAwareRandomErasing – noise-filled rectangular erasing
get_train_transform     – standard / heavy training pipeline
get_val_transform       – deterministic val/test pipeline
MixUp                   – batch-level label-mixing (Zhang 2018)
CutMix                  – batch-level patch-mixing  (Yun 2019)
TTAWrapper              – 5-view Test-Time Augmentation
tta_transforms          – returns the 5 deterministic TTA pipelines
"""

import math
import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


# ── Custom transform classes ──────────────────────────────────────────────────

class AddGaussianNoise:
    """
    Add zero-mean Gaussian noise to a float tensor.

    Parameters
    ----------
    sigma_range : (float, float)
        Noise std is drawn uniformly from this range per call.
    p : float
        Probability of applying.
    """

    def __init__(self, sigma_range: Tuple[float, float] = (0.005, 0.02), p: float = 0.5):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            return x + torch.randn_like(x) * sigma
        return x

    def __repr__(self) -> str:
        return f"AddGaussianNoise(sigma={self.sigma_range}, p={self.p})"


class RandomGamma:
    """
    Random gamma correction.  Values < 1 brighten; > 1 darken.

    Parameters
    ----------
    gamma_range : (float, float)
    p : float  probability of applying
    """

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.4):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return x.clamp(0.0, 1.0).pow(random.uniform(*self.gamma_range))
        return x

    def __repr__(self) -> str:
        return f"RandomGamma(gamma={self.gamma_range}, p={self.p})"


class RandomIntensityShift:
    """
    Random additive shift + multiplicative scale to simulate scanner
    intensity inhomogeneity / bias fields.

    Parameters
    ----------
    shift_range : (float, float)  additive range
    scale_range : (float, float)  multiplicative range
    p : float
    """

    def __init__(
        self,
        shift_range: Tuple[float, float] = (-0.1, 0.1),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.4,
    ):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return x * random.uniform(*self.scale_range) + random.uniform(*self.shift_range)
        return x

    def __repr__(self) -> str:
        return f"RandomIntensityShift(shift={self.shift_range}, scale={self.scale_range})"


class BrainAwareRandomErasing:
    """
    Randomly erase a rectangular region and fill with Gaussian noise,
    simulating MRI k-space corruption or motion artifacts.

    Parameters
    ----------
    p          : float   probability of applying
    scale      : (float, float)  fraction of image area to erase
    ratio      : (float, float)  aspect ratio range
    noise_std  : float   std of the Gaussian fill
    """

    def __init__(
        self,
        p: float = 0.3,
        scale: Tuple[float, float] = (0.01, 0.08),
        ratio: Tuple[float, float] = (0.3, 3.3),
        noise_std: float = 0.1,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        C, H, W = x.shape
        for _ in range(10):
            area = H * W * random.uniform(*self.scale)
            ar   = math.exp(random.uniform(math.log(self.ratio[0]),
                                            math.log(self.ratio[1])))
            eh = int(math.sqrt(area * ar))
            ew = int(math.sqrt(area / ar))
            if eh >= H or ew >= W:
                continue
            y0 = random.randint(0, H - eh)
            x0 = random.randint(0, W - ew)
            out = x.clone()
            out[:, y0:y0+eh, x0:x0+ew] = torch.randn(C, eh, ew) * self.noise_std
            return out
        return x

    def __repr__(self) -> str:
        return f"BrainAwareRandomErasing(p={self.p}, scale={self.scale})"


# ── Transform pipeline factories ──────────────────────────────────────────────

def get_train_transform(
    img_size: int = 224,
    mean: Tuple[float, ...] = (0.5,),
    std:  Tuple[float, ...] = (0.5,),
    heavy: bool = False,
) -> transforms.Compose:
    """
    Training augmentation pipeline.

    Parameters
    ----------
    img_size : int     target square size
    heavy    : bool    include gamma, intensity-shift, brain-erasing
                       (recommended for minority classes)
    """
    ops = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.10, 0.10),
                                scale=(0.90, 1.10), shear=5),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        AddGaussianNoise(sigma_range=(0.005, 0.015), p=0.4),
    ]
    if heavy:
        ops += [
            RandomGamma(gamma_range=(0.7, 1.4), p=0.4),
            RandomIntensityShift(p=0.4),
            BrainAwareRandomErasing(p=0.3),
        ]
    return transforms.Compose(ops)


def get_val_transform(
    img_size: int = 224,
    mean: Tuple[float, ...] = (0.5,),
    std:  Tuple[float, ...] = (0.5,),
) -> transforms.Compose:
    """Deterministic validation / test pipeline."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ── MixUp ─────────────────────────────────────────────────────────────────────

class MixUp:
    """
    MixUp (Zhang et al., 2018).

    Mixes pairs of images and converts integer labels to soft one-hot
    vectors proportional to the mixing coefficient lambda.

    Parameters
    ----------
    alpha       : float  Beta distribution parameter
    num_classes : int

    Examples
    --------
    >>> mixup = MixUp(alpha=0.4, num_classes=4)
    >>> imgs_m, labels_m = mixup(imgs, labels)   # imgs: (B, C, H, W)
    """

    def __init__(self, alpha: float = 0.4, num_classes: int = 4):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lam  = np.random.beta(self.alpha, self.alpha)
        perm = torch.randperm(images.size(0))
        mixed = lam * images + (1 - lam) * images[perm]
        oh     = F.one_hot(labels, self.num_classes).float()
        oh_p   = F.one_hot(labels[perm], self.num_classes).float()
        return mixed, lam * oh + (1 - lam) * oh_p

    def __repr__(self) -> str:
        return f"MixUp(alpha={self.alpha}, num_classes={self.num_classes})"


# ── CutMix ────────────────────────────────────────────────────────────────────

class CutMix:
    """
    CutMix (Yun et al., 2019).

    Pastes a random rectangular patch from one image onto another and
    mixes labels proportional to the actual patch area.

    Parameters
    ----------
    alpha       : float  Beta distribution parameter
    num_classes : int
    """

    def __init__(self, alpha: float = 1.0, num_classes: int = 4):
        self.alpha = alpha
        self.num_classes = num_classes

    @staticmethod
    def _rand_bbox(W: int, H: int, lam: float) -> Tuple[int, int, int, int]:
        cr = math.sqrt(1.0 - lam)
        cw, ch = int(W * cr), int(H * cr)
        cx = random.randint(0, W)
        cy = random.randint(0, H)
        return (max(cx - cw // 2, 0), max(cy - ch // 2, 0),
                min(cx + cw // 2, W),  min(cy + ch // 2, H))

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lam  = np.random.beta(self.alpha, self.alpha)
        perm = torch.randperm(images.size(0))
        W, H = images.shape[-1], images.shape[-2]
        x1, y1, x2, y2 = self._rand_bbox(W, H, lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam_adj = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
        oh  = F.one_hot(labels, self.num_classes).float()
        ohp = F.one_hot(labels[perm], self.num_classes).float()
        return mixed, lam_adj * oh + (1 - lam_adj) * ohp

    def __repr__(self) -> str:
        return f"CutMix(alpha={self.alpha}, num_classes={self.num_classes})"


# ── Test-Time Augmentation ────────────────────────────────────────────────────

def tta_transforms(img_size: int = 224) -> List[transforms.Compose]:
    """
    Five deterministic TTA pipelines:
    original, h-flip, v-flip, 90° rotation, 270° rotation.
    """
    norm = [transforms.Normalize(mean=(0.5,), std=(0.5,))]
    def pipeline(*extra):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            *extra,
            transforms.ToTensor(),
            *norm,
        ])
    return [
        pipeline(),
        pipeline(transforms.RandomHorizontalFlip(p=1.0)),
        pipeline(transforms.RandomVerticalFlip(p=1.0)),
        pipeline(transforms.RandomRotation(degrees=(90, 90))),
        pipeline(transforms.RandomRotation(degrees=(270, 270))),
    ]


class TTAWrapper:
    """
    Test-Time Augmentation wrapper: run inference over 5 augmented views
    and return the averaged softmax probabilities.

    Parameters
    ----------
    model            : trained nn.Module
    transforms_list  : list of Compose pipelines (from tta_transforms())
    device           : str | torch.device

    Examples
    --------
    >>> tta = TTAWrapper(model, tta_transforms(), device="cuda")
    >>> probs = tta.predict(img_tensor)   # shape (4,)
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        transforms_list: List[Callable],
        device: Union[str, "torch.device"] = "cpu",
    ):
        self.model = model.eval()
        self.tfms  = transforms_list
        self.device = torch.device(device)

    @torch.no_grad()
    def predict(
        self, image: torch.Tensor, text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor  shape (1, H, W) — normalised single image
        text  : str, optional — clinical description for VLM branch

        Returns
        -------
        torch.Tensor  averaged softmax probabilities, shape (num_classes,)
        """
        from PIL import Image as _PIL
        arr = (image.squeeze(0).numpy() * 0.5 + 0.5) * 255
        pil = _PIL.fromarray(arr.astype(np.uint8))

        probs_list = []
        for tf in self.tfms:
            t = tf(pil).unsqueeze(0).to(self.device)
            logits = self.model(t, [text]) if text else self.model(t)
            probs_list.append(torch.softmax(logits, -1).squeeze(0).cpu())
        return torch.stack(probs_list).mean(0)

    def __repr__(self) -> str:
        return f"TTAWrapper(n_views={len(self.tfms)})"
