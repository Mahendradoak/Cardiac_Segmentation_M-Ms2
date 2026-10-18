"""PyTorch datasets: 2D slices for training, whole volumes for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from .data import load_volume, normalise
from .labels import MNMS2_RAW, remap


@dataclass
class GridParams:
    orig_shape: tuple[int, int, int]
    resampled_hw: tuple[int, int]
    pad: tuple[int, int, int, int]   # top, bottom, left, right (negative = crop)


class Preprocessor:
    """Resample in-plane to a fixed spacing, then centre pad/crop to ``size``.

    Unlike a plain cv2.resize to 256x256, this preserves aspect ratio and
    physical scale, and is invertible so predictions can be scored on the
    original voxel grid.
    """

    def __init__(self, spacing_mm: float = 1.25, size: int = 256):
        self.spacing_mm, self.size = spacing_mm, size

    def _fit(self, arr2d_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = arr2d_shape
        dh, dw = self.size - h, self.size - w
        return dh // 2, dh - dh // 2, dw // 2, dw - dw // 2

    @staticmethod
    def _pad_crop(x: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
        t, b, l, r = pad
        x = np.pad(x, ((max(t, 0), max(b, 0)), (max(l, 0), max(r, 0))) + ((0, 0),) * (x.ndim - 2))
        h, w = x.shape[:2]
        return x[max(-t, 0): h - max(-b, 0), max(-l, 0): w - max(-r, 0)]

    def forward(self, vol: np.ndarray, spacing: tuple[float, float, float], order: int) -> tuple[np.ndarray, GridParams]:
        zoom = (spacing[0] / self.spacing_mm, spacing[1] / self.spacing_mm, 1.0)
        res = ndimage.zoom(vol, zoom, order=order, mode="nearest", grid_mode=True)
        pad = self._fit(res.shape[:2])
        out = self._pad_crop(res, pad)
        return out, GridParams(tuple(vol.shape), tuple(res.shape[:2]), pad)

    def inverse(self, mask: np.ndarray, p: GridParams) -> np.ndarray:
        t, b, l, r = p.pad
        un = self._pad_crop(mask, (-t, -b, -l, -r))
        assert un.shape[:2] == p.resampled_hw, (un.shape, p.resampled_hw)
        zoom = [o / s for o, s in zip(p.orig_shape[:2], un.shape[:2], strict=True)] + [1.0]
        back = ndimage.zoom(un, zoom, order=0, mode="nearest", grid_mode=True)
        return back[: p.orig_shape[0], : p.orig_shape[1], : p.orig_shape[2]]


def load_case(row: pd.Series, pre: Preprocessor, raw_map: dict[int, str] = MNMS2_RAW) -> dict:
    img, spacing = load_volume(row.image)
    lab, lab_spacing = load_volume(row.label)
    if img.shape != lab.shape or not np.allclose(spacing, lab_spacing, atol=1e-3):
        raise ValueError(f"image/label grid mismatch: {row.image} {img.shape} {spacing} "
                         f"vs {row.label} {lab.shape} {lab_spacing}")
    gt = remap(lab, raw_map)
    img_p, grid = pre.forward(normalise(img), spacing, order=1)
    gt_p, _ = pre.forward(gt, spacing, order=0)
    return dict(image=img_p, mask=gt_p, gt_orig=gt, spacing=spacing, grid=grid,
                subject=int(row.subject), view=row.view, phase=row.phase)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rotation, isotropic scaling, gamma and noise. No flips: left/right flips
    swap the anatomical side of the RV and teach the model an impossible heart."""
    angle = np.deg2rad(rng.uniform(-15, 15))
    scale = rng.uniform(0.9, 1.1)
    c, s = np.cos(angle) / scale, np.sin(angle) / scale
    mat = np.array([[c, -s], [s, c]])
    centre = (np.array(img.shape) - 1) / 2
    offset = centre - mat @ centre + rng.uniform(-8, 8, size=2)
    img = ndimage.affine_transform(img, mat, offset=offset, order=1, mode="constant", cval=float(img.min()))
    mask = ndimage.affine_transform(mask, mat, offset=offset, order=0, mode="constant", cval=0)
    lo, hi = img.min(), img.max()
    if hi > lo and rng.random() < 0.5:
        u = (img - lo) / (hi - lo)
        img = u ** rng.uniform(0.7, 1.5) * (hi - lo) + lo
    if rng.random() < 0.3:
        img = img + rng.normal(0, 0.05, img.shape)
    return img.astype(np.float32), mask.astype(np.uint8)


class SliceDataset(Dataset):
    """Every short-axis slice of every labelled ED/ES volume, as a 2D sample."""

    def __init__(self, cases: list[dict], train: bool):
        self.items = [(ci, z) for ci, c in enumerate(cases) for z in range(c["image"].shape[2])]
        self.cases, self.train = cases, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ci, z = self.items[i]
        img = self.cases[ci]["image"][..., z]
        msk = self.cases[ci]["mask"][..., z]
        if self.train:
            # torch.initial_seed() differs per worker and per epoch, so
            # augmentations are not duplicated across DataLoader workers.
            rng = np.random.default_rng([torch.initial_seed() % 2**32, i])
            img, msk = augment(img, msk, rng)
        return torch.from_numpy(img[None].copy()), torch.from_numpy(msk.astype(np.int64))
