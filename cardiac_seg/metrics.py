"""Evaluation metrics, computed per 3D volume (not per mini-batch).

Conventions that differ from the original notebook:
* A structure absent from both prediction and ground truth gives NaN, not 1.0,
  and is skipped when averaging. Scoring empty slices as perfect inflated Dice.
* Dice is computed on whole volumes; the reported number is the mean over
  volumes, not the mean over training batches.
* Clinical endpoints: LV/RV end-diastolic and end-systolic volumes (mL) and
  ejection fraction EF = (EDV - ESV) / EDV, with Bland-Altman agreement.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .labels import INTERNAL, STRUCTURES


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * (p & g).sum() / denom)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> float:
    """95th-percentile symmetric Hausdorff distance in mm (NaN if either is empty)."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() or not g.any():
        return float("nan")
    sp, sg = _surface(p), _surface(g)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def volume_ml(mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    return float(mask.astype(bool).sum() * np.prod(spacing) / 1000.0)


def per_structure(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in STRUCTURES:
        k = INTERNAL[name]
        out[f"dice_{name}"] = dice(pred == k, gt == k)
        out[f"hd95_{name}"] = hd95(pred == k, gt == k, spacing)
        out[f"vol_pred_{name}"] = volume_ml(pred == k, spacing)
        out[f"vol_gt_{name}"] = volume_ml(gt == k, spacing)
    return out


def ejection_fraction(edv: float, esv: float) -> float:
    return float("nan") if edv <= 0 else 100.0 * (edv - esv) / edv


def bland_altman(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, float) - np.asarray(ref, float)
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return {"bias": float("nan"), "loa_low": float("nan"), "loa_high": float("nan"), "mae": float("nan"), "n": 0}
    bias, sd = diff.mean(), diff.std(ddof=1) if diff.size > 1 else 0.0
    return {"bias": float(bias), "loa_low": float(bias - 1.96 * sd), "loa_high": float(bias + 1.96 * sd),
            "mae": float(np.abs(diff).mean()), "n": int(diff.size)}
