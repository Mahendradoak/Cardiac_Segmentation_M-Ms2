"""Label conventions and an anatomical sanity check.

Internally every mask is remapped to:  0 background, 1 LV cavity, 2 LV myocardium, 3 RV cavity.

Raw label values differ between datasets (ACDC uses RV=1/LV=3; M&Ms-1 and M&Ms-2
use LV=1/RV=3). Rather than trust a comment, ``check_label_order`` verifies the
mapping from anatomy: the LV cavity is enclosed by myocardium, the RV is not.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

STRUCTURES = ("LV", "MYO", "RV")
INTERNAL = {"LV": 1, "MYO": 2, "RV": 3}
# Raw value -> structure for the M&Ms-2 release.
MNMS2_RAW = {1: "LV", 2: "MYO", 3: "RV"}


def remap(mask: np.ndarray, raw_to_structure: dict[int, str] = MNMS2_RAW) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.uint8)
    m = np.rint(mask).astype(np.int64)
    unknown = set(np.unique(m)) - set(raw_to_structure) - {0}
    if unknown:
        raise ValueError(f"Unexpected label values {sorted(unknown)}")
    for raw, name in raw_to_structure.items():
        out[m == raw] = INTERNAL[name]
    return out


def myocardial_enclosure(mask: np.ndarray, cavity: int, myo: int = 2) -> float:
    """Fraction of a cavity's outer boundary that touches myocardium (per-slice 2D)."""
    touching, total = 0, 0
    for z in range(mask.shape[-1]):
        sl = mask[..., z]
        cav = sl == cavity
        if not cav.any():
            continue
        ring = ndimage.binary_dilation(cav) & ~cav
        total += int(ring.sum())
        touching += int((ring & (sl == myo)).sum())
    return touching / total if total else float("nan")


def check_label_order(raw_masks: list[np.ndarray], raw_to_structure: dict[int, str] = MNMS2_RAW) -> dict:
    """Confirm the label marked 'LV' is the cavity enclosed by myocardium.

    Returns the mean enclosure for each cavity label and whether the mapping is
    consistent. Use on a handful of SA ED volumes before training.
    """
    inv = {v: k for k, v in raw_to_structure.items()}
    lv_raw, rv_raw, myo_raw = inv["LV"], inv["RV"], inv["MYO"]
    lv = np.nanmean([myocardial_enclosure(np.rint(m), lv_raw, myo_raw) for m in raw_masks])
    rv = np.nanmean([myocardial_enclosure(np.rint(m), rv_raw, myo_raw) for m in raw_masks])
    return {"lv_enclosure": float(lv), "rv_enclosure": float(rv), "consistent": bool(lv > rv)}
