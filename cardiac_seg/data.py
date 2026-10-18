"""Indexing and loading of the M&Ms-2 dataset.

The M&Ms-2 release ships one folder per subject (``dataset/001`` ...
``dataset/360``) containing, for each view (SA = short axis, LA = long axis):

    {id}_{view}_CINE.nii.gz     full cardiac cycle (4D for SA), NO labels
    {id}_{view}_ED.nii.gz       end-diastolic frame
    {id}_{view}_ED_gt.nii.gz    manual labels for the ED frame
    {id}_{view}_ES.nii.gz       end-systolic frame
    {id}_{view}_ES_gt.nii.gz    manual labels for the ES frame

Only the ED and ES frames have ground truth. The original notebook paired the
CINE volumes with ED labels (and fell back to "any label of this subject"),
which mixed misaligned image/mask pairs into training. This module pairs
images and labels by exact (subject, view, phase) key and nothing else.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

LABELLED_PHASES = ("ED", "ES")
VIEWS = ("SA", "LA")
_IMAGE_RE = re.compile(r"^(?P<sid>\d{3})_(?P<view>SA|LA)_(?P<phase>ED|ES|CINE)\.nii(\.gz)?$")
_LABEL_RE = re.compile(r"^(?P<sid>\d{3})_(?P<view>SA|LA)_(?P<phase>ED|ES)_gt\.nii(\.gz)?$")

# Disease codes used in dataset_information.csv of the M&Ms-2 release.
DISEASE_NAMES = {
    "NOR": "Normal",
    "LV": "Dilated left ventricle",
    "HCM": "Hypertrophic cardiomyopathy",
    "ARR": "Congenital arrhythmogenesis",
    "FALL": "Tetralogy of Fallot",
    "CIA": "Interatrial communication",
    "TRI": "Tricuspid regurgitation",
    "RV": "Dilated right ventricle",
}


@dataclass(frozen=True)
class Pair:
    subject: int
    view: str
    phase: str
    image: Path
    label: Path


def index_dataset(dataset_root: str | Path, views: Iterable[str] = ("SA",)) -> pd.DataFrame:
    """Return one row per labelled (subject, view, phase) with exact file pairs.

    CINE volumes are never returned: they have no ground truth.
    Raises if an ED/ES image has no matching ``_gt`` file, rather than silently
    substituting another label.
    """
    root = Path(dataset_root)
    views = tuple(views)
    rows: list[dict] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        images, labels = {}, {}
        for f in subject_dir.iterdir():
            m = _LABEL_RE.match(f.name)
            if m:
                labels[(int(m["sid"]), m["view"], m["phase"])] = f
                continue
            m = _IMAGE_RE.match(f.name)
            if m and m["phase"] in LABELLED_PHASES:
                images[(int(m["sid"]), m["view"], m["phase"])] = f
        for key, img in sorted(images.items()):
            sid, view, phase = key
            if view not in views:
                continue
            if key not in labels:
                raise FileNotFoundError(f"No ground truth for {img.name} (expected {img.name.replace('.nii', '_gt.nii')})")
            if sid != int(subject_dir.name):
                raise ValueError(f"{img} sits in folder {subject_dir.name} but is named for subject {sid}")
            rows.append(dict(subject=sid, view=view, phase=phase, image=str(img), label=str(labels[key])))
    if not rows:
        raise FileNotFoundError(f"No labelled ED/ES volumes found under {root}")
    return pd.DataFrame(rows)


def read_metadata(info_csv: str | Path) -> pd.DataFrame:
    """Read dataset_information.csv; one row per subject with disease and vendor."""
    df = pd.read_csv(info_csv)
    df.columns = [c.strip().upper() for c in df.columns]
    if "SUBJECT_CODE" not in df.columns:
        raise KeyError(f"SUBJECT_CODE column missing from {info_csv}; columns: {list(df.columns)}")
    df = df.dropna(subset=["SUBJECT_CODE"]).drop_duplicates("SUBJECT_CODE")
    df["subject"] = df["SUBJECT_CODE"].astype(int)
    keep = ["subject"] + [c for c in ("DISEASE", "VENDOR", "SCANNER", "FIELD") if c in df.columns]
    out = df[keep].rename(columns=str.lower)
    return out.reset_index(drop=True)


def load_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a 3D NIfTI volume as (H, W, S) float32 plus voxel spacing in mm."""
    img = nib.load(str(path))
    vol = np.asanyarray(img.dataobj)
    if vol.ndim == 2:
        vol = vol[..., None]
    if vol.ndim != 3:
        raise ValueError(f"{path} has shape {vol.shape}; expected a single 3D frame (ED/ES), not CINE")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if len(zooms) == 2:
        zooms = (*zooms, 1.0)
    return vol.astype(np.float32), zooms  # type: ignore[return-value]


def normalise(vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile clip + z-score per volume (robust to MR intensity outliers)."""
    lo, hi = np.percentile(vol, [lo_pct, hi_pct])
    vol = np.clip(vol, lo, hi)
    return ((vol - vol.mean()) / (vol.std() + 1e-8)).astype(np.float32)
