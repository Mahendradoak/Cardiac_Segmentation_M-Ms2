"""Subject-level splits. Every split is by subject, so no subject's ED and ES
frames (or SA and LA views) can land in different partitions."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# Subject ranges of the M&Ms-2 challenge release (train / validation / test).
# Check them against the dataset_information.csv you downloaded.
OFFICIAL_RANGES = {"train": (1, 160), "val": (161, 200), "test": (201, 360)}


def official_split(meta: pd.DataFrame) -> dict[str, list[int]]:
    out = {}
    for part, (lo, hi) in OFFICIAL_RANGES.items():
        out[part] = sorted(meta.loc[meta.subject.between(lo, hi), "subject"].tolist())
    return out


def vendor_holdout_split(meta: pd.DataFrame, test_vendor: str, val_frac: float = 0.15, seed: int = 42) -> dict[str, list[int]]:
    """Train/val on all vendors except ``test_vendor``; test only on ``test_vendor``.

    This measures generalisation to an unseen scanner vendor, which is what
    M&Ms-2 was designed to test.
    """
    if "vendor" not in meta.columns:
        raise KeyError("metadata has no VENDOR column")
    vendors = meta.vendor.astype(str).str.strip()
    if test_vendor not in set(vendors):
        raise ValueError(f"vendor {test_vendor!r} not in {sorted(set(vendors))}")
    test = meta.loc[vendors == test_vendor, "subject"]
    rest = meta.loc[vendors != test_vendor]
    strat = rest["disease"] if "disease" in rest else None
    tr, va = train_test_split(rest.subject, test_size=val_frac, random_state=seed, stratify=strat)
    return {"train": sorted(tr.tolist()), "val": sorted(va.tolist()), "test": sorted(test.tolist())}


def stratified_split(meta: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> dict[str, list[int]]:
    """Random split stratified on disease x vendor (falls back to disease)."""
    key = meta["disease"].astype(str)
    if "vendor" in meta.columns:
        joint = key + "|" + meta["vendor"].astype(str)
        if joint.value_counts().min() >= 3:
            key = joint
    tr_va, te = train_test_split(meta.subject, test_size=test_frac, random_state=seed, stratify=key)
    key_tr = key.loc[tr_va.index]
    tr, va = train_test_split(tr_va, test_size=val_frac / (1 - test_frac), random_state=seed, stratify=key_tr)
    return {"train": sorted(tr.tolist()), "val": sorted(va.tolist()), "test": sorted(te.tolist())}


def assert_disjoint(split: dict[str, list[int]]) -> None:
    parts = list(split.values())
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            overlap = set(parts[i]) & set(parts[j])
            if overlap:
                raise AssertionError(f"subject leakage between splits: {sorted(overlap)[:10]}")


def make_split(meta: pd.DataFrame, mode: str, **kw) -> dict[str, list[int]]:
    if mode == "official":
        split = official_split(meta)
    elif mode == "vendor":
        split = vendor_holdout_split(meta, **kw)
    elif mode == "stratified":
        split = stratified_split(meta, **kw)
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    assert_disjoint(split)
    for part, ids in split.items():
        if not ids:
            raise ValueError(f"split {part!r} is empty")
    return split
