"""Synthetic M&Ms-2-shaped dataset: same folder layout, file names and label values."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

VENDORS = ["SIEMENS", "GE", "PHILIPS"]
DISEASES = ["NOR", "LV", "HCM"]


def synthetic_heart(h=96, w=88, s=6, lv_r=14, myo_t=5, shift=0):
    """LV cavity (1) inside a myocardial ring (2), RV cavity (3) beside the septum."""
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2 + 8 + shift
    r = np.hypot(yy - cy, xx - cx)
    sl = np.zeros((h, w), np.uint8)
    sl[r < lv_r + myo_t] = 2
    sl[r < lv_r] = 1
    rv = (np.hypot(yy - cy, xx - (cx - lv_r - myo_t - 10)) < 14) & (sl == 0)
    sl[rv] = 3
    lab = np.repeat(sl[..., None], s, axis=2)
    lab[..., 0] = 0  # an apical/basal slice with no heart
    img = np.where(lab == 1, 200, np.where(lab == 2, 80, np.where(lab == 3, 180, 30))).astype(np.float32)
    img += np.random.default_rng(0).normal(0, 5, img.shape).astype(np.float32)
    return img, lab


def _save(arr, path, spacing=(1.4, 1.4, 9.0)):
    aff = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(arr, aff), str(path))


@pytest.fixture(scope="session")
def mnms2_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("MnM2")
    ds = root / "dataset"
    rows = []
    for sid in range(1, 13):
        d = ds / f"{sid:03d}"
        d.mkdir(parents=True)
        for view in ("SA", "LA"):
            ed_img, ed_lab = synthetic_heart(lv_r=16)
            es_img, es_lab = synthetic_heart(lv_r=10, shift=2)  # ES differs from ED
            if view == "LA":
                ed_img, ed_lab, es_img, es_lab = (a[..., :1] for a in (ed_img, ed_lab, es_img, es_lab))
            _save(ed_img, d / f"{sid:03d}_{view}_ED.nii.gz")
            _save(ed_lab.astype(np.float32), d / f"{sid:03d}_{view}_ED_gt.nii.gz")
            _save(es_img, d / f"{sid:03d}_{view}_ES.nii.gz")
            _save(es_lab.astype(np.float32), d / f"{sid:03d}_{view}_ES_gt.nii.gz")
            cine = np.stack([ed_img, es_img, ed_img], axis=-1)  # 4D, no labels
            _save(cine, d / f"{sid:03d}_{view}_CINE.nii.gz")
        rows.append(dict(SUBJECT_CODE=sid, DISEASE=DISEASES[sid % 3], VENDOR=VENDORS[(sid // 3) % 3]))
    pd.DataFrame(rows).to_csv(root / "dataset_information.csv", index=False)
    return root
