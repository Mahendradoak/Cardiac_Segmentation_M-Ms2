import json

import numpy as np
import pytest

from cardiac_seg.cli import main
from cardiac_seg.data import index_dataset, load_volume, read_metadata
from cardiac_seg.dataset import Preprocessor, SliceDataset, load_case
from cardiac_seg.labels import check_label_order, remap
from cardiac_seg.metrics import bland_altman, dice, ejection_fraction, hd95
from cardiac_seg.splits import assert_disjoint, make_split

from .conftest import synthetic_heart


def test_index_pairs_exactly_and_skips_cine(mnms2_root):
    idx = index_dataset(mnms2_root / "dataset", views=("SA", "LA"))
    assert len(idx) == 12 * 2 * 2                       # subjects x views x {ED, ES}
    assert not idx.image.str.contains("CINE").any()
    for r in idx.itertuples():
        assert r.label == r.image.replace(".nii.gz", "_gt.nii.gz")


def test_index_refuses_missing_label(tmp_path, mnms2_root):
    import shutil
    d = tmp_path / "001"
    shutil.copytree(mnms2_root / "dataset" / "001", d)
    (d / "001_SA_ES_gt.nii.gz").unlink()
    with pytest.raises(FileNotFoundError, match="001_SA_ES"):
        index_dataset(tmp_path)


def test_label_order_check_detects_swap():
    _, lab = synthetic_heart()
    assert check_label_order([lab])["consistent"]
    swapped = lab.copy()
    swapped[lab == 1], swapped[lab == 3] = 3, 1
    assert not check_label_order([swapped])["consistent"]


def test_remap_rejects_unknown_values():
    with pytest.raises(ValueError):
        remap(np.array([[0, 1, 4]]))


def test_dice_nan_when_absent_everywhere():
    z = np.zeros((4, 4), bool)
    assert np.isnan(dice(z, z))
    a = np.zeros((4, 4), bool)
    a[1:3, 1:3] = True
    assert dice(a, a) == 1.0
    assert dice(a, z) == 0.0


def test_hd95_and_ef():
    a = np.zeros((20, 20, 3), bool)
    a[5:10, 5:10] = True
    assert hd95(a, a, (1, 1, 1)) == 0.0
    b = np.roll(a, 2, axis=0)
    assert hd95(a, b, (1, 1, 1)) == pytest.approx(2.0)
    assert hd95(a, b, (0.5, 1, 1)) == pytest.approx(1.0)
    assert ejection_fraction(150, 60) == pytest.approx(60.0)
    ba = bland_altman(np.array([1.0, 2, 3]), np.array([1.0, 2, 4]))
    assert ba["bias"] == pytest.approx(-1 / 3)


def test_preprocessor_roundtrip(mnms2_root):
    idx = index_dataset(mnms2_root / "dataset")
    lab, spacing = load_volume(idx.label.iloc[0])
    pre = Preprocessor(spacing_mm=1.25, size=128)
    fwd, grid = pre.forward(remap(lab), spacing, order=0)
    assert fwd.shape[:2] == (128, 128)
    back = pre.inverse(fwd, grid)
    assert back.shape == lab.shape
    assert dice(back == 1, remap(lab) == 1) > 0.95


def test_splits_disjoint_and_vendor_holdout(mnms2_root):
    meta = read_metadata(mnms2_root / "dataset_information.csv")
    s = make_split(meta, "stratified", val_frac=0.25, test_frac=0.25)
    assert_disjoint(s)
    v = make_split(meta, "vendor", test_vendor="GE", val_frac=0.4)
    test_vendors = set(meta.set_index("subject").loc[v["test"], "vendor"])
    train_vendors = set(meta.set_index("subject").loc[v["train"] + v["val"], "vendor"])
    assert test_vendors == {"GE"} and "GE" not in train_vendors


def test_slice_dataset_shapes(mnms2_root):
    idx = index_dataset(mnms2_root / "dataset")
    pre = Preprocessor(size=96)
    cases = [load_case(r, pre) for r in idx.head(2).itertuples()]
    ds = SliceDataset(cases, train=True)
    x, y = ds[1]
    assert x.shape == (1, 96, 96) and y.shape == (96, 96)
    assert set(np.unique(y.numpy())) <= {0, 1, 2, 3}


def test_end_to_end_cli(mnms2_root, tmp_path):
    run = tmp_path / "run"
    main(["check-labels", "--data", str(mnms2_root), "--n", "3"])
    main(["train", "--data", str(mnms2_root), "--out", str(run), "--split", "stratified",
          "--val-frac", "0.25", "--test-frac", "0.25", "--arch", "unet", "--encoder", "resnet18",
          "--no-pretrained", "--size", "96", "--epochs", "2", "--batch-size", "4", "--workers", "0"])
    main(["evaluate", "--data", str(mnms2_root), "--run", str(run), "--part", "test"])
    split = json.loads((run / "split.json").read_text())
    assert_disjoint(split)
    for f in ("best.pt", "history.csv", "test_per_volume.csv", "test_summary.csv", "test_clinical.csv", "test_agreement.json"):
        assert (run / f).exists(), f
