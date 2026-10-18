"""Command-line entry point.

    python -m cardiac_seg.cli check-labels --data /path/MnM2
    python -m cardiac_seg.cli train        --data /path/MnM2 --split official --arch unet --encoder resnet34 --out runs/unet_r34
    python -m cardiac_seg.cli evaluate     --data /path/MnM2 --run runs/unet_r34
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import DISEASE_NAMES, index_dataset, load_volume, read_metadata
from .dataset import Preprocessor, load_case
from .engine import build_model, evaluate_cases, mean_dice, train
from .labels import STRUCTURES, check_label_order
from .metrics import bland_altman, ejection_fraction
from .splits import make_split


def _paths(root: str) -> tuple[Path, Path]:
    root_p = Path(root)
    ds = root_p / "dataset" if (root_p / "dataset").is_dir() else root_p
    info = root_p / "dataset_information.csv"
    if not info.exists():
        raise FileNotFoundError(f"{info} not found; point --data at the extracted MnM2 folder")
    return ds, info


def _load(index: pd.DataFrame, subjects: list[int], pre: Preprocessor) -> list[dict]:
    rows = index[index.subject.isin(subjects)]
    return [load_case(r, pre) for r in rows.itertuples(index=False)]


def cmd_check_labels(a) -> None:
    ds, _ = _paths(a.data)
    idx = index_dataset(ds, views=("SA",))
    sample = idx[idx.phase == "ED"].head(a.n)
    masks = [load_volume(p)[0] for p in sample.label]
    res = check_label_order(masks)
    print(json.dumps(res, indent=2))
    if not res["consistent"]:
        raise SystemExit("Label order looks wrong: the label mapped to LV is not the myocardium-enclosed cavity. "
                         "Edit MNMS2_RAW in cardiac_seg/labels.py.")


def cmd_train(a) -> None:
    ds, info = _paths(a.data)
    meta = read_metadata(info)
    if a.diseases:
        meta = meta[meta.disease.isin(a.diseases)]
    if a.split == "vendor":
        kw = {"test_vendor": a.test_vendor, "val_frac": a.val_frac, "seed": a.seed}
    elif a.split == "stratified":
        kw = {"val_frac": a.val_frac, "test_frac": a.test_frac, "seed": a.seed}
    else:
        kw = {}
    split = make_split(meta, a.split, **kw)
    idx = index_dataset(ds, views=tuple(a.views))
    pre = Preprocessor(spacing_mm=a.spacing, size=a.size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    (out / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    train_cases, val_cases = _load(idx, split["train"], pre), _load(idx, split["val"], pre)
    print(f"train volumes={len(train_cases)} val volumes={len(val_cases)} test subjects={len(split['test'])}")
    model = build_model(a.arch, a.encoder, pretrained=not a.no_pretrained)
    res = train(model, train_cases, val_cases, pre, out, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                patience=a.patience, num_workers=a.workers, seed=a.seed)
    (out / "train_summary.json").write_text(json.dumps(res, indent=1))
    print(res)


def clinical_table(per_vol: pd.DataFrame) -> pd.DataFrame:
    """LV/RV EDV, ESV and EF from SA volumes, predicted vs manual, one row per subject."""
    sa = per_vol[per_vol.view == "SA"]
    rows = []
    for sid, g in sa.groupby("subject"):
        if set(g.phase) != {"ED", "ES"}:
            continue
        ed, es = g[g.phase == "ED"].iloc[0], g[g.phase == "ES"].iloc[0]
        r = {"subject": sid}
        for s in ("LV", "RV"):
            for src in ("pred", "gt"):
                edv, esv = ed[f"vol_{src}_{s}"], es[f"vol_{src}_{s}"]
                r[f"{s}_EDV_{src}"], r[f"{s}_ESV_{src}"] = edv, esv
                r[f"{s}_EF_{src}"] = ejection_fraction(edv, esv)
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_evaluate(a) -> None:
    ds, info = _paths(a.data)
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    split = json.loads((run / "split.json").read_text())
    meta = read_metadata(info)
    idx = index_dataset(ds, views=tuple(cfg["views"]))
    pre = Preprocessor(spacing_mm=cfg["spacing"], size=cfg["size"])
    model = build_model(cfg["arch"], cfg["encoder"], pretrained=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    model.to(dev)
    cases = _load(idx, split[a.part], pre)
    per_vol = evaluate_cases(model, cases, pre, dev).merge(meta, on="subject", how="left")
    per_vol.to_csv(run / f"{a.part}_per_volume.csv", index=False)

    cols = [f"dice_{s}" for s in STRUCTURES] + [f"hd95_{s}" for s in STRUCTURES]
    summary = per_vol[cols].agg(["mean", "std", "median"]).T
    summary.to_csv(run / f"{a.part}_summary.csv")
    for key in ("vendor", "disease"):
        if key in per_vol:
            per_vol.groupby(key)[cols].mean().to_csv(run / f"{a.part}_by_{key}.csv")

    clin = clinical_table(per_vol)
    clin.to_csv(run / f"{a.part}_clinical.csv", index=False)
    agreement = {f"{s}_{m}": bland_altman(clin[f"{s}_{m}_pred"], clin[f"{s}_{m}_gt"])
                 for s in ("LV", "RV") for m in ("EDV", "ESV", "EF")} if len(clin) else {}
    (run / f"{a.part}_agreement.json").write_text(json.dumps(agreement, indent=1))
    print(summary.round(3).to_string())
    print(f"mean Dice over structures: {mean_dice(per_vol):.4f}  ({len(per_vol)} volumes)")
    for k, v in agreement.items():
        print(f"{k:10s} bias={v['bias']:+.2f} LoA=[{v['loa_low']:.2f}, {v['loa_high']:.2f}] MAE={v['mae']:.2f} n={v['n']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cardiac_seg")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check-labels", help="verify the LV/RV label order from anatomy")
    c.add_argument("--data", required=True)
    c.add_argument("--n", type=int, default=20)
    c.set_defaults(func=cmd_check_labels)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--split", choices=["official", "vendor", "stratified"], default="official")
    t.add_argument("--test-vendor", help="vendor held out for testing when --split vendor")
    t.add_argument("--val-frac", type=float, default=0.15)
    t.add_argument("--test-frac", type=float, default=0.15)
    t.add_argument("--diseases", nargs="*", choices=sorted(DISEASE_NAMES), help="restrict subjects (default: all)")
    t.add_argument("--views", nargs="+", choices=["SA", "LA"], default=["SA"])
    t.add_argument("--arch", choices=["unet", "unetplusplus", "deeplabv3plus", "fpn"], default="unet")
    t.add_argument("--encoder", default="resnet34")
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--spacing", type=float, default=1.25)
    t.add_argument("--size", type=int, default=256)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--patience", type=int, default=20)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--run", required=True)
    e.add_argument("--part", choices=["val", "test"], default="test")
    e.set_defaults(func=cmd_evaluate)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
