# Cardiac MRI segmentation on M&Ms-2

Segments the **left-ventricular (LV) cavity, LV myocardium and right-ventricular (RV) cavity** in short-axis cine MRI from the [M&Ms-2 challenge](https://www.ub.edu/mnms-2/). It then turns the masks into the numbers a cardiologist uses: **end-diastolic and end-systolic volumes (EDV, ESV) and ejection fraction (EF) for both ventricles**, compared with the manual contours by Bland–Altman analysis.

## What the pipeline does

1. **Indexes the dataset strictly.** Each ED/ES image is paired only with the ground-truth file of the same subject, view and phase (`001_SA_ED.nii.gz` ↔ `001_SA_ED_gt.nii.gz`). CINE sequences have no labels and are never used for supervision. A missing label is an error, not a silent substitution.
2. **Checks the label convention from anatomy.** `check-labels` verifies that the label mapped to "LV" is the cavity enclosed by myocardium. This guards against the RV/LV order differing between ACDC and M&Ms.
3. **Splits by subject.** The modes are:
   - `official`: the challenge ranges 1–160 / 161–200 / 201–360.
   - `vendor`: train on two vendors, test on the unseen third. This is the domain-shift question M&Ms-2 was built for.
   - `stratified`: disease × vendor.
   Every split is checked for subject leakage and saved to `split.json`.
4. **Preprocesses without distortion.** Each volume gets a percentile clip and a z-score, is resampled in-plane to 1.25 mm, and is centre padded or cropped to 256×256. The transform is invertible, so all metrics are computed on the **original voxel grid and spacing**.
5. **Trains a 2D network on every short-axis slice.** Architectures are U-Net, U-Net++, DeepLabV3+ or FPN, with any `timm`/`smp` encoder. Training details:
   - Loss: Dice + cross-entropy.
   - Augmentation: rotation, scaling, gamma and noise. There are no left/right flips, because a flip moves the RV to the wrong side of the heart.
   - Optimiser: AdamW with a cosine schedule and mixed precision on GPU.
   - Model selection: by **volume-level** validation Dice.
6. **Evaluates on the held-out test subjects.** It reports, per structure:
   - Dice and HD95 in mm.
   - Breakdowns by vendor and by disease.
   - LV/RV EDV, ESV and EF, with bias, limits of agreement and MAE against the manual contours.

## Quick start

```bash
pip install -r requirements.txt          # or a CUDA build of torch first
# Download and extract M&Ms-2 so you have MnM2/dataset/001 ... and MnM2/dataset_information.csv

python -m cardiac_seg.cli check-labels --data /path/to/MnM2
python -m cardiac_seg.cli train --data /path/to/MnM2 --split official \
       --arch unet --encoder resnet34 --epochs 100 --out runs/unet_r34
python -m cardiac_seg.cli evaluate --data /path/to/MnM2 --run runs/unet_r34 --part test

# Vendor generalisation experiment (repeat for each vendor name in dataset_information.csv)
python -m cardiac_seg.cli train --data /path/to/MnM2 --split vendor --test-vendor <VENDOR> --out runs/holdout_<VENDOR>
```

Outputs in `runs/<name>/`:

| File | Contents |
|---|---|
| `config.json`, `split.json` | every argument and the exact subject IDs per split |
| `best.pt`, `history.csv` | best checkpoint, per-epoch loss and per-structure validation Dice |
| `test_per_volume.csv` | Dice, HD95 and volumes for every test volume, with vendor and disease |
| `test_summary.csv`, `test_by_vendor.csv`, `test_by_disease.csv` | aggregated metrics |
| `test_clinical.csv`, `test_agreement.json` | predicted vs manual LV/RV EDV, ESV and EF; Bland–Altman statistics |

## Tests

```bash
pytest -q        # 10 tests, ~15 s on CPU, no real data needed
ruff check .
```

The tests build a synthetic dataset with the M&Ms-2 folder layout and label values. They check:
- exact image–label pairing and that CINE is excluded;
- that the label-order check catches a swapped LV/RV;
- metric edge cases (empty structures, HD95 under anisotropic spacing, EF);
- that preprocessing round-trips to the original grid;
- subject-disjoint splits and the vendor hold-out;
- a full `train` → `evaluate` run through the CLI.

## What changed from the original notebook, and why the old numbers are void

The original notebook is kept in `archive/original_notebook.ipynb`. Its reported validation Dice (0.871 / 0.867, `archive/results_prefix_INVALID.csv`) should not be cited, for these reasons:

| Problem in the original | Effect | Fix |
|---|---|---|
| CINE images paired with ED labels; fallback glob `subj*_*_gt` took *any* label of the subject (816 images "paired" with only 544 labels) | About a third of training pairs had masks from a different cardiac phase or view | Exact `(subject, view, phase)` pairing; CINE excluded |
| Middle time frame of the 4D CINE used as the "image" | Image and mask from different time points | Only ED/ES 3D frames are loaded |
| Only the middle slice of each volume used | Most of each heart never seen; no 3D metrics possible | Every slice used; metrics per 3D volume |
| Structure absent in both prediction and truth scored as Dice = 1.0; Dice averaged over mini-batches | Inflated, batch-size-dependent Dice | NaN for empty structures; Dice per volume |
| Test split created but never evaluated | Model selected and reported on the same validation set | Separate test evaluation |
| Label comment assumed RV=1, LV=3 (the ACDC order) | Per-structure results would be mislabelled | M&Ms-2 order (LV=1, MYO=2, RV=3) plus an anatomical check |
| SA and LA views mixed; `cv2.resize` to 256×256 | Aspect ratio and physical scale distorted | SA by default; isotropic in-plane resampling |
| Random split; only NOR/LV/HCM subjects | Ignored the dataset's vendor shift and its RV pathologies | All diseases by default; vendor hold-out split available |
| Disease codes printed as "Fall related" and "CIA related" | Wrong clinical meaning | `FALL` = tetralogy of Fallot, `CIA` = interatrial communication, `ARR` = congenital arrhythmogenesis, `TRI` = tricuspid regurgitation |

## Status

The code is tested end-to-end on synthetic data. **Results on the real M&Ms-2 data still need to be produced** by running the commands above on a GPU; roughly 1–2 h per model on a T4.

Licence: none yet. Add one (MIT is common for research code) before citing the repository in an application.
