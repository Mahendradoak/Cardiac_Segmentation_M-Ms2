"""Model construction, training loop and volume-level inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import Preprocessor, SliceDataset
from .labels import STRUCTURES
from .metrics import per_structure

ARCHS = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus, "fpn": smp.FPN}


def build_model(arch: str, encoder: str, pretrained: bool = True) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {sorted(ARCHS)}")
    return ARCHS[arch](encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                       in_channels=1, classes=1 + len(STRUCTURES))


class DiceCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.dice(logits, target) + self.ce(logits, target)


@torch.no_grad()
def predict_volume(model: nn.Module, image: np.ndarray, device: torch.device, batch: int = 16) -> np.ndarray:
    """Slice-wise prediction of an (H, W, S) volume, returns (H, W, S) labels."""
    model.eval()
    x = torch.from_numpy(np.moveaxis(image, -1, 0)[:, None].copy()).float()
    out = []
    for i in range(0, x.shape[0], batch):
        out.append(model(x[i:i + batch].to(device)).argmax(1).cpu().numpy())
    return np.moveaxis(np.concatenate(out), 0, -1).astype(np.uint8)


def evaluate_cases(model: nn.Module, cases: list[dict], pre: Preprocessor, device: torch.device) -> pd.DataFrame:
    """Per-volume metrics on the ORIGINAL voxel grid and spacing."""
    rows = []
    for c in cases:
        pred = pre.inverse(predict_volume(model, c["image"], device), c["grid"])
        m = per_structure(pred, c["gt_orig"], c["spacing"])
        rows.append(dict(subject=c["subject"], view=c["view"], phase=c["phase"], **m))
    return pd.DataFrame(rows)


def mean_dice(df: pd.DataFrame) -> float:
    return float(np.nanmean(df[[f"dice_{s}" for s in STRUCTURES]].to_numpy()))


def train(model: nn.Module, train_cases: list[dict], val_cases: list[dict], pre: Preprocessor, out_dir: Path,
          epochs: int = 100, batch_size: int = 16, lr: float = 3e-4, weight_decay: float = 1e-4,
          patience: int = 20, num_workers: int = 2, device: str | None = None, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(SliceDataset(train_cases, train=True), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True, generator=g, pin_memory=dev.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = DiceCE()
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, bad, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item() * x.shape[0], n + x.shape[0]
        sched.step()
        val = evaluate_cases(model, val_cases, pre, dev)
        vd = mean_dice(val)
        history.append(dict(epoch=epoch, train_loss=tot / max(n, 1), val_mean_dice=vd,
                            **{f"val_dice_{s}": float(val[f'dice_{s}'].mean()) for s in STRUCTURES},
                            seconds=time.time() - t0))
        print(json.dumps(history[-1]))
        if vd > best:
            best, bad = vd, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch}")
                break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return {"best_val_mean_dice": best, "epochs_run": len(history)}
