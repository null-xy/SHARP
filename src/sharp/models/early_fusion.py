#!/usr/bin/env python3
"""
sharp.models.early_fusion

Early-fusion deep learning baseline for 3-class SSRL event recognition
(paper Table 4, "Early Fusion" row). This is the simplest of the three
neural fusion strategies compared in the paper: it tests whether directly
concatenating the raw modality streams and encoding them with one shared
network is enough, in contrast to Cross-Modal Attention
(sharp.models.cross_modal_attention), which keeps modality-specific
representations and learns directed cross-modal interactions instead.

Architecture:
  EDA(1,60) + EmoNet(2,60) + Gaze(2,60)
  -> concatenate along the channel dim -> (5, 60)
  -> single shared Conv1d encoder
  -> MLP classifier

Ablations: all 7 non-empty subsets of {EDA, EmoNet, Gaze}.

Protocol: Leave-One-Group-Out cross-validation (5 folds, one per retained
group), pooled-OOF macro-F1, 10 random seeds, fixed 100-epoch training
budget with checkpoints saved every 10 epochs.

Outputs: processed_data/analysis/early_fusion/
  results_early.csv   - per-ablation x per-fold F1 + accuracy
  summary_early.csv   - mean +/- std across folds
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

from ..dataset import load_sharp_dataset, get_cv_splits, impute_eda_fold, restore_eda
from ..utils import set_seed, make_ckpt_epochs, CheckpointTracker, eval_fold_dl

from ..config import ANALYSIS_DIR, ROOT
out_dir = ANALYSIS_DIR / "early_fusion"

# Channel count per modality
IN_CH = {"eda": 1, "emonet": 2, "gaze": 2}


# ── Model ─────────────────────────────────────────────────────────────────────

class _LN1d(nn.Module):
    """LayerNorm for (B, C, T) Conv1d outputs: normalises C at each time step."""
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class EarlyFusionEncoder(nn.Module):
    """
    Single Conv1d encoder operating on the concatenated multi-channel input.

    Input : (B, total_ch, 60)   total_ch = sum of active modality channels
    Output: (B, feat_dim)
    """
    def __init__(self, total_ch: int, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(total_ch, 32, kernel_size=5, padding=2),
            _LN1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            _LN1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),                  # (B, 64)
        )
        self.proj = nn.Sequential(
            nn.Linear(64, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class EarlyFusionNet(nn.Module):
    """
    Early fusion: concat raw modality signals along channel dim,
    then encode with a single shared encoder.
    """
    def __init__(
        self,
        active_modalities: list[str],
        feat_dim: int = 64,
        num_classes: int = 3,
    ):
        super().__init__()
        self.active = active_modalities
        total_ch = sum(IN_CH[m] for m in active_modalities)
        self.encoder = EarlyFusionEncoder(total_ch, feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        # Concatenate active modalities along channel dimension
        signals = torch.cat([batch[m] for m in self.active], dim=1)  # (B, C, 60)
        feat = self.encoder(signals)
        return self.classifier(feat)


# ── Training helpers ──────────────────────────────────────────────────────────

def class_weights(y: np.ndarray, n_classes: int = 3, device="cpu") -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def collate(batch: list[dict]) -> dict:
    keys = ["eda", "emonet", "gaze", "label"]
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def run_fold(
    ds,
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    active:    list[str],
    device:    str,
    fixed_epochs: int = 100,
    lr:         float = 3e-4,
    wd:         float = 1e-2,
    batch_size: int = 16,
    ckpt_epochs: list[int] | None = None,
    ckpt_save_dir: Path | None = None,
) -> tuple:
    """Train for a fixed number of epochs on the full training fold.
    No early stopping; cosine annealing provides LR decay.
    val_idx (test fold) is evaluated once after training completes.
    """
    saved_eda = impute_eda_fold(ds, train_idx) if "eda" in active else {}

    y_train = ds.labels[train_idx]
    cw = class_weights(y_train, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    model = EarlyFusionNet(active).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fixed_epochs)

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=batch_size,
                          shuffle=True, collate_fn=collate, drop_last=False)
    test_dl  = DataLoader(Subset(ds, val_idx),   batch_size=len(val_idx),
                          shuffle=False, collate_fn=collate)

    ckpt_set   = set(ckpt_epochs) if ckpt_epochs else set()
    ckpt_preds: dict[int, tuple[list, list]] = {}

    for epoch in range(fixed_epochs):
        model.train()
        for xb in train_dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            opt.zero_grad()
            loss = criterion(model(xb), xb["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (epoch + 1) in ckpt_set:
            ckpt_preds[epoch + 1] = eval_fold_dl(model, test_dl, device)
            if ckpt_save_dir is not None:
                ckpt_save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           ckpt_save_dir / f"ep{epoch+1:03d}.pt")

    # Final evaluation on test fold.
    y_o, preds_list = eval_fold_dl(model, test_dl, device)
    preds = np.array(preds_list)
    y_o_arr = np.array(y_o)
    test_f1  = f1_score(y_o_arr, preds, average="macro", zero_division=0)
    test_acc = (preds == y_o_arr).mean()

    restore_eda(ds, saved_eda)
    return test_f1, test_acc, y_o, preds_list, ckpt_preds


# ── Ablation config ───────────────────────────────────────────────────────────

ALL_MODALITIES = ["eda", "emonet", "gaze"]
ABLATIONS: list[tuple[str, ...]] = [
    combo
    for r in range(1, 4)
    for combo in itertools.combinations(ALL_MODALITIES, r)
]
ABLATION_LABELS = {
    ("eda",):                  "EDA",
    ("emonet",):               "EmoNet",
    ("gaze",):                 "Gaze",
    ("eda", "emonet"):         "EDA+EmoNet",
    ("eda", "gaze"):           "EDA+Gaze",
    ("emonet", "gaze"):        "EmoNet+Gaze",
    ("eda", "emonet", "gaze"): "All3",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(season: str | None = None, n_seeds: int = 10) -> None:
    out_dir = ANALYSIS_DIR / ("early_fusion" + (f"_{season}" if season else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Seeds: {n_seeds}\n")

    ds = load_sharp_dataset(individual=False, season=season, require_eda=True)

    result_rows: list[dict] = []
    seed_rows:   list[dict] = []
    epoch_rows:  list[dict] = []

    for combo in ABLATIONS:
        tag    = ABLATION_LABELS[combo]
        active = list(combo)
        seed_f1s: list[float] = []
        tracker = CheckpointTracker(make_ckpt_epochs(100))

        for seed in range(n_seeds):
            splits = get_cv_splits(ds, random_state=seed)
            seed_y_true: list[int] = []
            seed_y_pred: list[int] = []
            tracker.reset_seed()
            for fold, (tr_idx, va_idx) in enumerate(splits):
                set_seed(seed * 1000 + fold)
                ckpt_dir = out_dir / "checkpoints" / "_".join(active) / f"seed{seed:02d}_fold{fold}"
                f1, acc, y_true, y_pred, ckpt_preds = run_fold(
                    ds, tr_idx, va_idx, active, device,
                    ckpt_epochs=tracker.ckpt_epochs,
                    ckpt_save_dir=ckpt_dir,
                )
                seed_y_true.extend(y_true)
                seed_y_pred.extend(y_pred)
                result_rows.append({
                    "ablation": tag, "seed": seed, "fold": fold,
                    "val_f1":  round(f1, 4),
                    "val_acc": round(acc, 4),
                })
                tracker.add_fold(ckpt_preds)
            seed_f1 = float(f1_score(seed_y_true, seed_y_pred, average="macro", zero_division=0))
            seed_f1s.append(seed_f1)
            seed_rows.append({"ablation": tag, "seed": seed, "seed_f1": round(seed_f1, 4)})
            tracker.commit_seed()

        means, stds, best_ep = tracker.summary(label=tag)
        mean_f1 = means[best_ep]
        std_f1  = stds.get(best_ep, 0.0)
        print(f"{tag:20s}  F1={mean_f1:.3f}±{std_f1:.3f}  best@ep{best_ep:03d}")
        for ep, m in means.items():
            epoch_rows.append({"ablation": tag, "epoch": ep, "f1_mean": round(m, 4),
                                "f1_std": round(stds.get(ep, 0.0), 4)})

    # ── Save results ──────────────────────────────────────────────────────────
    df_res = pd.DataFrame(result_rows)
    df_res.to_csv(out_dir / "results_early.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_dir / "results_early_per_seed.csv", index=False)

    summary = (df_res.groupby("ablation")[["val_f1", "val_acc"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = ["f1_mean", "f1_std", "acc_mean", "acc_std"]
    summary = summary.sort_values("f1_mean", ascending=False)
    summary.to_csv(out_dir / "summary_early.csv")
    pd.DataFrame(epoch_rows).to_csv(out_dir / "results_early_by_epoch.csv", index=False)

    print(f"\n=== Summary (sorted by F1) ===")
    print(summary.to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["autumn", "spring"], default=None)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    main(season=args.season, n_seeds=args.seeds)
