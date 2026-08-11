#!/usr/bin/env python3
"""
sharp.evaluation.confusion_matrix

Generates the pooled-OOF confusion matrix for Cross-Modal Attention with
all three modalities (paper Figure 4), by training on all n=29 events
with 10 random seeds under Leave-One-Group-Out cross-validation and
pooling every fold's held-out predictions together.

Protocol: n=29, require_eda=True, season=autumn, LOGO, fixed 100-epoch
training budget with no checkpoint selection (same protocol as
sharp.evaluation.epoch100_metrics), 10 seeds.

Outputs: processed_data/analysis/confusion_matrix/
  confusion_matrix.png  - normalized confusion matrix (paper Figure 4)
  confusion_raw.csv     - raw counts (rows=true, cols=predicted)
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report

warnings.filterwarnings("ignore")

from ..dataset import load_sharp_dataset, get_cv_splits
from ..utils import set_seed, make_ckpt_epochs, CheckpointTracker
from ..config import ANALYSIS_DIR
from ..models.cross_modal_attention import run_fold

OUT_DIR = ANALYSIS_DIR / "confusion_matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE      = ["eda", "emonet", "gaze"]   # All3
BEST_EP     = 100                          # fixed final epoch, no checkpoint selection
N_SEEDS     = 10
CLASS_NAMES = ["Negative", "Positive", "Regulate"]


def main() -> None:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Seeds: {N_SEEDS}  Best checkpoint: ep{BEST_EP:03d}")

    ds = load_sharp_dataset(
        individual=False, load_text=False,
        season="autumn", require_eda=True,
    )
    print(f"Dataset: n={len(ds)}")

    ckpt_epochs = make_ckpt_epochs(100)
    all_true, all_pred = [], []
    seed_f1s = []
    # Pooled (across folds) per-(seed, epoch) predictions, for the full curve.
    seed_true_by_ep = {ep: [] for ep in ckpt_epochs}
    seed_pred_by_ep = {ep: [] for ep in ckpt_epochs}

    for seed in range(N_SEEDS):
        splits = get_cv_splits(ds, random_state=seed)
        seed_true, seed_pred = [], []
        fold_true_by_ep = {ep: [] for ep in ckpt_epochs}
        fold_pred_by_ep = {ep: [] for ep in ckpt_epochs}

        for fold, (tr_idx, va_idx) in enumerate(splits):
            set_seed(seed * 1000 + fold)
            _, _, _, _, _, _, ckpt_preds = run_fold(
                ds, tr_idx, va_idx, ACTIVE, device,
                ckpt_epochs=ckpt_epochs,
                ckpt_save_dir=None,
            )
            for ep in ckpt_epochs:
                if ep in ckpt_preds:
                    yt_ep, yp_ep = ckpt_preds[ep]
                    fold_true_by_ep[ep].extend(yt_ep)
                    fold_pred_by_ep[ep].extend(yp_ep)
            # Use best-checkpoint predictions for the primary confusion matrix
            if BEST_EP in ckpt_preds:
                yt, yp = ckpt_preds[BEST_EP]
            else:
                ep = max(ckpt_preds.keys())
                yt, yp = ckpt_preds[ep]
            seed_true.extend(yt)
            seed_pred.extend(yp)

        sf1 = f1_score(seed_true, seed_pred, average="macro", zero_division=0)
        seed_f1s.append(sf1)
        print(f"  seed {seed:02d}  pooled-OOF F1={sf1:.3f}")
        all_true.extend(seed_true)
        all_pred.extend(seed_pred)
        for ep in ckpt_epochs:
            seed_true_by_ep[ep].append(fold_true_by_ep[ep])
            seed_pred_by_ep[ep].append(fold_pred_by_ep[ep])

    print(f"\nMean F1 = {np.mean(seed_f1s):.3f} ± {np.std(seed_f1s):.3f}")

    # --- per-class recall curve across all checkpoint epochs ---
    curve_rows = []
    for ep in ckpt_epochs:
        recalls_by_class = {c: [] for c in CLASS_NAMES}
        f1s_ep = []
        for s in range(N_SEEDS):
            yt_s, yp_s = seed_true_by_ep[ep][s], seed_pred_by_ep[ep][s]
            cm = confusion_matrix(yt_s, yp_s, labels=[0, 1, 2]).astype(float)
            recall = cm.diagonal() / cm.sum(axis=1)
            for c, r in zip(CLASS_NAMES, recall):
                recalls_by_class[c].append(r)
            f1s_ep.append(f1_score(yt_s, yp_s, average="macro", zero_division=0))
        row = {"epoch": ep, "macro_f1_mean": round(float(np.mean(f1s_ep)), 4),
               "macro_f1_std": round(float(np.std(f1s_ep)), 4)}
        for c in CLASS_NAMES:
            row[f"recall_{c}_mean"] = round(float(np.mean(recalls_by_class[c])), 4)
            row[f"recall_{c}_std"]  = round(float(np.std(recalls_by_class[c])), 4)
        curve_rows.append(row)
    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "confusion_ckpt_curve.csv", index=False)
    print(f"Saved: {(OUT_DIR/'confusion_ckpt_curve.csv').relative_to(ANALYSIS_DIR.parent)}")

    # --- build confusion matrix (pooled across all seeds) ---
    labels = [0, 1, 2]   # Negative=0, Positive=1, Regulate=2
    cm_raw = confusion_matrix(all_true, all_pred, labels=labels)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)  # row-norm (recall)

    # Save raw counts
    pd.DataFrame(cm_raw, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUT_DIR / "confusion_raw.csv"
    )

    # --- plot ---
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)

    for i in range(3):
        for j in range(3):
            raw  = cm_raw[i, j]
            norm = cm_norm[i, j]
            color = "white" if norm > 0.55 else "black"
            ax.text(j, i, f"{norm:.2f}\n({raw})",
                    ha="center", va="center", fontsize=9.5,
                    color=color, fontweight="bold" if i == j else "normal")

    ax.set_title("CrossModal Attention All3", fontsize=11, pad=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUT_DIR}/confusion_matrix.png")
    print(f"\nClassification report (pooled):\n")
    print(classification_report(all_true, all_pred,
                                target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
