#!/usr/bin/env python3
"""
sharp.evaluation.modality_ablation_metrics

Computes Precision / Recall / Accuracy (macro + per-class) for the
Cross-Modal Attention modality-ablation configurations reported in the
paper's Table 2 (Unimodal performance: EDA-only, Gaze-only, Facial-only)
and Table 3 (Modality Combination Ablation: all bimodal pairs plus All3),
reusing checkpoints already saved by sharp.models.cross_modal_attention
--season autumn.

Protocol: n=29, require_eda=True, season=autumn, LOGO 5-fold, pooled-OOF,
epoch=100 (fixed final epoch, no checkpoint selection, matching the
protocol used for Table 4 in sharp.evaluation.epoch100_metrics). 10-seed
mean +/- std.

Outputs: processed_data/analysis/enriched_metrics/
  headline_metrics_modality.csv
  per_class_metrics_modality.csv
"""
from __future__ import annotations

import warnings

import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_recall_fscore_support,
                              precision_score, recall_score)
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")

from ..dataset import load_sharp_dataset, get_cv_splits
from ..config import ANALYSIS_DIR
from ..models.cross_modal_attention import CrossModalAttentionNet as CrossModalNet, collate

CKPT_ROOT = ANALYSIS_DIR / "attn_fusion_autumn" / "checkpoints"
OUT_DIR   = ANALYSIS_DIR / "enriched_metrics"
N_SEEDS   = 10
EP        = 100
CLASS_NAMES = ["Negative", "Positive", "Regulate"]

# tag (== checkpoint dir name == LABELS_MAP value in sharp.models.cross_modal_attention) -> active modalities
CONFIGS = {
    "EDA":         ["eda"],
    "EmoNet":      ["emonet"],
    "Gaze":        ["gaze"],
    "EDA+EmoNet":  ["eda", "emonet"],
    "EDA+Gaze":    ["eda", "gaze"],
    "EmoNet+Gaze": ["emonet", "gaze"],
    "All3":        ["eda", "emonet", "gaze"],
}


def eval_ckpt(model, val_dl, device):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb in val_dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            preds = model(xb).argmax(1).cpu().numpy()
            yt.extend(xb["label"].cpu().numpy().tolist())
            yp.extend(preds.tolist())
    return yt, yp


def macro_metrics(yt, yp):
    return dict(
        precision=precision_score(yt, yp, average="macro", zero_division=0),
        recall=recall_score(yt, yp, average="macro", zero_division=0),
        f1=f1_score(yt, yp, average="macro", zero_division=0),
        accuracy=accuracy_score(yt, yp),
    )


def run_config(tag, active, device, ds, splits):
    per_seed_rows, grand_yt, grand_yp = [], [], []
    for seed in range(N_SEEDS):
        seed_yt, seed_yp = [], []
        for fold, (tr_idx, va_idx) in enumerate(splits):
            ckpt_path = CKPT_ROOT / tag / f"seed{seed:02d}_fold{fold}" / f"ep{EP:03d}.pt"
            model = CrossModalNet(active).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            val_dl = DataLoader(Subset(ds, va_idx), batch_size=len(va_idx), shuffle=False, collate_fn=collate)
            yt, yp = eval_ckpt(model, val_dl, device)
            seed_yt.extend(yt); seed_yp.extend(yp)
        per_seed_rows.append(macro_metrics(seed_yt, seed_yp))
        grand_yt.extend(seed_yt); grand_yp.extend(seed_yp)

    df = pd.DataFrame(per_seed_rows)
    headline = {f"{k}_mean": df[k].mean() for k in df.columns}
    headline.update({f"{k}_std": df[k].std() for k in df.columns})
    headline["model"] = tag
    headline["ckpt_epoch"] = EP
    headline["n_pooled"] = len(grand_yt)
    p, r, f1, _ = precision_recall_fscore_support(grand_yt, grand_yp, labels=[0, 1, 2], zero_division=0)
    per_class = pd.DataFrame({"model": tag, "class": CLASS_NAMES, "precision": p, "recall": r, "f1": f1})
    return headline, per_class


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  Fixed epoch: {EP}")

    ds = load_sharp_dataset(individual=False, load_text=False, season="autumn", require_eda=True)
    splits = get_cv_splits(ds)

    headlines, per_class_tables = [], []
    for tag, active in CONFIGS.items():
        print(f"\n=== {tag} @ ep{EP} ===")
        h, pc = run_config(tag, active, device, ds, splits)
        print(f"  macro-F1={h['f1_mean']:.4f} +/- {h['f1_std']:.4f}")
        headlines.append(h); per_class_tables.append(pc)

    df_head = pd.DataFrame(headlines)
    cols = ["model", "f1_mean", "f1_std", "precision_mean", "precision_std",
            "recall_mean", "recall_std", "accuracy_mean", "accuracy_std",
            "ckpt_epoch", "n_pooled"]
    df_head = df_head[cols]
    df_head.to_csv(OUT_DIR / "headline_metrics_modality.csv", index=False)
    print("\n" + df_head.to_string(index=False))

    df_pc = pd.concat(per_class_tables, ignore_index=True)
    df_pc.to_csv(OUT_DIR / "per_class_metrics_modality.csv", index=False)

    print(f"\nSaved: {OUT_DIR/'headline_metrics_modality.csv'}")
    print(f"Saved: {OUT_DIR/'per_class_metrics_modality.csv'}")


if __name__ == "__main__":
    main()
