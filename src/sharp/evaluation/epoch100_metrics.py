#!/usr/bin/env python3
"""
sharp.evaluation.epoch100_metrics

Computes Precision / Recall / F1 / Accuracy for every model in the paper's
Table 4 (Fusion Strategy Comparison): SVM, Early Fusion, Cross-Modal
Attention, and the five graph-based fusion variants (GNN1-Mean, GNN2-Mean,
Participant-Attn, Serial-GNN-Attn, Parallel-GNN-Attn).

Protocol: all deep-learning models are evaluated at a single, leakage-free
checkpoint -- the literal end of a fixed 100-epoch training budget, with no
best-of-N-checkpoints selection (selecting the reported epoch by looking at
validation-fold performance would itself be a form of information leakage).
All numbers are read from checkpoints already saved to disk during training
(no retraining here). The SVM baseline is deterministic and has no epoch
concept, so its metrics are reused as-is.

Protocol: n=29, require_eda=True, season=autumn, LOGO 5-fold, pooled-OOF,
epoch=100 (fixed budget, no checkpoint selection), 10-seed mean +/- std
for the deep-learning models.

Outputs: processed_data/analysis/enriched_metrics/
  headline_metrics_ep100.csv
  per_class_metrics_ep100.csv
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
from ..models.cross_modal_attention import CrossModalAttentionNet as CrossModalNet, collate as collate15
from ..models.early_fusion import EarlyFusionNet
from ..models.gnn_fusion import collate as collate18, ABLATIONS as _ABLATIONS_LIST

ABLATIONS = dict(_ABLATIONS_LIST)

OUT_DIR   = ANALYSIS_DIR / "enriched_metrics"
N_SEEDS   = 10
EP        = 100
CLASS_NAMES = ["Negative", "Positive", "Regulate"]

# (display name, checkpoint root, checkpoint tag, ModelClass, collate_fn,
#  active-modalities-arg-or-None, individual-level-dataset)
GROUP_MODELS = [
    ("CrossModal Attn",   ANALYSIS_DIR / "missing_modality" / "checkpoints", "CrossModal_All3",
     CrossModalNet, collate15, ["eda", "emonet", "gaze"]),
    ("Early Fusion",      ANALYSIS_DIR / "missing_modality" / "checkpoints", "Early_Fusion_All3",
     EarlyFusionNet, collate15, ["eda", "emonet", "gaze"]),
]
GNN_TAGS = ["GNN1-Mean", "GNN2-Mean", "CrossAttn", "Serial-GNN-Attn", "Parallel-GNN-Attn"]
GNN_DISPLAY = {"CrossAttn": "Participant-Attn"}
GNN_CKPT_ROOT = ANALYSIS_DIR / "gnn_autumn" / "checkpoints"


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


def run_group_model(name, ckpt_root, ckpt_tag, ModelClass, collate, active, device, ds, splits):
    per_seed_rows, grand_yt, grand_yp = [], [], []
    for seed in range(N_SEEDS):
        seed_yt, seed_yp = [], []
        for fold, (tr_idx, va_idx) in enumerate(splits):
            ckpt_path = ckpt_root / ckpt_tag / f"seed{seed:02d}_fold{fold}" / f"ep{EP:03d}.pt"
            model = ModelClass(active).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            val_dl = DataLoader(Subset(ds, va_idx), batch_size=len(va_idx), shuffle=False, collate_fn=collate)
            yt, yp = eval_ckpt(model, val_dl, device)
            seed_yt.extend(yt); seed_yp.extend(yp)
        per_seed_rows.append(macro_metrics(seed_yt, seed_yp))
        grand_yt.extend(seed_yt); grand_yp.extend(seed_yp)
    return _finalize(name, per_seed_rows, grand_yt, grand_yp)


def run_gnn_variant(tag, device, ds, splits):
    model_fn = ABLATIONS[tag]
    per_seed_rows, grand_yt, grand_yp = [], [], []
    for seed in range(N_SEEDS):
        seed_yt, seed_yp = [], []
        for fold, (tr_idx, va_idx) in enumerate(splits):
            ckpt_path = GNN_CKPT_ROOT / tag / f"seed{seed:02d}_fold{fold}" / f"ep{EP:03d}.pt"
            model = model_fn().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            val_dl = DataLoader(Subset(ds, va_idx), batch_size=len(va_idx), shuffle=False, collate_fn=collate18)
            yt, yp = eval_ckpt(model, val_dl, device)
            seed_yt.extend(yt); seed_yp.extend(yp)
        per_seed_rows.append(macro_metrics(seed_yt, seed_yp))
        grand_yt.extend(seed_yt); grand_yp.extend(seed_yp)
    return _finalize(GNN_DISPLAY.get(tag, tag), per_seed_rows, grand_yt, grand_yp)


def _finalize(name, per_seed_rows, grand_yt, grand_yp):
    df = pd.DataFrame(per_seed_rows)
    headline = {f"{k}_mean": df[k].mean() for k in df.columns}
    headline.update({f"{k}_std": df[k].std() for k in df.columns})
    headline["model"] = name
    headline["ckpt_epoch"] = EP
    headline["n_pooled"] = len(grand_yt)
    p, r, f1, _ = precision_recall_fscore_support(grand_yt, grand_yp, labels=[0, 1, 2], zero_division=0)
    per_class = pd.DataFrame({"model": name, "class": CLASS_NAMES, "precision": p, "recall": r, "f1": f1})
    return headline, per_class


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  Fixed epoch: {EP} (no checkpoint selection)")

    headlines, per_class_tables = [], []

    ds_group = load_sharp_dataset(individual=False, load_text=False, season="autumn", require_eda=True)
    splits_group = get_cv_splits(ds_group)
    for name, ckpt_root, ckpt_tag, ModelClass, collate, active in GROUP_MODELS:
        print(f"\n=== {name} @ ep{EP} ===")
        h, pc = run_group_model(name, ckpt_root, ckpt_tag, ModelClass, collate, active, device, ds_group, splits_group)
        print(f"  macro-F1={h['f1_mean']:.4f} +/- {h['f1_std']:.4f}")
        headlines.append(h); per_class_tables.append(pc)

    ds_ind = load_sharp_dataset(individual=True, season="autumn", require_eda=True)
    splits_ind = get_cv_splits(ds_ind)
    for tag in GNN_TAGS:
        name = GNN_DISPLAY.get(tag, tag)
        print(f"\n=== {name} (ckpt tag={tag}) @ ep{EP} ===")
        h, pc = run_gnn_variant(tag, device, ds_ind, splits_ind)
        print(f"  macro-F1={h['f1_mean']:.4f} +/- {h['f1_std']:.4f}")
        headlines.append(h); per_class_tables.append(pc)

    df_head = pd.DataFrame(headlines)
    cols = ["model", "f1_mean", "f1_std", "precision_mean", "precision_std",
            "recall_mean", "recall_std", "accuracy_mean", "accuracy_std",
            "ckpt_epoch", "n_pooled"]
    df_head = df_head[cols]
    df_head.to_csv(OUT_DIR / "headline_metrics_ep100.csv", index=False)
    print("\n" + df_head.to_string(index=False))

    df_pc = pd.concat(per_class_tables, ignore_index=True)
    df_pc.to_csv(OUT_DIR / "per_class_metrics_ep100.csv", index=False)
    print("\n" + df_pc.to_string(index=False))

    print(f"\nSaved: {OUT_DIR/'headline_metrics_ep100.csv'}")
    print(f"Saved: {OUT_DIR/'per_class_metrics_ep100.csv'}")


if __name__ == "__main__":
    main()
