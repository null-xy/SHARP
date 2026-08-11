#!/usr/bin/env python3
"""
sharp.models.svm_baseline

2x2 ablation: Feature Source x Classifier, plus a third classifier
(CrossModal-Feature attention) applied to the engineered feature matrix.
This is the source of the SVM number reported in the paper's Table 4
(Fusion Strategy Comparison).

  Feature source:  feature_matrix.csv (hand-crafted stats, n=29)
                   raw time series (1 Hz, 60 s window, n=29)
  Classifier:      SVM (RBF)
                   MLP (3-layer FC)
                   CrossModal-Feature (feature vector -> linear proj -> cross-attn)

Protocol: LeaveOneGroupOut n=29 (same as the deep-learning scripts), 10 seeds.
All conditions share the same 5 LOGO folds, so paired tests (e.g. Wilcoxon)
across conditions are valid.

Outputs: processed_data/analysis/feature_ablation_{season}/
  results.csv   - per-seed x per-fold rows (condition, feature_set, seed, fold, fold_group, val_f1)
  summary.csv   - mean +/- std per condition
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

from ..config import FM_PATH, ANALYSIS_DIR, TARGET_LABELS as LABELS, ROOT
from ..dataset import (load_sharp_dataset, get_cv_splits,
                        impute_eda_fold, restore_eda)

# ── Feature columns ────────────────────────────────────────────────────────────

# Original feature sets (per-person decomposed)
EMO_COLS = [
    "emo_group_val_delta", "emo_group_aro_delta",
    "emo_p1_val_delta", "emo_p1_aro_delta",
    "emo_p2_val_delta", "emo_p2_aro_delta",
    "emo_p3_val_delta", "emo_p3_aro_delta",
    "emo_speaker_val_delta", "emo_speaker_aro_delta",
]
GAZE_COLS = [
    "gaze_p1_pre_laptop_frac", "gaze_p1_post_laptop_frac",
    "gaze_p1_pre_peer_frac",   "gaze_p1_post_peer_frac",
    "gaze_p2_pre_laptop_frac", "gaze_p2_post_laptop_frac",
    "gaze_p2_pre_peer_frac",   "gaze_p2_post_peer_frac",
    "gaze_p3_pre_laptop_frac", "gaze_p3_post_laptop_frac",
    "gaze_p3_pre_peer_frac",   "gaze_p3_post_peer_frac",
]
EDA_COLS = [
    "eda_group_delta",
    "eda_p1_delta", "eda_p2_delta", "eda_p3_delta",
    "eda_speaker_delta",
]

# Expert-grouped feature sets (group/speaker aggregates + trajectory)
# EmoNet: group/speaker mean delta + slope_delta (dynamics, not just level)
EMO_COLS_EXPERT = [
    "emo_group_val_delta",         "emo_group_aro_delta",
    "emo_speaker_val_delta",       "emo_speaker_aro_delta",
    "emo_group_val_slope_delta",   "emo_group_aro_slope_delta",
    "emo_speaker_val_slope_delta", "emo_speaker_aro_slope_delta",
]
# Gaze: speaker/group behavioral transitions (not per-person static fracs)
GAZE_COLS_EXPERT = [
    "gaze_speaker_delta_lp_switch",
    "gaze_speaker_delta_switch_count",
    "gaze_speaker_delta_mean_dur_s",
    "gaze_group_delta_lp_switch",
    "gaze_group_delta_switch_count",
    "gaze_group_delta_mean_dur_s",
]

# 6-token Change-Token feature families
# EmoNet trajectory: slope + volatility delta (dynamics, not just mean level)
EMO_TRAJ_COLS = [
    "emo_group_val_slope_delta",   "emo_group_aro_slope_delta",
    "emo_speaker_val_slope_delta", "emo_speaker_aro_slope_delta",
    "emo_group_val_volatility_delta",   "emo_group_aro_volatility_delta",
    "emo_speaker_val_volatility_delta", "emo_speaker_aro_volatility_delta",
]
# Gaze allocation: per-person pre/post frac (how gaze time is allocated)
GAZE_ALLOC_COLS = GAZE_COLS   # same 12 laptop+peer frac features
# Gaze transition: speaker/group behavioral switching delta
GAZE_TRANS_COLS = GAZE_COLS_EXPERT  # same 6 lp_switch/switch_count/mean_dur
# Coupling: EDA-EmoNet coupling change (directly encodes D-metric signal)
COUPLING_COLS = [
    "coupling_aro_eda_group_delta",   "coupling_val_eda_group_delta",
    "coupling_aro_eda_p1_delta",      "coupling_val_eda_p1_delta",
    "coupling_aro_eda_p2_delta",      "coupling_val_eda_p2_delta",
    "coupling_aro_eda_p3_delta",      "coupling_val_eda_p3_delta",
    "coupling_aro_eda_speaker_delta", "coupling_val_eda_speaker_delta",
]

ALL_CHANGE_COLS = EDA_COLS + EMO_COLS + EMO_TRAJ_COLS + GAZE_ALLOC_COLS + GAZE_TRANS_COLS + COUPLING_COLS

FEATURE_SETS = {
    "EmoNet+Gaze+EDA":    EMO_COLS + GAZE_COLS + EDA_COLS,          # 27 feats, 3 tokens
    "Change-Token":       ALL_CHANGE_COLS,                            # 51 feats, 6 tokens
}

RAW_ABLATIONS = {
    "EmoNet+Gaze+EDA": ["eda", "emonet", "gaze"],
}

# Modality split for CrossModal-Feature attention
FEAT_MODALITY_SPLIT = {
    "EmoNet+Gaze+EDA": {
        "EDA":    EDA_COLS,
        "EmoNet": EMO_COLS,
        "Gaze":   GAZE_COLS,
    },
    "Change-Token": {
        "EDA-change":   EDA_COLS,
        "EmoNet-level": EMO_COLS,
        "EmoNet-traj":  EMO_TRAJ_COLS,
        "Gaze-alloc":   GAZE_ALLOC_COLS,
        "Gaze-trans":   GAZE_TRANS_COLS,
        "Coupling":     COUPLING_COLS,
    },
}


# ── MLP on feature matrix ─────────────────────────────────────────────────────

class FeatureMLP(nn.Module):
    def __init__(self, n_in: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_mlp_features(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    device: str,
    fixed_epochs: int = 200,
    lr: float = 3e-4, wd: float = 1e-2,
) -> float:
    counts = np.bincount(y_tr, minlength=3).astype(float)
    cw = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw)
    model = FeatureMLP(X_tr.shape[1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fixed_epochs)
    Xtr = torch.from_numpy(X_tr).float().to(device)
    ytr = torch.from_numpy(y_tr).long().to(device)
    Xte = torch.from_numpy(X_te).float().to(device)
    for _ in range(fixed_epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xtr), ytr); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1).cpu().numpy()
    return float(f1_score(y_te, preds, average="macro", zero_division=0)), preds


# ── CrossModal-Feature attention (modality-level embedding) ───────────────────
#
# Design: each modality's full feature vector -> linear projection -> a single
# embedding. The 3 modality embeddings form a sequence -> Transformer
# cross-modal attention -> classification.
# This is the feature-matrix equivalent of the raw-stream CrossModal model:
#   raw version:      modality time series -> Conv1d -> token sequence -> cross-attn
#   feature version:  modality feature vector -> Linear -> single token -> cross-attn
#
# n=29 is a very small sample, so capacity is kept low: dim=16, dropout=0.4,
# 1 Transformer layer, no positional embedding.

class CrossModalFeatV2(nn.Module):
    def __init__(self, modal_sizes: list[int], dim: int = 16, num_classes: int = 3,
                 dropout: float = 0.4):
        super().__init__()
        n_mod = len(modal_sizes)
        # Each modality: full feature vector -> a single dim-dimensional embedding
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for n in modal_sizes
        ])
        # Cross-modal attention across n_mod tokens (one per modality)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=2, dim_feedforward=dim * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            ),
            num_layers=1,
        )
        self.norm = nn.LayerNorm(dim)
        # Classification head: concatenate the attended embedding of every modality
        self.head = nn.Linear(dim * n_mod, num_classes)

    def forward(self, modals: list[torch.Tensor]) -> torch.Tensor:
        embs = [proj(m) for proj, m in zip(self.projectors, modals)]
        x = torch.stack(embs, dim=1)          # (B, n_mod, dim)
        x = self.transformer(x)               # cross-modal attention
        x = self.norm(x)
        out = x.reshape(x.shape[0], -1)       # (B, n_mod * dim)
        return self.head(out)


def run_crossmodal_feature(
    X_tr_mods: list[np.ndarray], y_tr: np.ndarray,
    X_te_mods: list[np.ndarray], y_te: np.ndarray,
    device: str,
    fixed_epochs: int = 300,
    lr: float = 1e-3, wd: float = 5e-2,
    dim: int = 16,
) -> float:
    modal_sizes = [m.shape[1] for m in X_tr_mods]
    counts = np.bincount(y_tr, minlength=3).astype(float)
    cw = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw)
    model = CrossModalFeatV2(modal_sizes, dim=dim, dropout=0.4).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fixed_epochs)

    tr_mods = [torch.from_numpy(m).float().to(device) for m in X_tr_mods]
    te_mods = [torch.from_numpy(m).float().to(device) for m in X_te_mods]
    ytr = torch.from_numpy(y_tr).long().to(device)

    for _ in range(fixed_epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(tr_mods), ytr); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    model.eval()
    with torch.no_grad():
        preds = model(te_mods).argmax(1).cpu().numpy()
    return float(f1_score(y_te, preds, average="macro", zero_division=0)), preds


# ── Late Fusion: per-modality SVM → average decision scores ──────────────────
#
# Trains one SVM per modality on the same LOGO fold, then averages decision
# function scores (OvR) across modalities before argmax.
# This is the standard late-fusion baseline for comparison with CrossModal-Feat.

def run_late_fusion(
    X_tr_mods: list[np.ndarray], y_tr: np.ndarray,
    X_te_mods: list[np.ndarray], y_te: np.ndarray,
    seed: int,
) -> tuple[float, np.ndarray]:
    score_acc = np.zeros((len(y_te), 3), dtype=float)
    for X_tr, X_te in zip(X_tr_mods, X_te_mods):
        clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                  class_weight="balanced", decision_function_shape="ovr",
                  random_state=seed)
        clf.fit(X_tr, y_tr)
        df_raw = clf.decision_function(X_te)   # (n_te, n_cls) or (n_te,) if 1 class
        if df_raw.ndim == 1:
            df_raw = df_raw.reshape(-1, 1)
        # align to full 3-class grid (handles rare missing class in clf.classes_)
        for j, c in enumerate(clf.classes_):
            score_acc[:, c] += df_raw[:, j] if df_raw.shape[1] > j else df_raw[:, 0]
    preds = score_acc.argmax(axis=1)
    f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
    return f1, preds


# ── SVM / MLP on raw time series ──────────────────────────────────────────────

def flatten_batch(ds, idx: np.ndarray, active: list[str]) -> np.ndarray:
    parts = []
    for i in idx:
        s = ds._samples[i]
        vecs = [s[m].numpy().ravel() for m in active]
        parts.append(np.concatenate(vecs))
    return np.stack(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(season: str | None = None, n_seeds: int = 10) -> None:
    out_dir = ANALYSIS_DIR / ("feature_ablation" + (f"_{season}" if season else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Seeds: {n_seeds}")
    print("Protocol: LOGO n=29 (require_eda=True)\n")

    # ── Load raw dataset (n=29, LOGO) ─────────────────────────────────────────
    ds = load_sharp_dataset(individual=False, season=season, require_eda=True)
    print(f"Raw time-series events: {len(ds)}")

    # ── Load + filter feature matrix to same n=29 events ──────────────────────
    df_full = pd.read_csv(FM_PATH)
    if season == "autumn":
        df_full = df_full[~df_full["group"].str.startswith("SD")]
    elif season == "spring":
        df_full = df_full[df_full["group"].str.startswith("SD")]
    df_full = df_full[df_full["deliberation"].isin(LABELS)].copy()
    df_full["group"] = df_full["group"].str.upper()

    valid_keys = {(s["group"], round(s["start_sec"])) for s in ds._samples}
    df = df_full[
        df_full.apply(
            lambda r: (r["group"], round(float(r["start_sec"]))) in valid_keys, axis=1
        )
    ].copy().reset_index(drop=True)
    print(f"Feature matrix events (after EDA filter): {len(df)}")
    print(df["deliberation"].value_counts().to_string(), "\n")

    le = LabelEncoder()
    y_fm = le.fit_transform(df["deliberation"])
    groups_fm = df["group"].values
    logo_fm   = LeaveOneGroupOut()

    result_rows: list[dict] = []
    seed_rows:   list[dict] = []

    conditions = [
        ("SVM",             "feature_matrix"),
        ("MLP",             "feature_matrix"),
        ("CrossModal-Feat", "feature_matrix"),
        ("LateFusion",      "feature_matrix"),
        ("SVM",             "raw_timeseries"),
        ("MLP",             "raw_timeseries"),
    ]

    for clf_name, feat_src in conditions:
        label    = f"{clf_name}+{feat_src}"
        ablations = FEATURE_SETS if feat_src == "feature_matrix" else RAW_ABLATIONS

        for fs_name, cols in ablations.items():
            seed_f1s: list[float] = []
            print(f"  {label}  {fs_name}", flush=True)

            for seed in range(n_seeds):
                seed_y_true: list[int] = []
                seed_y_pred: list[int] = []

                if feat_src == "feature_matrix":
                    splits_fm = list(logo_fm.split(np.zeros(len(y_fm)), y_fm, groups_fm))
                    for fold_i, (tr_idx, te_idx) in enumerate(splits_fm):
                        fold_group = groups_fm[te_idx[0]]
                        X_tr = df.iloc[tr_idx][cols].values.astype(float)
                        X_te = df.iloc[te_idx][cols].values.astype(float)
                        y_tr = y_fm[tr_idx]
                        y_te = y_fm[te_idx]

                        imp = SimpleImputer(strategy="mean")
                        X_tr = imp.fit_transform(X_tr); X_te = imp.transform(X_te)

                        if clf_name in ("CrossModal-Feat", "LateFusion"):
                            scaler = StandardScaler()
                            X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
                            modal_cols = FEAT_MODALITY_SPLIT[fs_name]
                            modal_keys = list(modal_cols.keys())
                            all_cols = cols
                            tr_mods = [
                                X_tr[:, [all_cols.index(c) for c in modal_cols[k]]]
                                for k in modal_keys
                            ]
                            te_mods = [
                                X_te[:, [all_cols.index(c) for c in modal_cols[k]]]
                                for k in modal_keys
                            ]
                            if clf_name == "CrossModal-Feat":
                                # 6-token Change-Token → dim=8 (more tokens, smaller per-token dim);
                                # 3-token EmoNet+Gaze+EDA → dim=16
                                _dim = 8 if fs_name == "Change-Token" else 16
                                f1, fold_preds = run_crossmodal_feature(tr_mods, y_tr, te_mods, y_te, device, dim=_dim)
                            else:
                                f1, fold_preds = run_late_fusion(tr_mods, y_tr, te_mods, y_te, seed)
                        else:
                            scaler = StandardScaler()
                            X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
                            if clf_name == "SVM":
                                clf_obj = SVC(kernel="rbf", C=1.0, gamma="scale",
                                              class_weight="balanced")
                                clf_obj.fit(X_tr, y_tr)
                                fold_preds = clf_obj.predict(X_te)
                                f1 = float(f1_score(y_te, fold_preds, average="macro",
                                                    zero_division=0))
                            else:
                                f1, fold_preds = run_mlp_features(X_tr, y_tr, X_te, y_te, device)

                        seed_y_true.extend(y_te.tolist())
                        seed_y_pred.extend(fold_preds.tolist())
                        result_rows.append({
                            "condition": label, "feature_set": fs_name,
                            "seed": seed, "fold": fold_i, "fold_group": fold_group,
                            "val_f1": round(f1, 4),
                        })

                else:  # raw_timeseries
                    splits = get_cv_splits(ds, random_state=seed)
                    for fold_i, (tr_idx, te_idx) in enumerate(splits):
                        fold_group = ds._samples[te_idx[0]]["group"]
                        saved = impute_eda_fold(ds, tr_idx) if "eda" in cols else {}
                        X_tr = flatten_batch(ds, tr_idx, cols)
                        X_te = flatten_batch(ds, te_idx,  cols)
                        y_tr = ds.labels[tr_idx]
                        y_te = ds.labels[te_idx]
                        restore_eda(ds, saved)

                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)

                        if clf_name == "SVM":
                            clf_obj = SVC(kernel="rbf", C=1.0, gamma="scale",
                                          class_weight="balanced")
                            clf_obj.fit(X_tr, y_tr)
                            fold_preds = clf_obj.predict(X_te)
                            f1 = float(f1_score(y_te, fold_preds, average="macro",
                                                zero_division=0))
                        else:
                            f1, fold_preds = run_mlp_features(X_tr, y_tr, X_te, y_te, device)

                        seed_y_true.extend(y_te.tolist())
                        seed_y_pred.extend(fold_preds.tolist())
                        result_rows.append({
                            "condition": label, "feature_set": fs_name,
                            "seed": seed, "fold": fold_i, "fold_group": fold_group,
                            "val_f1": round(f1, 4),
                        })

                seed_f1 = float(f1_score(seed_y_true, seed_y_pred, average="macro", zero_division=0))
                seed_f1s.append(seed_f1)
                seed_rows.append({"condition": label, "feature_set": fs_name,
                                   "seed": seed, "seed_f1": round(seed_f1, 4)})

            mean_f1 = float(np.mean(seed_f1s))
            std_f1  = float(np.std(seed_f1s))
            print(f"    → F1={mean_f1:.3f}±{std_f1:.3f}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(result_rows)
    df_res.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_dir / "results_per_seed.csv", index=False)

    summary = (pd.DataFrame(seed_rows).groupby(["condition", "feature_set"])["seed_f1"]
               .agg(["mean", "std"]).round(4))
    summary.columns = ["f1_mean", "f1_std"]
    summary.to_csv(out_dir / "summary.csv")

    print(f"\n=== 2×2+CrossModal Summary (LOGO n=29, pooled-OOF macro-F1 per seed, mean±std over {n_seeds} seeds) ===")
    print(summary.to_string())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fs_names  = list(FEATURE_SETS.keys())
    clf_names = ["SVM", "MLP", "CrossModal-Feat", "LateFusion"]
    colors    = ["#4878d0", "#ee854a", "#6acc65", "#da8bc3"]
    x         = np.arange(len(fs_names))
    w         = 0.18

    for ax, feat_src, title in zip(
        axes,
        ["feature_matrix", "raw_timeseries"],
        ["Feature Matrix: Original vs Expert-Grouped (LOGO n=29)",
         "Raw Time Series (1 Hz 60 s, LOGO n=29)"],
    ):
        shown_clfs = clf_names if feat_src == "feature_matrix" else ["SVM", "MLP"]
        for i, clf_name in enumerate(shown_clfs):
            cond = f"{clf_name}+{feat_src}"
            vals, errs = [], []
            for fs in fs_names:
                if (cond, fs) in summary.index:
                    row = summary.xs((cond, fs))
                    vals.append(row["f1_mean"]); errs.append(row["f1_std"])
                else:
                    vals.append(0); errs.append(0)
            offset = (i - len(shown_clfs) / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, yerr=errs, capsize=4,
                          label=clf_name, color=colors[i],
                          alpha=0.85, edgecolor="black", linewidth=0.6)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.axhline(0.333, color="gray", linestyle=":", linewidth=1, label="Chance")
        ax.set_xticks(x); ax.set_xticklabels(fs_names, fontsize=9)
        ax.set_ylim(0, 1.0); ax.set_ylabel("macro-F1")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Feature Engineering vs Raw Streams × Classifier (LOGO n=29)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "2x2_ablation.png", dpi=150)
    plt.close()
    print(f"\nSaved: {(out_dir / '2x2_ablation.png').relative_to(ROOT)}")
    print(f"Outputs: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["autumn", "spring"], default=None)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    main(season=args.season, n_seeds=args.seeds)
