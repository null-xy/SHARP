#!/usr/bin/env python3
"""
sharp.stats.correlation_kruskal

Feature vs. event-class association analysis for SHARP socio-emotional
deliberation classification.

Methods (matching domain literature, e.g. Yu et al. 2023):
  1. Shapiro-Wilk normality check per feature per class
  2. Kruskal-Wallis H-test (non-parametric, 3-class) for each feature
  3. Post-hoc pairwise Mann-Whitney U (with Bonferroni correction, 3 pairs)
  4. Effect size: eta-squared (η²) from KW H statistic
  5. Point-biserial r retained for one-vs-rest comparison (as in prior analysis)

Outputs: processed_data/analysis/correlation/
  kruskal_results.csv         — KW H, p, η², significance per feature
  posthoc_mannwhitney.csv     — pairwise MW-U p-values (Bonferroni corrected)
  correlation_pointbiserial.csv — r, p per feature per class (one-vs-rest)
  kruskal_top20_bar.png       — top-20 features by η²
  posthoc_heatmap.png         — post-hoc significance heatmap
  distribution_top12.png      — violin plots for top-12 features by η²
"""
from __future__ import annotations

import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, pointbiserialr, shapiro

from ..config import FM_PATH as DATA_PATH, CORR_DIR as OUT_DIR, TARGET_LABELS as LABELS, CLASS_SHORT_INV, ROOT

warnings.filterwarnings("ignore")

LABEL_SHORT = CLASS_SHORT_INV   # full label → "Positive" / "Negative" / "Regulate"

EMO_COLS = [
    "emo_group_val_delta", "emo_group_aro_delta",
    "emo_p1_val_delta", "emo_p1_aro_delta",
    "emo_p2_val_delta", "emo_p2_aro_delta",
    "emo_p3_val_delta", "emo_p3_aro_delta",
    "emo_speaker_val_delta", "emo_speaker_aro_delta",
]
GAZE_COLS = [
    "gaze_p1_pre_laptop_frac",  "gaze_p1_post_laptop_frac",
    "gaze_p1_pre_peer_frac",    "gaze_p1_post_peer_frac",
    "gaze_p2_pre_laptop_frac",  "gaze_p2_post_laptop_frac",
    "gaze_p2_pre_peer_frac",    "gaze_p2_post_peer_frac",
    "gaze_p3_pre_laptop_frac",  "gaze_p3_post_laptop_frac",
    "gaze_p3_pre_peer_frac",    "gaze_p3_post_peer_frac",
]
EDA_COLS = [
    "eda_group_delta",
    "eda_p1_delta", "eda_p2_delta", "eda_p3_delta",
    "eda_speaker_delta",
]
SCR_COLS = [
    "scr_group_count_delta", "scr_group_amp_delta", "scr_group_hr_delta",
    "scr_p1_count_delta", "scr_p2_count_delta", "scr_p3_count_delta",
    "scr_speaker_count_delta", "scr_speaker_amp_delta", "scr_speaker_hr_delta",
]
# 6.3 emotion trajectory (slope/volatility)
EMO_TRAJ_COLS = [
    "emo_speaker_val_slope_pre",    "emo_speaker_val_slope_post",    "emo_speaker_val_slope_delta",
    "emo_speaker_val_volatility_pre","emo_speaker_val_volatility_post","emo_speaker_val_volatility_delta",
    "emo_speaker_aro_slope_delta",  "emo_speaker_aro_volatility_delta",
    "emo_group_val_slope_delta",    "emo_group_aro_slope_delta",
    "emo_group_val_volatility_delta","emo_group_aro_volatility_delta",
]
# 6.2 utterance density + SSRL speaker features
UTT_COLS = [
    "utt_count_pre",    "utt_count_post",    "utt_count_delta",
    "utt_speech_s_pre", "utt_speech_s_post", "utt_speech_s_delta",
    "utt_silence_frac_pre", "utt_silence_frac_post", "utt_silence_frac_delta",
    "ssrl_speakers_count_pre", "ssrl_speakers_count_post", "ssrl_speakers_count_delta",
    "ssrl_turns_pre",          "ssrl_turns_post",          "ssrl_turns_delta",
]
# 6.4 gaze transition features (group and speaker level)
GAZE_TRANS_COLS = [
    "gaze_group_pre_switch_count",   "gaze_group_post_switch_count",   "gaze_group_delta_switch_count",
    "gaze_group_pre_lp_switch",      "gaze_group_post_lp_switch",      "gaze_group_delta_lp_switch",
    "gaze_group_pre_mean_dur_s",     "gaze_group_post_mean_dur_s",     "gaze_group_delta_mean_dur_s",
    "gaze_speaker_pre_switch_count", "gaze_speaker_post_switch_count", "gaze_speaker_delta_switch_count",
    "gaze_speaker_pre_lp_switch",    "gaze_speaker_post_lp_switch",    "gaze_speaker_delta_lp_switch",
]
ALL_FEAT_COLS = EMO_COLS + GAZE_COLS + EDA_COLS + SCR_COLS + EMO_TRAJ_COLS + UTT_COLS + GAZE_TRANS_COLS

FEAT_LABELS = {
    "emo_group_val_delta":      "Grp valence Δ",
    "emo_group_aro_delta":      "Grp arousal Δ",
    "emo_p1_val_delta":         "P1 valence Δ",
    "emo_p1_aro_delta":         "P1 arousal Δ",
    "emo_p2_val_delta":         "P2 valence Δ",
    "emo_p2_aro_delta":         "P2 arousal Δ",
    "emo_p3_val_delta":         "P3 valence Δ",
    "emo_p3_aro_delta":         "P3 arousal Δ",
    "emo_speaker_val_delta":    "Spk valence Δ",
    "emo_speaker_aro_delta":    "Spk arousal Δ",
    "gaze_p1_pre_laptop_frac":  "P1 pre laptop",
    "gaze_p1_post_laptop_frac": "P1 post laptop",
    "gaze_p1_pre_peer_frac":    "P1 pre peer",
    "gaze_p1_post_peer_frac":   "P1 post peer",
    "gaze_p2_pre_laptop_frac":  "P2 pre laptop",
    "gaze_p2_post_laptop_frac": "P2 post laptop",
    "gaze_p2_pre_peer_frac":    "P2 pre peer",
    "gaze_p2_post_peer_frac":   "P2 post peer",
    "gaze_p3_pre_laptop_frac":  "P3 pre laptop",
    "gaze_p3_post_laptop_frac": "P3 post laptop",
    "gaze_p3_pre_peer_frac":    "P3 pre peer",
    "gaze_p3_post_peer_frac":   "P3 post peer",
    "eda_group_delta":           "Grp EDA Δ",
    "eda_p1_delta":              "P1 EDA Δ",
    "eda_p2_delta":              "P2 EDA Δ",
    "eda_p3_delta":              "P3 EDA Δ",
    "eda_speaker_delta":         "Spk EDA Δ",
    "scr_group_count_delta":     "Grp SCR count Δ",
    "scr_group_amp_delta":       "Grp SCR amp Δ",
    "scr_group_hr_delta":        "Grp HR Δ",
    "scr_p1_count_delta":        "P1 SCR count Δ",
    "scr_p2_count_delta":        "P2 SCR count Δ",
    "scr_p3_count_delta":        "P3 SCR count Δ",
    "scr_speaker_count_delta":   "Spk SCR count Δ",
    "scr_speaker_amp_delta":     "Spk SCR amp Δ",
    "scr_speaker_hr_delta":      "Spk HR Δ",
}

CLASS_COLORS = {
    "Negative": "#d62728",
    "Positive": "#2ca02c",
    "Regulate": "#1f77b4",
}

PAIR_LABELS = ["Neg vs Pos", "Neg vs Reg", "Pos vs Reg"]
PAIRS = [("Negative", "Positive"), ("Negative", "Regulate"), ("Positive", "Regulate")]


def fl(col: str) -> str:
    return FEAT_LABELS.get(col, col)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def eta_squared_kw(H: float, n: int, k: int) -> float:
    """η² = (H - k + 1) / (n - k), clamped to [0, 1]."""
    val = (H - k + 1) / (n - k)
    return float(np.clip(val, 0, 1))


def run_kruskal(groups: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """Returns (H, p, eta_sq)."""
    arrays = [g for g in groups.values() if len(g) > 0]
    if len(arrays) < 2:
        return np.nan, np.nan, np.nan
    try:
        H, p = kruskal(*arrays)
    except ValueError:
        return np.nan, np.nan, np.nan
    n = sum(len(a) for a in arrays)
    k = len(arrays)
    return H, p, eta_squared_kw(H, n, k)


def run_posthoc(groups: dict[str, np.ndarray]) -> dict[str, float]:
    """Bonferroni-corrected Mann-Whitney U p-values for each pair."""
    results = {}
    n_pairs = len(PAIRS)
    for (c1, c2), label in zip(PAIRS, PAIR_LABELS):
        g1, g2 = groups.get(c1, np.array([])), groups.get(c2, np.array([]))
        if len(g1) < 2 or len(g2) < 2:
            results[label] = np.nan
            continue
        try:
            _, p = mannwhitneyu(g1, g2, alternative="two-sided")
            results[label] = min(p * n_pairs, 1.0)   # Bonferroni
        except ValueError:
            results[label] = np.nan
    return results


def run_normality(groups: dict[str, np.ndarray]) -> dict[str, bool]:
    """True = normal (Shapiro-Wilk p>0.05)."""
    out = {}
    for cname, g in groups.items():
        if len(g) < 3:
            out[cname] = False
            continue
        try:
            _, p = shapiro(g)
            out[cname] = p > 0.05
        except Exception:
            out[cname] = False
    return out


def build_feature_groups(df: pd.DataFrame, class_col: str = "class_short",
                          feat_col: str = "") -> dict[str, np.ndarray]:
    out = {}
    for cname in ["Negative", "Positive", "Regulate"]:
        vals = df.loc[df[class_col] == cname, feat_col].dropna().values.astype(float)
        out[cname] = vals
    return out


# ── plots ────────────────────────────────────────────────────────────────────

def plot_kruskal_bar(kw_df: pd.DataFrame, path: Path) -> None:
    top = kw_df.nlargest(20, "eta_sq")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b" if p < 0.05 else "#aaaaaa" for p in top["p_corrected"]]
    ax.barh(range(len(top)), top["eta_sq"].values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([fl(f) for f in top["feature"].values[::-1]], fontsize=9)
    ax.set_xlabel("Effect size η² (Kruskal-Wallis)", fontsize=9)
    ax.set_title("Top-20 Features by Effect Size (Kruskal-Wallis)\n"
                 "Red = significant after FDR correction (p<0.05)", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c0392b", label="p<0.05 (corrected)"),
                        Patch(color="#aaaaaa", label="ns")],
              fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path.relative_to(ROOT)}")


def plot_posthoc_heatmap(kw_df: pd.DataFrame, posthoc_df: pd.DataFrame, path: Path) -> None:
    top20 = kw_df.nlargest(20, "eta_sq")["feature"].values
    sub = posthoc_df[posthoc_df["feature"].isin(top20)].copy()
    sub = sub.set_index("feature").reindex(top20)

    mat = sub[PAIR_LABELS].values.astype(float)
    sig = mat < 0.05

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.2)
    ax.set_xticks(range(len(PAIR_LABELS)))
    ax.set_xticklabels(PAIR_LABELS, fontsize=9)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([fl(f) for f in top20], fontsize=8)
    for ri in range(len(top20)):
        for ci in range(len(PAIR_LABELS)):
            p = mat[ri, ci]
            label = sig_stars(p) if not np.isnan(p) else ""
            ax.text(ci, ri, label, ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="Bonferroni-corrected p-value")
    ax.set_title("Post-hoc Mann-Whitney U (Bonferroni)\nTop-20 features by η²",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path.relative_to(ROOT)}")


def plot_violin_top12(df: pd.DataFrame, kw_df: pd.DataFrame, path: Path) -> None:
    top12_feats = kw_df.nlargest(12, "eta_sq")["feature"].values
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    class_order = ["Negative", "Positive", "Regulate"]
    ccolors = [CLASS_COLORS[c] for c in class_order]

    for i, feat in enumerate(top12_feats):
        ax = axes[i]
        data_by_class = [df.loc[df["class_short"] == c, feat].dropna().values for c in class_order]
        parts = ax.violinplot(data_by_class, positions=[0, 1, 2],
                              showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], ccolors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        # overlay jitter
        rng = np.random.default_rng(42)
        for xi, (vals, color) in enumerate(zip(data_by_class, ccolors)):
            if len(vals):
                jitter = rng.uniform(-0.1, 0.1, len(vals))
                ax.scatter(xi + jitter, vals, s=8, color=color, alpha=0.5, zorder=3)
        row = kw_df[kw_df["feature"] == feat].iloc[0]
        ax.set_title(f"{fl(feat)}\nη²={row['eta_sq']:.3f}, {sig_stars(row['p_corrected'])}",
                     fontsize=8, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Neg", "Pos", "Reg"], fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    for j in range(len(top12_feats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Event Class (Top-12 by Kruskal-Wallis η²)\n"
                 "Violin + jittered points; median line shown",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path.relative_to(ROOT)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_csv(DATA_PATH)
    df = df_full[df_full["deliberation"].isin(LABELS)].copy().reset_index(drop=True)
    df["class_short"] = df["deliberation"].map(LABEL_SHORT)
    print(f"Events: {len(df)}")
    print(df["class_short"].value_counts().to_string(), "\n")

    feat_cols = ALL_FEAT_COLS
    # Impute missing with column mean for analysis
    for col in feat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # ── 1. Kruskal-Wallis ────────────────────────────────────────────────────
    print("Running Kruskal-Wallis tests...")
    kw_rows = []
    posthoc_rows = []
    pb_rows = []

    n_tests = len(feat_cols)
    for feat in feat_cols:
        if feat not in df.columns:
            continue
        groups = build_feature_groups(df, "class_short", feat)
        H, p, eta = run_kruskal(groups)
        norm = run_normality(groups)
        all_normal = all(norm.values())
        kw_rows.append({
            "feature": feat,
            "H": round(H, 4) if not np.isnan(H) else np.nan,
            "p_raw": round(p, 5) if not np.isnan(p) else np.nan,
            "p_corrected": min(p * n_tests, 1.0) if not np.isnan(p) else np.nan,  # Bonferroni
            "eta_sq": round(eta, 4) if not np.isnan(eta) else np.nan,
            "sig": sig_stars(min(p * n_tests, 1.0)) if not np.isnan(p) else "na",
            "all_normal": all_normal,
            "n_Neg": len(groups["Negative"]),
            "n_Pos": len(groups["Positive"]),
            "n_Reg": len(groups["Regulate"]),
        })

        ph = run_posthoc(groups)
        posthoc_rows.append({"feature": feat, **ph})

        # Point-biserial (one-vs-rest)
        pb_row = {"feature": feat}
        for cname in ["Negative", "Positive", "Regulate"]:
            binary = (df["class_short"] == cname).astype(int).values
            vals = df[feat].values.astype(float)
            try:
                r, pv = pointbiserialr(binary, vals)
            except Exception:
                r, pv = np.nan, np.nan
            pb_row[f"r_{cname[:3]}"] = round(r, 4) if not np.isnan(r) else np.nan
            pb_row[f"p_{cname[:3]}"] = round(pv, 5) if not np.isnan(pv) else np.nan
        pb_rows.append(pb_row)

    kw_df = pd.DataFrame(kw_rows)
    posthoc_df = pd.DataFrame(posthoc_rows)
    pb_df = pd.DataFrame(pb_rows)

    # ── Save CSVs ────────────────────────────────────────────────────────────
    kw_path = OUT_DIR / "kruskal_results.csv"
    kw_df.to_csv(kw_path, index=False)
    print(f"Saved: {kw_path.relative_to(ROOT)}")

    ph_path = OUT_DIR / "posthoc_mannwhitney.csv"
    posthoc_df.to_csv(ph_path, index=False)
    print(f"Saved: {ph_path.relative_to(ROOT)}")

    pb_path = OUT_DIR / "correlation_pointbiserial.csv"
    pb_df.to_csv(pb_path, index=False)
    print(f"Saved: {pb_path.relative_to(ROOT)}")

    # ── Print summary ────────────────────────────────────────────────────────
    sig_feats = kw_df[kw_df["p_corrected"] < 0.05].sort_values("eta_sq", ascending=False)
    print(f"\n=== Significant features (KW, Bonferroni p<0.05): {len(sig_feats)}/{len(kw_df)} ===")
    print(f"{'Feature':22s}  {'H':6s}  {'p_corr':8s}  {'η²':6s}  {'sig':4s}")
    print("-" * 55)
    for _, row in sig_feats.iterrows():
        print(f"{fl(row['feature']):22s}  {row['H']:6.2f}  {row['p_corrected']:8.4f}  {row['eta_sq']:6.4f}  {row['sig']}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_kruskal_bar(kw_df, OUT_DIR / "kruskal_top20_bar.png")
    plot_posthoc_heatmap(kw_df, posthoc_df, OUT_DIR / "posthoc_heatmap.png")
    plot_violin_top12(df, kw_df, OUT_DIR / "distribution_top12.png")

    print(f"\nDone. Outputs: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
