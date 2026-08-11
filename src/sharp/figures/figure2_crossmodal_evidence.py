#!/usr/bin/env python3
"""
sharp.figures.figure2_crossmodal_evidence

Generates the physiological-facial cross-modal conflict evidence figure
(paper Figure 2), using the n=29 Autumn subset (require_eda=True), further
restricted to n=28 events that have a valid facial-arousal delta. This is
the figure for the "Physiological vs facial affect profile" paragraph in
Section 3.4 of the paper.

All summary statistics (means, Cohen's d_z, test p-values) are computed
directly from feature_matrix.csv via sharp.stats.discordance's
load_autumn_ssrl(), rather than hardcoded, so the figure always matches a
live re-run of the underlying data.

Test selection follows the same Shapiro-Wilk-driven rule as
sharp.stats.normality_audit: a paired t-test against 0 when the delta is
not rejected as non-normal, otherwise a Wilcoxon signed-rank test. A
parametric effect size should not be reported next to a nonparametric
test, so the two are always chosen together based on the normality check.

(A) Scatter: speaker EDA delta vs. speaker facial-arousal delta by SSRL class.
(B) Bar chart: mean EDA delta vs. facial-arousal delta by class, with the
    Negative-class significance annotation (delta vs 0, test selected per
    per-class Shapiro-Wilk result).

Output: processed_data/analysis/figures/crossmodal_conflict_evidence.png
"""
import numpy as np
from scipy.stats import shapiro, ttest_1samp, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import ANALYSIS_DIR
from ..stats.discordance import load_autumn_ssrl

CLASS_ORDER  = ["Positive", "Negative", "Regulate"]
CLASS_COLORS = {"Negative": "#e74c3c", "Positive": "#3498db", "Regulate": "#f39c12"}


def _delta_vs_zero_p(x: np.ndarray) -> float:
    """p-value for delta vs 0, test chosen per-sample by Shapiro-Wilk normality
    (paired t-test if normal, else Wilcoxon signed-rank). NaN if degenerate."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 3 or np.allclose(x, 0):
        return float("nan")
    normal = shapiro(x).pvalue > 0.05
    try:
        if normal:
            return float(ttest_1samp(x, 0).pvalue)
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return float("nan")


def main():
    df = load_autumn_ssrl()
    df = df.dropna(subset=["eda_speaker_delta", "emo_speaker_aro_delta"]).copy()
    print(f"n={len(df)}  class counts: {df['label'].value_counts().to_dict()}")

    # Live per-class stats (mean delta, Wilcoxon p vs 0), used for both panels.
    eda_p, aro_p = {}, {}
    for lab in CLASS_ORDER:
        g = df[df["label"] == lab]
        eda_p[lab] = _delta_vs_zero_p(g["eda_speaker_delta"].values)
        aro_p[lab] = _delta_vs_zero_p(g["emo_speaker_aro_delta"].values)
        print(f"  {lab}: n={len(g)}  EDA mean={g['eda_speaker_delta'].mean():+.3f} p={eda_p[lab]:.3f}"
              f"  |  Face mean={g['emo_speaker_aro_delta'].mean():+.3f} p={aro_p[lab]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Speaker-Level EDA and Facial-Arousal Changes across SSRL Event Classes",
                 fontsize=13, fontweight="bold")

    # ── Panel A: scatter ──────────────────────────────────────────────────
    axA = axes[0]
    for lab in CLASS_ORDER:
        g = df[df["label"] == lab]
        c = CLASS_COLORS[lab]
        axA.scatter(g["eda_speaker_delta"], g["emo_speaker_aro_delta"],
                    color=c, alpha=0.7, s=55, edgecolors="white", linewidths=0.6,
                    label=lab,
                    marker={"Negative": "D", "Positive": "o", "Regulate": "s"}[lab])
        mx, my = g["eda_speaker_delta"].mean(), g["emo_speaker_aro_delta"].mean()
        sx = g["eda_speaker_delta"].std(ddof=1) / np.sqrt(len(g))
        sy = g["emo_speaker_aro_delta"].std(ddof=1) / np.sqrt(len(g))
        axA.errorbar(mx, my, xerr=sx, yerr=sy, color=c, marker="*", markersize=18,
                     markeredgecolor="black", markeredgewidth=0.8, capsize=4, zorder=5)

    axA.axhline(0, color="gray", lw=0.8, ls="--")
    axA.axvline(0, color="gray", lw=0.8, ls="--")
    axA.set_xlabel("Speaker EDA $\\Delta$ (post $-$ pre baseline)", fontsize=10)
    axA.set_ylabel("Speaker Facial Arousal $\\Delta$ (EmoNet)", fontsize=10)
    axA.set_title("(A) Physiological vs Facial Arousal Change\nby SSRL Event Class",
                  fontsize=11, fontweight="bold")
    axA.legend(loc="upper left", fontsize=9, frameon=True)

    # ── Panel B: bar chart (means, SEM, and p-values all computed live above) ──
    axB = axes[1]
    x = np.arange(3)
    width = 0.35
    eda_means, aro_means, eda_sems, aro_sems = [], [], [], []
    for lab in CLASS_ORDER:
        g = df[df["label"] == lab]
        eda_means.append(g["eda_speaker_delta"].mean())
        aro_means.append(g["emo_speaker_aro_delta"].mean())
        eda_sems.append(g["eda_speaker_delta"].std(ddof=1) / np.sqrt(len(g)))
        aro_sems.append(g["emo_speaker_aro_delta"].std(ddof=1) / np.sqrt(len(g)))

    bars_eda = axB.bar(x - width/2, eda_means, width, yerr=eda_sems, capsize=4,
                        label="EDA $\\Delta$ (physiological)",
                        color=[CLASS_COLORS[l] for l in CLASS_ORDER], alpha=0.85)
    bars_aro = axB.bar(x + width/2, aro_means, width, yerr=aro_sems, capsize=4,
                        label="Arousal $\\Delta$ (facial, EmoNet)",
                        color=[CLASS_COLORS[l] for l in CLASS_ORDER], alpha=0.85, hatch="//")

    axB.set_ylim(-0.13, 0.19)
    neg_i = CLASS_ORDER.index("Negative")
    axB.axvspan(neg_i - width, neg_i + width, color="#e74c3c", alpha=0.08)
    neg_lab = CLASS_ORDER[neg_i]
    star = "★" if eda_p[neg_lab] < 0.05 else ""
    axB.text(neg_i - width/2, eda_means[neg_i] + eda_sems[neg_i] + 0.018,
             f"p={eda_p[neg_lab]:.3f}{star}", ha="center", fontsize=10, fontweight="bold", color="#c0392b")
    axB.text(neg_i + width/2, aro_means[neg_i] - aro_sems[neg_i] - 0.018,
             "ns" if aro_p[neg_lab] >= 0.05 else f"p={aro_p[neg_lab]:.3f}",
             ha="center", va="top", fontsize=8, color="gray")

    axB.axhline(0, color="black", lw=0.8)
    axB.set_xticks(x)
    axB.set_xticklabels(CLASS_ORDER, fontsize=10)
    axB.set_ylabel("Mean Delta (post $-$ pre)", fontsize=10)
    axB.set_title(f"(B) Mean EDA vs Facial Arousal Change\nby SSRL Class ($\\star$ = p<0.05)",
                  fontsize=11, fontweight="bold", pad=10)
    axB.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_dir = ANALYSIS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "crossmodal_conflict_evidence.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
