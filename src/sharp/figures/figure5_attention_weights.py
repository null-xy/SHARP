#!/usr/bin/env python3
"""
sharp.figures.figure5_attention_weights

Generates the paper's Figure 5 (Interpretability Analysis): mean peak
cross-modal attention weight for all 6 directed modality pairs, broken
down by SSRL event class (Cross-Modal Attention, all three modalities,
n=29, 10 seeds).

(Figure 3 in the paper is unrelated to this script -- it's the hand-drawn
framework diagram, `overleaf_paper/figures/pipeline.pdf`, not something
generated from data.)

Source data: sharp.evaluation.attention_heatmap --season autumn
             (processed_data/analysis/attention_heatmap_autumn/attention_weights_raw.csv)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import ANALYSIS_DIR

def main():
    print("Generating Figure 5 (Cross-modal attention selectivity by class)...")

    csv_path = ANALYSIS_DIR / "attention_heatmap_autumn" / "attention_weights_raw.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run "
              "`python -m sharp.evaluation.attention_heatmap --season autumn` first.")
        return

    df = pd.read_csv(csv_path)
    assert len(df) == 29, f"Expected n=29, got {len(df)}"

    # All 6 directed pairs among the 3 modalities (Eq. 4 in the paper's
    # Interpretability Analysis section).
    pairs = ["eda←gaze", "eda←emonet", "gaze←emonet", "gaze←eda", "emonet←eda", "emonet←gaze"]
    pair_labels = ["EDA ← Gaze", "EDA ← EmoNet", "Gaze ← EmoNet", "Gaze ← EDA", "EmoNet ← EDA", "EmoNet ← Gaze"]

    neg_means = [df[df["label"] == "Negative"][f"focus_{p}"].mean() for p in pairs]
    pos_means = [df[df["label"] == "Positive"][f"focus_{p}"].mean() for p in pairs]
    reg_means = [df[df["label"] == "Regulate"][f"focus_{p}"].mean() for p in pairs]

    CLASS_COLORS = {"Negative": "#e74c3c", "Positive": "#3498db", "Regulate": "#f39c12"}
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(pairs))
    width = 0.26

    ax.bar(x - width, neg_means, width, label='Negative Events', color=CLASS_COLORS["Negative"], alpha=0.7)
    ax.bar(x,          pos_means, width, label='Positive Events', color=CLASS_COLORS["Positive"], alpha=0.7)
    ax.bar(x + width,  reg_means, width, label='Regulate Events', color=CLASS_COLORS["Regulate"], alpha=0.7)
    ax.axhline(0.125, color='gray', linestyle='--', linewidth=0.8, label='Uniform attention (1/T)')

    ax.set_ylabel('Mean Peak Attention Weight', fontsize=11)
    ax.set_title('Cross-Modal Mean Peak Attention Weight by SSRL Event Class',
                 fontweight='bold', fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_ylim(0, max(neg_means + pos_means + reg_means) * 1.25)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Annotate the near-zero Neg-vs-Pos gap on Gaze<-EDA (the one route that would
    # show a pivot toward physiology, and does not).
    gze_idx = pairs.index("gaze←eda")
    ax.annotate('Gaze$\\leftarrow$EDA nearly\nunchanged (Neg vs Pos)',
                xy=(gze_idx, max(neg_means[gze_idx], pos_means[gze_idx], reg_means[gze_idx]) + 0.005),
                xytext=(gze_idx, max(neg_means + pos_means + reg_means) * 1.16),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=9, fontweight='bold', ha='center')

    plt.tight_layout()

    out_dir = ANALYSIS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure5_attention_weights.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure 5 saved to {out_path}")

if __name__ == "__main__":
    main()
