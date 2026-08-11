#!/usr/bin/env python3
"""
sharp.stats.discordance

Computes the joint EDA-facial change pattern used in the paper's "Facial
affect profile" paragraph (Section 3.4): for each retained SSRL event,

    Q_i = 1[ dEDA_i > 0  and  dArousal_i < 0 ]

i.e. whether physiological arousal (EDA) rose while facial arousal (EmoNet)
fell over the same pre/post window -- physiological activation without a
matching change in observable facial expressivity. Q_i is defined directly
on the raw (non-standardized) deltas, so "no change" is the natural
baseline rather than each event's deviation from the sample median.

The proportion of events showing this pattern is compared across the three
SSRL classes (Negative / Positive / Regulate) using an exact
Fisher-Freeman-Halton test on the resulting 3x2 contingency table (exact
rather than an asymptotic chi-square approximation, since class sizes here
are small: n=29).

Outputs: processed_data/analysis/discordance/
  joint_pattern_summary.csv  - per-class Q_i counts/proportions + exact p-value
"""
from __future__ import annotations

from math import comb

import pandas as pd

from ..config import ANALYSIS_DIR

OUT_DIR = ANALYSIS_DIR / "discordance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_CSV = ANALYSIS_DIR / "feature_matrix.csv"

SSRL_LABELS = [
    "Negative socioemotional interaction",
    "Positive socioemotional interaction",
    "Regulate group emo-mo",
]
LABEL_MAP = {
    "Negative socioemotional interaction": "Negative",
    "Positive socioemotional interaction": "Positive",
    "Regulate group emo-mo": "Regulate",
}
CLASS_ORDER = ["Negative", "Positive", "Regulate"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _get_n29_whitelist() -> set:
    """Return (group_upper, start_sec_int) pairs for the n=29 events used
    throughout the paper's deep-learning evaluation (Section 4)."""
    from ..dataset import load_sharp_dataset
    ds = load_sharp_dataset(individual=False, season="autumn", require_eda=True)
    return {(s["group"].upper(), int(round(float(s["start_sec"])))) for s in ds._samples}


def load_autumn_ssrl() -> pd.DataFrame:
    """Load the 29 Autumn SSRL events that pass require_eda=True (same
    whitelist as the classification protocol, so this analysis and the
    benchmark results in Section 4/5 refer to the same event set)."""
    df = pd.read_csv(FEAT_CSV)
    mask = df["deliberation"].isin(SSRL_LABELS) & df["group"].str.startswith("D", na=False)
    df = df[mask].copy()
    df["label"] = df["deliberation"].map(LABEL_MAP)
    whitelist = _get_n29_whitelist()
    df["_key"] = list(zip(df["group"].str.upper(), df["start_sec"].round().astype(int)))
    df = df[df["_key"].apply(lambda k: k in whitelist)].drop(columns=["_key"])
    return df.reset_index(drop=True)


# ── Joint EDA-facial change pattern ────────────────────────────────────────────

def fisher_freeman_halton_p(yes_counts: list[int], row_totals: list[int]) -> float:
    """Exact test for an r x 2 contingency table with fixed row and column
    margins (two-sided Freeman-Halton generalization of Fisher's exact test).

    Enumerates every table with the same margins and sums the probabilities
    of tables no more likely than the observed one.
    """
    K = sum(yes_counts)
    N = sum(row_totals)

    def table_prob(ys: list[int]) -> float:
        num = 1
        for y, R in zip(ys, row_totals):
            num *= comb(R, y)
        return num / comb(N, K)

    obs_prob = table_prob(yes_counts)
    total_p = 0.0
    r0, r1, r2 = row_totals
    for y0 in range(0, min(r0, K) + 1):
        for y1 in range(0, min(r1, K - y0) + 1):
            y2 = K - y0 - y1
            if 0 <= y2 <= r2:
                p = table_prob([y0, y1, y2])
                if p <= obs_prob * (1 + 1e-9):
                    total_p += p
    return total_p


def compute_joint_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Q_i = 1[dEDA_i > 0 and dArousal_i < 0], on raw (non-standardized) deltas."""
    df = df.dropna(subset=["eda_speaker_delta", "emo_speaker_aro_delta"]).copy()
    df["Q"] = (df["eda_speaker_delta"] > 0) & (df["emo_speaker_aro_delta"] < 0)

    rows = []
    yes_counts, row_totals = [], []
    for lab in CLASS_ORDER:
        sub = df[df["label"] == lab]
        n_yes = int(sub["Q"].sum())
        n = len(sub)
        rows.append({"class": lab, "n": n, "n_joint_pattern": n_yes,
                      "proportion": n_yes / n if n else float("nan")})
        yes_counts.append(n_yes)
        row_totals.append(n)

    summary = pd.DataFrame(rows)
    summary["FFH_exact_p"] = fisher_freeman_halton_p(yes_counts, row_totals)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_autumn_ssrl()
    print(f"Autumn SSRL events: {len(df)}")
    print(df["label"].value_counts().to_dict())

    joint_summary = compute_joint_pattern(df)
    joint_summary.to_csv(OUT_DIR / "joint_pattern_summary.csv", index=False)
    print("\nJoint pattern Q_i = 1[dEDA>0 and dArousal<0]:")
    for _, row in joint_summary.iterrows():
        print(f"  {row['class']}: {int(row['n_joint_pattern'])}/{int(row['n'])} "
              f"({row['proportion']*100:.1f}%)")
    print(f"Fisher-Freeman-Halton exact p = {joint_summary['FFH_exact_p'].iloc[0]:.4f}")

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
