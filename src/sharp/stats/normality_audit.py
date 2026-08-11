#!/usr/bin/env python3
"""
sharp.stats.normality_audit

Runs the pre/post EDA and facial-arousal delta significance tests used in
the paper's "Multimodal Characterization of SSRL Events" section (Speaking-
participant EDA change / Group-mean EDA changes / Physiological vs facial
affect profile), applying a single explicit, per-class/per-metric test-
selection rule instead of picking a test ad hoc:

  A parametric effect size (Cohen's d_z) should only be reported alongside
  a parametric test, and a nonparametric test needs a nonparametric effect
  size -- so the choice between them is made by first checking normality:

    Shapiro-Wilk test on the delta array
        p > 0.05 (fail to reject normality) -> paired t-test + Cohen's d_z
        p <= 0.05 (reject normality)        -> Wilcoxon signed-rank +
                                                matched-pairs rank-biserial r

Uses the same n=29 whitelist (require_eda=True, Autumn) as
sharp.stats.discordance.

Outputs: processed_data/analysis/stat_audit/normality_stat_audit.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import ANALYSIS_DIR

OUT_DIR = ANALYSIS_DIR / "stat_audit"
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

METRICS = {
    "speaker_eda": "eda_speaker_delta",
    "group_eda": "eda_group_delta",
    "speaker_facial_arousal": "emo_speaker_aro_delta",
}


def _get_n29_whitelist() -> set:
    from ..dataset import load_sharp_dataset
    ds = load_sharp_dataset(individual=False, season="autumn", require_eda=True)
    return {(s["group"].upper(), int(round(float(s["start_sec"])))) for s in ds._samples}


def load_autumn_ssrl() -> pd.DataFrame:
    df = pd.read_csv(FEAT_CSV)
    mask = df["deliberation"].isin(SSRL_LABELS) & df["group"].str.startswith("D", na=False)
    df = df[mask].copy()
    df["label"] = df["deliberation"].map(LABEL_MAP)
    whitelist = _get_n29_whitelist()
    df["_key"] = list(zip(df["group"].str.upper(), df["start_sec"].round().astype(int)))
    df = df[df["_key"].apply(lambda k: k in whitelist)].drop(columns=["_key"])
    return df.reset_index(drop=True)


def matched_rank_biserial(delta: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from Wilcoxon signed ranks
    (Kerby 2014 simple-difference formula): r = (W+ - W-) / (W+ + W-)."""
    d = delta[delta != 0]
    ranks = stats.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def audit_one(delta: np.ndarray) -> dict:
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    row = {"n": n, "mean": delta.mean() if n else np.nan}
    if n < 3:
        row["shapiro_p"] = np.nan
        row["test_used"] = "insufficient_n"
        return row

    sw_stat, sw_p = stats.shapiro(delta)
    row["shapiro_W"] = sw_stat
    row["shapiro_p"] = sw_p
    normal = sw_p > 0.05
    row["normal"] = normal

    if normal:
        t_stat, t_p = stats.ttest_1samp(delta, 0)
        row["test_used"] = "paired_t"
        row["statistic"] = t_stat
        row["p"] = t_p
        row["effect_size_name"] = "cohen_d_z"
        row["effect_size"] = delta.mean() / delta.std(ddof=1)
    else:
        if n < 5 or (delta == 0).all():
            row["test_used"] = "insufficient_n_for_wilcoxon"
            return row
        w_res = stats.wilcoxon(delta, alternative="two-sided")
        row["test_used"] = "wilcoxon"
        row["statistic"] = w_res.statistic
        row["p"] = w_res.pvalue
        row["effect_size_name"] = "rank_biserial_r"
        row["effect_size"] = matched_rank_biserial(delta)
    return row


def audit_pairwise_correlation() -> list[dict]:
    """Pre-to-post change in mean pairwise EDA correlation (Script 40 output,
    already filtered to the same n=29 whitelist)."""
    sync_csv = ANALYSIS_DIR / "event_synchrony" / "event_synchrony.csv"
    df = pd.read_csv(sync_csv)
    rows = []
    for label in CLASS_ORDER:
        sub = df[df["label"] == label]
        res = audit_one(sub["delta_r"].dropna().values.astype(float))
        res["metric"] = "pairwise_eda_corr_delta"
        res["class"] = label
        rows.append(res)
    return rows


def main():
    df = load_autumn_ssrl()
    rows = []
    for metric_key, col in METRICS.items():
        for label in CLASS_ORDER:
            sub = df[df["label"] == label]
            res = audit_one(sub[col].values.astype(float))
            res["metric"] = metric_key
            res["class"] = label
            rows.append(res)

    rows.extend(audit_pairwise_correlation())

    out = pd.DataFrame(rows)
    cols = ["metric", "class", "n", "mean", "shapiro_W", "shapiro_p", "normal",
            "test_used", "statistic", "p", "effect_size_name", "effect_size"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT_DIR / "normality_stat_audit.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'normality_stat_audit.csv'}")


if __name__ == "__main__":
    main()
