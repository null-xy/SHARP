#!/usr/bin/env python3
"""
sharp.stats.event_synchrony

Event-level inter-personal EDA synchrony, split by SSRL class. This is the
source of the "Interpersonal EDA covariation" paragraph in the paper's
Section 3.4: whether participants' EDA moves together (rather than just
looking at the group-mean signal, as sharp.stats.eda_event_analysis does).

For each event in the canonical n=29 working subset (Autumn,
require_eda=True -- the same subset used throughout the paper), this
computes pairwise inter-personal EDA Pearson r within the pre-onset
[-30,0) and post-onset [0,30) windows, averages across available
participant pairs to get one group-synchrony score per window per event,
and tests whether synchrony differs across Positive/Negative/Regulate.

Method
  - Per-participant EDA phasic, z-scored per session (same convention as
    sharp.stats.eda_event_analysis).
  - For each event and each window (pre/post), pairwise Pearson r is computed
    for every participant pair with >= MIN_OVERLAP overlapping 1-Hz samples.
  - Event-level synchrony = mean of available pairwise r's (Fisher-z
    averaged), NOT a group-mean signal correlated with itself.
  - delta_r = post_r - pre_r (does the group synchronize more after the event?)
  - Kruskal-Wallis across the 3 SSRL classes on pre_r / post_r / delta_r.
  - Wilcoxon signed-rank pre vs post within each class (does sync itself shift?).

Outputs  processed_data/analysis/event_synchrony/
  event_synchrony.csv     per-event pre/post/delta synchrony + pair details
  synchrony_stats.csv     per-class Kruskal-Wallis / Wilcoxon / Cohen's d
  event_synchrony.png     pre vs post boxplot by class
"""
from __future__ import annotations

from functools import reduce
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..config import ROOT

PROCESSED_ROOT = ROOT / "processed_data"
OUT_DIR = PROCESSED_ROOT / "analysis" / "event_synchrony"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN = 30            # seconds per pre/post window
MIN_OVERLAP = 15    # min overlapping 1-Hz samples required to trust a pairwise r

LABEL_MAP = {
    "Positive socioemotional interaction": "Positive",
    "Negative socioemotional interaction": "Negative",
    "Regulate group emo-mo":               "Regulate",
}
CLASS_ORDER = ["Negative", "Positive", "Regulate"]
CLASS_COLOR = {"Negative": "#e74c3c", "Positive": "#2ecc71", "Regulate": "#e67e22"}


# ── data loading ─────────────────────────────────────────────────────────────

def load_group_eda_1s(group: str, vmap: pd.DataFrame, task_start_ms: float) -> pd.DataFrame:
    """Per-second, per-participant z-scored EDA phasic. Same logic as
    07_event_eda_analysis.load_group_eda_1s (duplicated to keep this script
    standalone). Returns columns: second, eda_group, eda_{p_id}..."""
    rows = vmap[vmap["group_project"] == group]
    dfs = []
    for _, row in rows.iterrows():
        eda_val = str(row.get("eda_processed_path", ""))
        if eda_val in ("", "nan", "N/A"):
            continue
        eda_path = ROOT / eda_val
        if not eda_path.exists():
            continue
        df = pd.read_csv(eda_path, usecols=["AdjustedTime", "EDA_Phasic"]).dropna()
        if df.empty:
            continue
        df["task_sec"] = (df["AdjustedTime"] - task_start_ms) / 1000.0
        df = df[(df["task_sec"] >= -14400) & (df["task_sec"] <= 14400)]
        if df.empty:
            continue
        std = df["EDA_Phasic"].std()
        df["eda_z"] = (df["EDA_Phasic"] - df["EDA_Phasic"].mean()) / std if std > 0 else 0.0
        df["second"] = df["task_sec"].round().astype(int)
        sec = df.groupby("second")["eda_z"].mean().reset_index()
        sec.columns = ["second", f"eda_{row['p_id']}"]
        dfs.append(sec)

    if not dfs:
        return pd.DataFrame()
    merged = reduce(lambda a, b: pd.merge(a, b, on="second", how="outer"), dfs)
    pid_cols = [c for c in merged.columns if c.startswith("eda_")]
    merged["eda_group"] = merged[pid_cols].mean(axis=1)
    return merged[["second", "eda_group"] + pid_cols].sort_values("second").reset_index(drop=True)


def canonical_n29_events() -> pd.DataFrame:
    """(group, start_sec, deliberation) for the n=29 Autumn/require_eda=True
    working subset used throughout the paper (sharp.dataset.load_sharp_dataset)."""
    from ..dataset import load_sharp_dataset

    ds = load_sharp_dataset(season="autumn", require_eda=True)
    rows = [
        {"group": s["group"], "start_sec": float(s["start_sec"]),
         "deliberation": s["deliberation"]}
        for s in ds._samples
    ]
    return pd.DataFrame(rows)


# ── synchrony computation ───────────────────────────────────────────────────

def window_pairwise_r(df_eda: pd.DataFrame, pid_cols: list[str],
                       lo: float, hi: float) -> tuple[float | None, list[dict]]:
    """Mean pairwise Pearson r among available participants within [lo, hi).
    Pairwise r's are Fisher-z transformed before averaging and back-transformed
    for reporting (averaging raw correlations is not distance-preserving; z-space
    averaging is the standard treatment). Returns (mean_r or None, list of
    per-pair dicts for auditing)."""
    win = df_eda[(df_eda["second"] >= lo) & (df_eda["second"] < hi)]
    pair_rows = []
    r_values = []
    for a, b in combinations(pid_cols, 2):
        joint = win[[a, b]].dropna()
        if len(joint) < MIN_OVERLAP:
            continue
        r = float(np.corrcoef(joint[a], joint[b])[0, 1])
        if np.isnan(r):
            continue
        pair_rows.append({"pair": f"{a[4:]}-{b[4:]}", "r": round(r, 4), "n": len(joint)})
        r_values.append(r)
    if not r_values:
        return None, pair_rows
    z_values = np.arctanh(np.clip(r_values, -0.999999, 0.999999))
    mean_r = float(np.tanh(np.mean(z_values)))
    return mean_r, pair_rows


def cohen_d(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[~np.isnan(delta)]
    if len(delta) < 2 or delta.std(ddof=1) == 0:
        return float("nan")
    return float(delta.mean() / delta.std(ddof=1))


def wilcoxon_safe(pre: np.ndarray, post: np.ndarray) -> float:
    delta = post - pre
    if len(delta) < 5 or np.all(delta == 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(pre, post, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


# ── stats + plotting ────────────────────────────────────────────────────────

def compute_stats(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["pre_r", "post_r", "delta_r"]:
        groups = [events.loc[events["label"] == c, metric].dropna().values for c in CLASS_ORDER]
        if all(len(g) >= 2 for g in groups):
            h, p = stats.kruskal(*groups)
        else:
            h, p = float("nan"), float("nan")
        row = {"metric": metric, "kw_H": round(h, 4) if not np.isnan(h) else None,
               "kw_p": round(p, 4) if not np.isnan(p) else None}
        for c, g in zip(CLASS_ORDER, groups):
            row[f"{c}_n"] = len(g)
            row[f"{c}_mean"] = round(float(np.mean(g)), 4) if len(g) else None
            row[f"{c}_median"] = round(float(np.median(g)), 4) if len(g) else None
        rows.append(row)

        # post-hoc pairwise Mann-Whitney
        for c1, c2 in combinations(CLASS_ORDER, 2):
            g1 = events.loc[events["label"] == c1, metric].dropna().values
            g2 = events.loc[events["label"] == c2, metric].dropna().values
            if len(g1) >= 3 and len(g2) >= 3:
                mw_p = float(stats.mannwhitneyu(g1, g2, alternative="two-sided").pvalue)
            else:
                mw_p = float("nan")
            rows.append({"metric": f"{metric}__{c1}_vs_{c2}", "kw_H": None,
                         "kw_p": round(mw_p, 4) if not np.isnan(mw_p) else None})

    # within-class pre vs post shift (does the group synchronize more after onset?)
    for c in CLASS_ORDER:
        sub = events[events["label"] == c].dropna(subset=["pre_r", "post_r"])
        pre, post = sub["pre_r"].values, sub["post_r"].values
        d = cohen_d(post - pre)
        p = wilcoxon_safe(pre, post)
        rows.append({
            "metric": f"prepost_shift__{c}", "kw_H": None, "kw_p": None,
            f"{c}_n": len(sub),
            f"{c}_mean": round(float(np.mean(post - pre)), 4) if len(sub) else None,
            "cohen_d": round(d, 4) if not np.isnan(d) else None,
            "wilcoxon_p": round(p, 4) if not np.isnan(p) else None,
        })

    return pd.DataFrame(rows)


def plot_synchrony(events: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)
    for ax, cls in zip(axes, CLASS_ORDER):
        sub = events[events["label"] == cls].dropna(subset=["pre_r", "post_r"])
        pre, post = sub["pre_r"].values, sub["post_r"].values
        color = CLASS_COLOR[cls]

        bp = ax.boxplot([pre, post], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
        bp["boxes"][0].set_facecolor(color + "55")
        bp["boxes"][1].set_facecolor(color)
        for p_val, q_val in zip(pre, post):
            ax.plot([1, 2], [p_val, q_val], color="gray", alpha=0.35, linewidth=0.9)

        d = cohen_d(post - pre)
        p = wilcoxon_safe(pre, post)
        star = "*" if (not np.isnan(p) and p < 0.05) else ""
        ax.set_title(f"{cls} (n={len(sub)})\nd={d:.2f}, p={p:.3f}{star}",
                     fontsize=10, fontweight="bold", color=color)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre\n[-30,0)", "Post\n[0,30)"], fontsize=9)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Inter-personal EDA synchrony\n(mean pairwise Pearson r)", fontsize=10)
    plt.suptitle("Event-Level Group EDA Synchrony — Pre vs Post SSRL Onset",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    vmap_path = PROCESSED_ROOT / "eda" / "video_timing_map.csv"
    sync_path = PROCESSED_ROOT / "eda" / "sync_timing_map.csv"

    df_vmap = pd.read_csv(vmap_path)
    df_vmap = df_vmap[
        df_vmap["eda_processed_path"].notna() &
        (df_vmap["eda_processed_path"].astype(str) != "N/A")
    ]
    df_sync = pd.read_csv(sync_path)
    sync_lookup = {
        str(r["group_project"]).upper(): float(r["processed_video_start_unix_ms"])
        for _, r in df_sync.iterrows()
        if r.get("group_project") and r.get("processed_video_start_unix_ms")
    }

    events = canonical_n29_events()
    events["label"] = events["deliberation"].map(LABEL_MAP)
    print(f"Canonical working subset: n={len(events)}  groups={sorted(events['group'].unique())}")
    print(events["label"].value_counts().to_string())

    eda_cache: dict[str, pd.DataFrame] = {}
    pid_cols_cache: dict[str, list[str]] = {}
    for gp in sorted(events["group"].unique()):
        task_ms = sync_lookup.get(gp.upper())
        if task_ms is None:
            print(f"  [Skip group] {gp}: no sync timing")
            continue
        df_eda = load_group_eda_1s(gp.upper(), df_vmap, task_ms)
        if df_eda.empty:
            print(f"  [Skip group] {gp}: no EDA loaded")
            continue
        pid_cols = [c for c in df_eda.columns if c.startswith("eda_") and c != "eda_group"]
        eda_cache[gp] = df_eda
        pid_cols_cache[gp] = pid_cols
        print(f"  {gp}: participants={[c[4:] for c in pid_cols]}, "
              f"{len(df_eda)} s ({df_eda['second'].min()}..{df_eda['second'].max()})")

    N_REQUIRED_PAIRS = 3  # full triad: 3 participants, all 3 pairwise r's valid in both windows

    rows = []
    dropped_incomplete = 0
    for _, ev in events.iterrows():
        gp = ev["group"]
        df_eda = eda_cache.get(gp)
        pid_cols = pid_cols_cache.get(gp, [])
        if df_eda is None or len(pid_cols) < 2:
            dropped_incomplete += 1
            continue

        origin = ev["start_sec"]
        pre_r, pre_pairs = window_pairwise_r(df_eda, pid_cols, origin - WIN, origin)
        post_r, post_pairs = window_pairwise_r(df_eda, pid_cols, origin, origin + WIN)

        full_triad = (
            len(pid_cols) == N_REQUIRED_PAIRS
            and len(pre_pairs) == N_REQUIRED_PAIRS
            and len(post_pairs) == N_REQUIRED_PAIRS
        )
        if not full_triad:
            dropped_incomplete += 1

        rows.append({
            "group": gp, "start_sec": round(origin, 2), "label": ev["label"],
            "n_participants": len(pid_cols),
            "n_pairs_pre": len(pre_pairs), "n_pairs_post": len(post_pairs),
            "full_triad": full_triad,
            "pre_r": pre_r, "post_r": post_r,
            "delta_r": (post_r - pre_r) if (pre_r is not None and post_r is not None) else None,
            "pre_pairs": str(pre_pairs), "post_pairs": str(post_pairs),
        })

    out_all = pd.DataFrame(rows)
    out_all.to_csv(OUT_DIR / "event_synchrony.csv", index=False)
    print(f"Wrote: {OUT_DIR / 'event_synchrony.csv'}  (all {len(out_all)} events, incl. incomplete triads)")

    # Statistics and plots use ONLY events where all 3 participants and all 3
    # pairwise correlations are available in BOTH windows. A "mean of whatever
    # pairs happened to be valid" is not a consistent construct across events
    # (1-pair vs 3-pair averages measure different things), so partial-triad
    # events are excluded rather than silently averaged in.
    out = out_all[out_all["full_triad"]].copy()
    print(f"Full-triad events (3/3 participants, 3/3 pairs in pre AND post): "
          f"{len(out)} / {len(out_all)} ({dropped_incomplete} dropped)")
    if len(out) < len(out_all):
        print(out_all.loc[~out_all["full_triad"], ["group", "start_sec", "label", "n_participants",
                                                     "n_pairs_pre", "n_pairs_post"]].to_string(index=False))

    df_stats = compute_stats(out)
    df_stats.to_csv(OUT_DIR / "synchrony_stats.csv", index=False)
    print(f"Wrote: {OUT_DIR / 'synchrony_stats.csv'}")
    print("\n=== Kruskal-Wallis across classes ===")
    print(df_stats[df_stats["metric"].isin(["pre_r", "post_r", "delta_r"])]
          [["metric", "kw_H", "kw_p"] + [f"{c}_mean" for c in CLASS_ORDER]].to_string(index=False))

    delta_row = df_stats[df_stats["metric"] == "delta_r"].iloc[0]
    if delta_row["kw_p"] is None or delta_row["kw_p"] >= 0.05:
        print(f"\n[NOTE] delta_r (post-pre change) Kruskal-Wallis p={delta_row['kw_p']} is NOT significant.")
        print("       'post_r differs but pre_r does not' is NOT sufficient evidence that the")
        print("       divergence is class-specific / event-triggered -- that claim requires a")
        print("       significant delta_r (or interaction) test, which this is not. Report the")
        print("       post_r class difference on its own merits; do not infer an interaction.")

    print("\n=== Pre vs Post shift within class ===")
    print(df_stats[df_stats["metric"].str.startswith("prepost_shift__")]
          .to_string(index=False))

    plot_synchrony(out, OUT_DIR / "event_synchrony.png")


if __name__ == "__main__":
    main()
