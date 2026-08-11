#!/usr/bin/env python3
"""
sharp.stats.eda_event_analysis

Event-level EDA analysis around SSRL-annotated Socio-emotional regulation events.
All window boundaries are configurable via command-line arguments.

Usage examples
  # Default (symmetric ±30 s, physiologically naive)
  python -m sharp.stats.eda_event_analysis

  # Physiologically correct: tight baseline + SCR peak window
  python -m sharp.stats.eda_event_analysis --post-start 1 --post-end 10

  # Short pre-baseline, wider post
  python -m sharp.stats.eda_event_analysis --pre-start -15 --pre-end 0 --post-start 1 --post-end 20

  # Compare multiple configs back-to-back
  for cfg in "0 30" "1 10" "2 15"; do
      read ps pe <<< $cfg
      python -m sharp.stats.eda_event_analysis --post-start $ps --post-end $pe
  done

Window parameters
  --pre-start   INT   Start of pre-window relative to event onset (default: -30)
  --pre-end     INT   End   of pre-window relative to event onset (default:   0)
  --post-start  INT   Start of post-window relative to event onset (default:  0)
  --post-end    INT   End   of post-window relative to event onset (default: 30)
  --tc-window   INT   Half-window for ERP time-course plot (default: 60)
  --smooth      INT   Rolling average seconds for time-course (default: 5)
  --min-cov     FLOAT Minimum EDA coverage fraction to keep event (default: 0.6)

Outputs  processed_data/analysis/event_eda/<window_tag>/
  event_eda_epochs.csv      per-event pre/post means, delta
  event_eda_stats.csv       per-type Wilcoxon + Cohen's d + Bonferroni
  event_eda_timecourse.png  ERP-style mean±SEM
  event_eda_prepost.png     pre vs post boxplot

Design notes
  - Unit of analysis: unique (Group, Start Time, Deliberation) → 49 interaction moments
  - Deduplication: multiple speaker rows at same timestamp → keep first
  - EDA: per-participant z-score, 1-s mean, group average
  - Timestamp outlier filter: |task_sec| > 14400 (D3G1 P1 corruption)
  - Non-independence caveat: 49 events from 5 sessions; report as nested
"""
from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..config import ROOT, ANNO_PATH
from ..utils import hms_to_sec, build_color_pid_lookup

PROCESSED_ROOT = ROOT / "processed_data"


DELIB_ORDER = [
    "Positive socioemotional interaction",
    "Regulate group emo-mo",
    "Negative socioemotional interaction",
    "Educate each other",
    "Define the problem",
    "Evaluate the result",
    "Generate options",
    "Attempt ideas",
    "Reach agreement",
    "Monitor environmental context",
    "Monitor the result",
    "Evaluate options",
]
DELIB_SHORT = {
    "Positive socioemotional interaction": "Pos-SE",
    "Regulate group emo-mo":               "Reg-SE",
    "Negative socioemotional interaction": "Neg-SE",
    "Educate each other":                  "Edu",
    "Define the problem":                  "Def",
    "Evaluate the result":                 "Eval",
    "Generate options":                    "Gen",
    "Attempt ideas":                       "Attempt",
    "Reach agreement":                     "Agree",
    "Monitor environmental context":       "Mon-Env",
    "Monitor the result":                  "Mon-Res",
    "Evaluate options":                    "Eval-Opt",
}
DELIB_COLOR = {
    "Positive socioemotional interaction": "#2ecc71",
    "Regulate group emo-mo":               "#e67e22",
    "Negative socioemotional interaction": "#e74c3c",
    "Educate each other":                  "#3498db",
    "Define the problem":                  "#9b59b6",
    "Evaluate the result":                 "#7f8c8d",
    "Generate options":                    "#1abc9c",
    "Attempt ideas":                       "#f39c12",
    "Reach agreement":                     "#27ae60",
    "Monitor environmental context":       "#bdc3c7",
    "Monitor the result":                  "#bdc3c7",
    "Evaluate options":                    "#95a5a6",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-level EDA analysis around SSRL events")
    p.add_argument("--pre-start",  type=int,   default=-30,
                   help="Pre-window start (s, relative to event onset, default: -30)")
    p.add_argument("--pre-end",    type=int,   default=0,
                   help="Pre-window end   (s, relative to event onset, default:   0)")
    p.add_argument("--post-start", type=int,   default=0,
                   help="Post-window start (s, relative to event onset, default:  0)")
    p.add_argument("--post-end",   type=int,   default=30,
                   help="Post-window end   (s, relative to event onset, default: 30)")
    p.add_argument("--tc-window",  type=int,   default=60,
                   help="Half-window for ERP time-course plot (default: 60)")
    p.add_argument("--smooth",     type=int,   default=5,
                   help="Rolling average seconds for time-course (default: 5)")
    p.add_argument("--min-cov",    type=float, default=0.6,
                   help="Min EDA coverage fraction to keep event (default: 0.6)")
    return p.parse_args()


def window_tag(args: argparse.Namespace) -> str:
    return f"pre{args.pre_start}_{args.pre_end}__post{args.post_start}_{args.post_end}"


# ── helpers (hms_to_sec imported from sharp_utils) ────────────────────────────

def cohen_d(delta: np.ndarray) -> float:
    if len(delta) < 2 or delta.std(ddof=1) == 0:
        return float("nan")
    return float(delta.mean() / delta.std(ddof=1))


def wilcoxon_safe(pre: np.ndarray, post: np.ndarray) -> tuple[float, float]:
    delta = post - pre
    if len(delta) < 5 or (delta == 0).all():
        return float("nan"), float("nan")
    try:
        res = stats.wilcoxon(pre, post, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


# ── data loading (build_color_pid_lookup imported from sharp_utils) ───────────


def load_group_eda_1s(
    group: str,
    vmap: pd.DataFrame,
    task_start_ms: float,
) -> pd.DataFrame:
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
    eda_cols = [c for c in merged.columns if c.startswith("eda_")]
    merged["eda_group"] = merged[eda_cols].mean(axis=1)
    # keep individual columns too for speaker-specific extraction
    return merged[["second", "eda_group"] + eda_cols].sort_values("second").reset_index(drop=True)


# ── event extraction ──────────────────────────────────────────────────────────

def extract_event_epoch(
    event_sec: float,
    df_eda: pd.DataFrame,
    pre_start: int,
    pre_end: int,
    post_start: int,
    post_end: int,
    tc_window: int,
    min_coverage: float,
    speaker_col: str | None = None,
) -> tuple[float, float, float, float, pd.Series] | None:
    """
    Extract EDA for one event using both group-mean and (optionally) speaker-specific EDA.

    Pre  window : [event_sec + pre_start,  event_sec + pre_end)
    Post window : [event_sec + post_start, event_sec + post_end)
    TC   window : [event_sec - tc_window,  event_sec + tc_window)

    Returns (group_pre, group_post, speaker_pre, speaker_post, timecourse) or None.
    speaker_pre/post are NaN when speaker_col is None or missing from df_eda.
    timecourse uses group-mean EDA.
    """
    origin = int(round(event_sec))

    pre_lo,  pre_hi  = origin + pre_start,  origin + pre_end
    post_lo, post_hi = origin + post_start, origin + post_end

    pre_win  = df_eda[(df_eda["second"] >= pre_lo)  & (df_eda["second"] < pre_hi)]
    post_win = df_eda[(df_eda["second"] >= post_lo) & (df_eda["second"] < post_hi)]

    if len(pre_win)  < (pre_hi  - pre_lo)  * min_coverage:
        return None
    if len(post_win) < (post_hi - post_lo) * min_coverage:
        return None

    group_pre  = float(pre_win["eda_group"].mean())
    group_post = float(post_win["eda_group"].mean())

    # speaker-specific EDA (NaN if column absent or all-NaN)
    spk_pre = spk_post = float("nan")
    if speaker_col and speaker_col in df_eda.columns:
        spk_pre  = float(pre_win[speaker_col].mean())
        spk_post = float(post_win[speaker_col].mean())

    # time-course (group mean)
    tc_lo, tc_hi = origin - tc_window, origin + tc_window
    tc_win = df_eda[(df_eda["second"] >= tc_lo) & (df_eda["second"] < tc_hi)].copy()
    tc_win["rel"] = tc_win["second"] - origin
    tc = tc_win.set_index("rel")["eda_group"].reindex(range(-tc_window, tc_window))

    return group_pre, group_post, spk_pre, spk_post, tc


# ── statistics ────────────────────────────────────────────────────────────────

def compute_stats(epochs: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []

    def _row(label: str, sub: pd.DataFrame) -> dict:
        pre   = sub["eda_pre"].values
        post  = sub["eda_post"].values
        delta = post - pre
        stat, p = wilcoxon_safe(pre, post)

        # speaker-specific
        spk_valid = sub.dropna(subset=["spk_eda_pre", "spk_eda_post"])
        spk_pre   = spk_valid["spk_eda_pre"].values
        spk_post  = spk_valid["spk_eda_post"].values
        spk_delta = spk_post - spk_pre
        spk_stat, spk_p = wilcoxon_safe(spk_pre, spk_post)

        return {
            "deliberation":      label,
            "n_events":          len(sub),
            "n_sessions":        sub["group"].nunique(),
            "pre_window":        f"[{args.pre_start}, {args.pre_end})",
            "post_window":       f"[{args.post_start}, {args.post_end})",
            # group EDA
            "mean_pre":          round(float(np.nanmean(pre)),   4),
            "mean_post":         round(float(np.nanmean(post)),  4),
            "mean_delta":        round(float(np.nanmean(delta)), 4),
            "sd_delta":          round(float(np.nanstd(delta, ddof=1)), 4),
            "cohen_d":           round(cohen_d(delta), 4),
            "p_value":           round(p, 4)    if not np.isnan(p)    else None,
            # speaker-specific EDA
            "n_spk":             len(spk_valid),
            "spk_mean_pre":      round(float(np.nanmean(spk_pre)),   4) if len(spk_valid) else None,
            "spk_mean_post":     round(float(np.nanmean(spk_post)),  4) if len(spk_valid) else None,
            "spk_mean_delta":    round(float(np.nanmean(spk_delta)), 4) if len(spk_valid) else None,
            "spk_cohen_d":       round(cohen_d(spk_delta), 4)           if len(spk_valid) >= 2 else None,
            "spk_p_value":       round(spk_p, 4) if not np.isnan(spk_p) else None,
        }

    for delib in DELIB_ORDER:
        sub = epochs[epochs["deliberation"] == delib]
        if len(sub) > 0:
            rows.append(_row(delib, sub))

    rows.append(_row("OVERALL", epochs))

    df = pd.DataFrame(rows)
    n_tests = (df["deliberation"] != "OVERALL").sum()
    df["p_bonferroni"] = df["p_value"].apply(
        lambda p: round(min(float(p) * n_tests, 1.0), 4) if p is not None else None
    )
    return df


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_timecourse(
    epochs: pd.DataFrame,
    tc_store: dict[str, list],
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    types_present = [d for d in DELIB_ORDER
                     if d in tc_store and len(tc_store[d]) >= 3]
    if not types_present:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(types_present) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    x = np.arange(-args.tc_window, args.tc_window)

    def _smooth(arr):
        return pd.Series(arr).rolling(args.smooth, center=True, min_periods=1).mean().values

    for i, delib in enumerate(types_present):
        ax = axes[i]
        mat = np.array([tc.reindex(x).values for tc in tc_store[delib]], dtype=float)
        mat_s = np.array([_smooth(row) for row in mat])
        mean = np.nanmean(mat_s, axis=0)
        n_valid = np.sum(~np.isnan(mat_s), axis=0)
        sem  = np.nanstd(mat_s, axis=0, ddof=1) / np.sqrt(np.where(n_valid > 0, n_valid, 1))

        color = DELIB_COLOR.get(delib, "#888888")
        n = len(tc_store[delib])
        ax.plot(x, mean, color=color, linewidth=2)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.25, color=color)

        # mark pre/post windows
        ax.axvspan(args.pre_start,  args.pre_end,  color="steelblue", alpha=0.08,
                   label=f"Pre [{args.pre_start},{args.pre_end})")
        ax.axvspan(args.post_start, args.post_end, color="tomato",    alpha=0.08,
                   label=f"Post [{args.post_start},{args.post_end})")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        short = DELIB_SHORT.get(delib, delib[:10])
        ax.set_title(f"{short} (n={n})", fontsize=9, fontweight="bold", color=color)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlim(-args.tc_window, args.tc_window)

    # shared legend on first subplot
    axes[0].legend(fontsize=7, loc="upper left")
    for j in range(len(types_present), len(axes)):
        axes[j].set_visible(False)

    fig.supxlabel("Seconds relative to event onset", fontsize=11)
    fig.supylabel("Group-mean EDA phasic z-score", fontsize=11)
    tag = window_tag(args)
    plt.suptitle(
        f"EDA Time-course Around SSRL Events  |  pre [{args.pre_start},{args.pre_end})  "
        f"post [{args.post_start},{args.post_end})  |  mean±SEM, {args.smooth}-s smooth",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_prepost(
    epochs: pd.DataFrame,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    types_present = [d for d in DELIB_ORDER
                     if d in epochs["deliberation"].values
                     and (epochs["deliberation"] == d).sum() >= 3]
    if not types_present:
        return

    n = len(types_present)
    fig, axes = plt.subplots(1, n, figsize=(max(3 * n, 8), 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, delib in zip(axes, types_present):
        sub   = epochs[epochs["deliberation"] == delib]
        pre   = sub["eda_pre"].values
        post  = sub["eda_post"].values
        color = DELIB_COLOR.get(delib, "#888888")

        bp = ax.boxplot([pre, post], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        bp["boxes"][0].set_facecolor(color + "55")
        bp["boxes"][1].set_facecolor(color)

        for p_val, q_val in zip(pre, post):
            ax.plot([1, 2], [p_val, q_val], color="gray", alpha=0.35, linewidth=0.9)

        delta = post - pre
        d = cohen_d(delta)
        d_str = f"d={d:.2f}" if not np.isnan(d) else ""
        ax.set_title(f"{DELIB_SHORT.get(delib, delib[:8])}\n(n={len(sub)}, {d_str})",
                     fontsize=8, fontweight="bold", color=color)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Pre\n[{args.pre_start},{args.pre_end})",
             f"Post\n[{args.post_start},{args.post_end})"],
            fontsize=8,
        )
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Group-mean EDA phasic z-score", fontsize=10)
    plt.suptitle(
        f"Pre vs Post EDA — SSRL Deliberation Types\n"
        f"pre [{args.pre_start},{args.pre_end})  post [{args.post_start},{args.post_end})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # validate
    if args.pre_start >= args.pre_end:
        raise ValueError(f"pre_start ({args.pre_start}) must be < pre_end ({args.pre_end})")
    if args.post_start >= args.post_end:
        raise ValueError(f"post_start ({args.post_start}) must be < post_end ({args.post_end})")

    tag     = window_tag(args)
    out_dir = PROCESSED_ROOT / "analysis" / "event_eda" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Window config : pre [{args.pre_start}, {args.pre_end})  "
          f"post [{args.post_start}, {args.post_end})  "
          f"tc ±{args.tc_window} s  smooth {args.smooth} s")
    print(f"Output dir    : {out_dir.relative_to(ROOT)}\n")

    vmap_path = PROCESSED_ROOT / "eda" / "video_timing_map.csv"
    sync_path = PROCESSED_ROOT / "eda" / "sync_timing_map.csv"

    df_vmap = pd.read_csv(vmap_path)
    df_vmap = df_vmap[
        df_vmap["eda_processed_path"].notna() &
        (df_vmap["eda_processed_path"].astype(str) != "N/A")
    ]

    # Derive from data — includes all groups (Autumn + Spring) with valid EDA
    EDA_GROUPS = sorted(df_vmap["group_project"].str.upper().dropna().unique())

    df_sync = pd.read_csv(sync_path)
    sync_lookup = {
        str(r["group_project"]).upper(): float(r["processed_video_start_unix_ms"])
        for _, r in df_sync.iterrows()
        if r.get("group_project") and r.get("processed_video_start_unix_ms")
    }

    # Build per-session color → p_id lookup (varies across sessions!)
    color_pid_lookup = build_color_pid_lookup(df_vmap)
    print("Color→PID per session:")
    for gp in EDA_GROUPS:
        print(f"  {gp}: {color_pid_lookup.get(gp, {})}")
    print()

    # Load & filter annotations
    df_anno = pd.read_excel(ANNO_PATH)
    df_anno["task_sec_start"] = df_anno["Start Time"].apply(hms_to_sec)
    df_anno["task_sec_end"]   = df_anno["End Time"].apply(hms_to_sec)
    df_anno = df_anno[df_anno["Group"].isin(EDA_GROUPS)].copy()

    # Deduplicate: one row per unique interaction moment × deliberation type
    df_anno = df_anno.drop_duplicates(
        subset=["Group", "Start Time", "Deliberation"]
    ).reset_index(drop=True)
    print(f"Unique interaction moments: {len(df_anno)}")
    print(df_anno["Deliberation"].value_counts().to_string())
    print()

    # Load EDA once per group
    eda_cache: dict[str, pd.DataFrame] = {}
    for gp in EDA_GROUPS:
        task_ms = sync_lookup.get(gp)
        if task_ms is None:
            continue
        df_eda = load_group_eda_1s(gp, df_vmap, task_ms)
        if not df_eda.empty:
            eda_cache[gp] = df_eda
            print(f"  EDA {gp}: {len(df_eda)} s  "
                  f"({df_eda['second'].min()}–{df_eda['second'].max()})")

    # Extract epochs
    epoch_rows = []
    tc_store: dict[str, list] = {}
    skipped = 0

    for _, ev in df_anno.iterrows():
        gp    = str(ev["Group"])
        delib = str(ev["Deliberation"])
        esec  = float(ev["task_sec_start"])

        df_eda = eda_cache.get(gp)
        if df_eda is None:
            skipped += 1
            continue

        # resolve speaker-specific EDA column using per-session color→pid mapping
        speaker_label = str(ev.get("Speaker", "")).strip()
        pid = color_pid_lookup.get(gp, {}).get(speaker_label)
        speaker_col = f"eda_{pid}" if pid else None

        result = extract_event_epoch(
            esec, df_eda,
            args.pre_start, args.pre_end,
            args.post_start, args.post_end,
            args.tc_window, args.min_cov,
            speaker_col=speaker_col,
        )
        if result is None:
            skipped += 1
            continue

        group_pre, group_post, spk_pre, spk_post, tc = result
        epoch_rows.append({
            "group":            gp,
            "event_sec":        round(esec, 2),
            "deliberation":     delib,
            "speaker":          speaker_label,
            "speaker_pid":      pid or "",
            "dialogue":         str(ev.get("Dialogue", ""))[:80],
            # group-mean EDA
            "eda_pre":          round(group_pre,              4),
            "eda_post":         round(group_post,             4),
            "eda_delta":        round(group_post - group_pre, 4),
            # speaker-specific EDA
            "spk_eda_pre":      round(spk_pre,  4) if not np.isnan(spk_pre)  else None,
            "spk_eda_post":     round(spk_post, 4) if not np.isnan(spk_post) else None,
            "spk_eda_delta":    round(spk_post - spk_pre, 4)
                                if not (np.isnan(spk_pre) or np.isnan(spk_post)) else None,
            "pre_window":       f"[{args.pre_start},{args.pre_end})",
            "post_window":      f"[{args.post_start},{args.post_end})",
        })
        tc_store.setdefault(delib, []).append(tc)

    epochs = pd.DataFrame(epoch_rows)
    print(f"\nEpochs extracted: {len(epochs)}  (skipped: {skipped})")

    epochs.to_csv(out_dir / "event_eda_epochs.csv", index=False)
    print(f"Wrote: event_eda_epochs.csv")

    df_stats = compute_stats(epochs, args)
    df_stats.to_csv(out_dir / "event_eda_stats.csv", index=False)
    print(f"Wrote: event_eda_stats.csv")

    print("\n=== Group-mean EDA ===")
    cols = ["deliberation", "n_events", "mean_delta", "cohen_d", "p_value", "p_bonferroni"]
    print(df_stats[[c for c in cols if c in df_stats.columns]].to_string(index=False))

    print("\n=== Speaker-specific EDA ===")
    spk_cols = ["deliberation", "n_spk", "spk_mean_delta", "spk_cohen_d", "spk_p_value"]
    print(df_stats[[c for c in spk_cols if c in df_stats.columns]].to_string(index=False))

    plot_timecourse(epochs, tc_store, args, out_dir / "event_eda_timecourse.png")
    plot_prepost(epochs, args,               out_dir / "event_eda_prepost.png")

    print(f"\nDone → {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
