#!/usr/bin/env python3
"""
sharp.features.build_feature_matrix

Build event-level multimodal feature matrix.
One row per unique SSRL annotation event (Group + StartTime + Deliberation).

Modalities integrated:
  EDA     : per-participant z-scored phasic EDA pre/post mean delta
             (Autumn Room 1: D1G1–D6G1 only)
  SCR     : per-participant SCR peak count + mean amplitude delta
             HR (PPGtoHR) delta — derived from processed_data/eda/scr/
  EmoNet  : per-participant valence/arousal pre/post mean delta
             (all 10 groups with processed emotion CSVs)
  Gaze    : per-participant gaze duration toward Laptop / Peers in window
             (D1G1–D6G1 + SD1G1–SD3G1)

Window (configurable via CLI):
  pre  = [PRE_START,  PRE_END)  relative to event start_sec
  post = [POST_START, POST_END) relative to event start_sec

Output: processed_data/analysis/feature_matrix.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, ANNO_PATH
from ..utils import hms_to_sec, build_color_pid_lookup, load_vmap

PROCESSED_ROOT = ROOT / "processed_data"

EMO_DIR  = PROCESSED_ROOT / "education" / "emotion"
GAZE_CSV = PROCESSED_ROOT / "gaze" / "gaze_events.csv"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-level multimodal feature matrix")
    p.add_argument("--pre-start",  type=int, default=-30)
    p.add_argument("--pre-end",    type=int, default=0)
    p.add_argument("--post-start", type=int, default=0)
    p.add_argument("--post-end",   type=int, default=30)
    return p.parse_args()


# ── helpers (hms_to_sec, build_color_pid_lookup imported from sharp_utils) ────


def window_mean(s: pd.Series, lo: int, hi: int) -> float:
    v = s[(s.index >= lo) & (s.index < hi)]
    return float(v.mean()) if len(v) > 0 else float("nan")


# ── EDA loading ───────────────────────────────────────────────────────────────

def load_all_eda(
    vmap: pd.DataFrame, sync_lookup: dict[str, float]
) -> dict[str, dict[str, pd.Series]]:
    """
    Returns {group: {p_id: Series(index=second_int, values=z_eda)}}.
    Only groups with both sync timing and processed EDA are included.
    """
    result: dict[str, dict[str, pd.Series]] = {}
    for gp, task_ms in sync_lookup.items():
        rows = vmap[
            (vmap["group_project"] == gp) &
            vmap["eda_processed_path"].notna() &
            (vmap["eda_processed_path"].astype(str) != "N/A")
        ]
        if rows.empty:
            continue
        pid_series: dict[str, pd.Series] = {}
        for _, row in rows.iterrows():
            path = ROOT / str(row["eda_processed_path"])
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=["AdjustedTime", "EDA_Phasic"]).dropna()
            df["task_sec"] = (df["AdjustedTime"] - task_ms) / 1000.0
            df = df[(df["task_sec"] >= -14400) & (df["task_sec"] <= 14400)]
            if df.empty:
                continue
            std = df["EDA_Phasic"].std()
            df["z"] = (df["EDA_Phasic"] - df["EDA_Phasic"].mean()) / std if std > 0 else 0.0
            df["second"] = df["task_sec"].round().astype(int)
            pid_series[str(row["p_id"])] = df.groupby("second")["z"].mean()
        if pid_series:
            result[gp] = pid_series
    return result


# ── SCR loading ───────────────────────────────────────────────────────────────

SCR_DIR = PROCESSED_ROOT / "eda" / "scr"


def load_all_scr(
    vmap: pd.DataFrame, sync_lookup: dict[str, float]
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Returns {group: {p_id: DataFrame(index=second_int,
                                     cols=[scr_count, scr_amp_mean, hr_mean])}}.
    Derives scr path from eda_processed_path by swapping /filtered/ → /scr/.
    """
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for gp, task_ms in sync_lookup.items():
        rows = vmap[
            (vmap["group_project"] == gp) &
            vmap["eda_processed_path"].notna() &
            (vmap["eda_processed_path"].astype(str) != "N/A")
        ]
        if rows.empty:
            continue
        pid_frames: dict[str, pd.DataFrame] = {}
        for _, row in rows.iterrows():
            filtered_path = str(row["eda_processed_path"])
            scr_rel = filtered_path.replace("/filtered/", "/scr/")
            path = ROOT / scr_rel
            if not path.exists():
                continue
            df = pd.read_csv(
                path,
                usecols=["AdjustedTime", "SCR_Peaks", "SCR_Height", "PPGtoHR"]
            ).dropna(subset=["AdjustedTime"])
            df["task_sec"] = (df["AdjustedTime"] - task_ms) / 1000.0
            df = df[(df["task_sec"] >= -14400) & (df["task_sec"] <= 14400)]
            if df.empty:
                continue
            df["second"] = df["task_sec"].round().astype(int)
            # per-second aggregates
            agg = df.groupby("second").agg(
                scr_count=("SCR_Peaks", "sum"),
                scr_amp_mean=("SCR_Height", lambda x: x[df.loc[x.index, "SCR_Peaks"] > 0].mean()),
                hr_mean=("PPGtoHR", lambda x: x[x > 0].mean()),
            )
            pid_frames[str(row["p_id"])] = agg
        if pid_frames:
            result[gp] = pid_frames
    return result


def _window_scr(
    agg: pd.DataFrame, lo: int, hi: int
) -> tuple[float, float, float]:
    """Return (peak_count, mean_amplitude, mean_hr) for seconds in [lo, hi)."""
    sub = agg[(agg.index >= lo) & (agg.index < hi)]
    if sub.empty:
        return float("nan"), float("nan"), float("nan")
    count   = float(sub["scr_count"].sum())
    amp_vals = sub["scr_amp_mean"].dropna()
    amp      = float(amp_vals.mean()) if len(amp_vals) > 0 else float("nan")
    hr_vals  = sub["hr_mean"].dropna()
    hr       = float(hr_vals.mean()) if len(hr_vals) > 0 else float("nan")
    return count, amp, hr


def extract_scr_features(
    ev_start: float,
    pid_scr: dict[str, pd.DataFrame],
    speaker_pid: str,
    pre_lo: int, pre_hi: int,
    post_lo: int, post_hi: int,
) -> dict:
    origin = int(round(ev_start))
    row: dict = {}
    count_deltas, amp_deltas, hr_deltas = [], [], []

    for pid in ["P1", "P2", "P3"]:
        agg = pid_scr.get(pid)
        if agg is None:
            for suffix in ("count_pre", "count_post", "count_delta",
                           "amp_pre", "amp_post", "amp_delta",
                           "hr_pre", "hr_post", "hr_delta"):
                row[f"scr_{pid.lower()}_{suffix}"] = float("nan")
        else:
            c_pre,  a_pre,  h_pre  = _window_scr(agg, origin + pre_lo,  origin + pre_hi)
            c_post, a_post, h_post = _window_scr(agg, origin + post_lo, origin + post_hi)
            c_delta = c_post - c_pre if not (np.isnan(c_pre) or np.isnan(c_post)) else float("nan")
            a_delta = a_post - a_pre if not (np.isnan(a_pre) or np.isnan(a_post)) else float("nan")
            h_delta = h_post - h_pre if not (np.isnan(h_pre) or np.isnan(h_post)) else float("nan")
            row[f"scr_{pid.lower()}_count_pre"]   = round(c_pre,  2)
            row[f"scr_{pid.lower()}_count_post"]  = round(c_post, 2)
            row[f"scr_{pid.lower()}_count_delta"] = round(c_delta, 2)
            row[f"scr_{pid.lower()}_amp_pre"]     = round(a_pre,  4) if not np.isnan(a_pre)  else float("nan")
            row[f"scr_{pid.lower()}_amp_post"]    = round(a_post, 4) if not np.isnan(a_post) else float("nan")
            row[f"scr_{pid.lower()}_amp_delta"]   = round(a_delta,4) if not np.isnan(a_delta) else float("nan")
            row[f"scr_{pid.lower()}_hr_pre"]      = round(h_pre,  2) if not np.isnan(h_pre)  else float("nan")
            row[f"scr_{pid.lower()}_hr_post"]     = round(h_post, 2) if not np.isnan(h_post) else float("nan")
            row[f"scr_{pid.lower()}_hr_delta"]    = round(h_delta,2) if not np.isnan(h_delta) else float("nan")
            if not np.isnan(c_delta): count_deltas.append(c_delta)
            if not np.isnan(a_delta): amp_deltas.append(a_delta)
            if not np.isnan(h_delta): hr_deltas.append(h_delta)

        if pid == speaker_pid:
            row["scr_speaker_count_delta"] = row[f"scr_{pid.lower()}_count_delta"]
            row["scr_speaker_amp_delta"]   = row[f"scr_{pid.lower()}_amp_delta"]
            row["scr_speaker_hr_delta"]    = row[f"scr_{pid.lower()}_hr_delta"]

    row["scr_group_count_delta"] = round(float(np.nanmean(count_deltas)), 2) if count_deltas else float("nan")
    row["scr_group_amp_delta"]   = round(float(np.nanmean(amp_deltas)),   4) if amp_deltas  else float("nan")
    row["scr_group_hr_delta"]    = round(float(np.nanmean(hr_deltas)),    2) if hr_deltas   else float("nan")

    for k in ("scr_speaker_count_delta", "scr_speaker_amp_delta", "scr_speaker_hr_delta"):
        if k not in row:
            row[k] = float("nan")
    return row


# ── EmoNet loading ────────────────────────────────────────────────────────────

def load_all_emonet(
    sync_lookup: dict[str, float]
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Returns {group: {p_id: DataFrame(index=second, cols=[valence,arousal])}}.
    EmoNet `second` column is already task_sec (integer seconds from video start).
    Frames with valence <= -1.5 are face-not-detected; set to NaN.
    """
    # Build group→date_str mapping from sync lookup (need date_str for filename)
    sync = pd.read_csv(PROCESSED_ROOT / "eda" / "sync_timing_map.csv")
    sync = sync[sync["group_project"].notna() & (sync["group_project"] != "Group/project")]
    group_to_date: dict[str, str] = {}
    for _, r in sync.iterrows():
        g = str(r["group_project"]).upper()
        d = str(r["date_str"])
        if g not in group_to_date:
            group_to_date[g] = d

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for gp in sync_lookup:
        date_str = group_to_date.get(gp)
        if not date_str:
            continue
        pid_dfs: dict[str, pd.DataFrame] = {}
        for pid in ["P1", "P2", "P3"]:
            path = EMO_DIR / f"{date_str}_{pid}_1s.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=["second", "valence", "arousal"])
            # Mark face-not-detected rows as NaN
            df.loc[df["valence"] <= -1.5, ["valence", "arousal"]] = float("nan")
            df = df.set_index("second")
            pid_dfs[pid] = df
        if pid_dfs:
            result[gp] = pid_dfs
    return result


# ── Gaze loading ──────────────────────────────────────────────────────────────

def load_gaze() -> pd.DataFrame:
    """
    Load gaze events. start_seconds / end_seconds are task_sec.
    Peer entities are Pink / Green / Yellow (any non-speaker color).
    """
    gaze = pd.read_csv(GAZE_CSV, usecols=[
        "group_project", "p_id",
        "Gazed_entity", "start_seconds", "end_seconds"
    ])
    gaze["group_project"] = gaze["group_project"].str.upper()
    return gaze


def gaze_in_window(
    gaze: pd.DataFrame,
    group: str,
    pid: str,
    win_lo: float,
    win_hi: float,
    entity_set: set[str],
) -> float:
    """Total seconds that participant gazed at any entity in entity_set within [win_lo, win_hi]."""
    sub = gaze[(gaze["group_project"] == group) & (gaze["p_id"] == pid) &
               gaze["Gazed_entity"].isin(entity_set)]
    overlap = (sub["end_seconds"].clip(upper=win_hi) -
               sub["start_seconds"].clip(lower=win_lo)).clip(lower=0)
    return float(overlap.sum())


# ── feature extraction per event ──────────────────────────────────────────────

def extract_eda_features(
    ev_start: float,
    pid_eda: dict[str, pd.Series],
    speaker_pid: str,
    pre_lo: int, pre_hi: int,
    post_lo: int, post_hi: int,
) -> dict:
    origin = int(round(ev_start))
    row: dict = {}
    deltas = []
    for pid in ["P1", "P2", "P3"]:
        s = pid_eda.get(pid)
        if s is None:
            row[f"eda_{pid.lower()}_pre"]   = float("nan")
            row[f"eda_{pid.lower()}_post"]  = float("nan")
            row[f"eda_{pid.lower()}_delta"] = float("nan")
        else:
            pre  = window_mean(s, origin + pre_lo,  origin + pre_hi)
            post = window_mean(s, origin + post_lo, origin + post_hi)
            delta = post - pre if not (np.isnan(pre) or np.isnan(post)) else float("nan")
            row[f"eda_{pid.lower()}_pre"]   = round(pre, 4)
            row[f"eda_{pid.lower()}_post"]  = round(post, 4)
            row[f"eda_{pid.lower()}_delta"] = round(delta, 4)
            if not np.isnan(delta):
                deltas.append(delta)
        # speaker flag
        if pid == speaker_pid:
            row["eda_speaker_delta"] = row[f"eda_{pid.lower()}_delta"]
    row["eda_group_delta"] = round(float(np.nanmean(deltas)), 4) if deltas else float("nan")
    if "eda_speaker_delta" not in row:
        row["eda_speaker_delta"] = float("nan")
    return row


def extract_emo_features(
    ev_start: float,
    pid_emo: dict[str, pd.DataFrame],
    speaker_pid: str,
    pre_lo: int, pre_hi: int,
    post_lo: int, post_hi: int,
) -> dict:
    origin = int(round(ev_start))
    row: dict = {}
    val_deltas, aro_deltas = [], []
    for pid in ["P1", "P2", "P3"]:
        df = pid_emo.get(pid)
        for feat in ["val", "aro"]:
            col = "valence" if feat == "val" else "arousal"
            if df is None:
                row[f"emo_{pid.lower()}_{feat}_pre"]   = float("nan")
                row[f"emo_{pid.lower()}_{feat}_post"]  = float("nan")
                row[f"emo_{pid.lower()}_{feat}_delta"] = float("nan")
            else:
                s = df[col]
                pre  = window_mean(s, origin + pre_lo,  origin + pre_hi)
                post = window_mean(s, origin + post_lo, origin + post_hi)
                delta = post - pre if not (np.isnan(pre) or np.isnan(post)) else float("nan")
                row[f"emo_{pid.lower()}_{feat}_pre"]   = round(pre, 4)
                row[f"emo_{pid.lower()}_{feat}_post"]  = round(post, 4)
                row[f"emo_{pid.lower()}_{feat}_delta"] = round(delta, 4)
                if not np.isnan(delta):
                    (val_deltas if feat == "val" else aro_deltas).append(delta)
        if pid == speaker_pid:
            row["emo_speaker_val_delta"] = row[f"emo_{pid.lower()}_val_delta"]
            row["emo_speaker_aro_delta"] = row[f"emo_{pid.lower()}_aro_delta"]
    row["emo_group_val_delta"] = round(float(np.nanmean(val_deltas)), 4) if val_deltas else float("nan")
    row["emo_group_aro_delta"] = round(float(np.nanmean(aro_deltas)), 4) if aro_deltas else float("nan")
    if "emo_speaker_val_delta" not in row:
        row["emo_speaker_val_delta"] = float("nan")
        row["emo_speaker_aro_delta"] = float("nan")
    return row


PEER_ENTITIES = {"Pink", "Green", "Yellow"}
LAPTOP_ENTITY  = {"Laptop"}


def extract_gaze_features(
    ev_start: float,
    gaze: pd.DataFrame,
    group: str,
    pre_lo: int, pre_hi: int,
    post_lo: int, post_hi: int,
) -> dict:
    pre_lo_s  = ev_start + pre_lo
    pre_hi_s  = ev_start + pre_hi
    post_lo_s = ev_start + post_lo
    post_hi_s = ev_start + post_hi
    pre_dur   = pre_hi - pre_lo    # seconds in window
    post_dur  = post_hi - post_lo

    row: dict = {}
    for pid in ["P1", "P2", "P3"]:
        for phase, lo_s, hi_s, dur in [
            ("pre",  pre_lo_s,  pre_hi_s,  pre_dur),
            ("post", post_lo_s, post_hi_s, post_dur),
        ]:
            laptop = gaze_in_window(gaze, group, pid, lo_s, hi_s, LAPTOP_ENTITY)
            peer   = gaze_in_window(gaze, group, pid, lo_s, hi_s, PEER_ENTITIES)
            row[f"gaze_{pid.lower()}_{phase}_laptop_s"]    = round(laptop, 2)
            row[f"gaze_{pid.lower()}_{phase}_peer_s"]      = round(peer, 2)
            row[f"gaze_{pid.lower()}_{phase}_laptop_frac"] = round(laptop / dur, 3) if dur > 0 else float("nan")
            row[f"gaze_{pid.lower()}_{phase}_peer_frac"]   = round(peer / dur, 3) if dur > 0 else float("nan")
    return row


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    pre_lo, pre_hi   = args.pre_start, args.pre_end
    post_lo, post_hi = args.post_start, args.post_end
    print(f"Window: pre [{pre_lo},{pre_hi})  post [{post_lo},{post_hi})")

    # ── Load infrastructure ───────────────────────────────────────────────────
    vmap = pd.read_csv(PROCESSED_ROOT / "eda" / "video_timing_map.csv")
    sync = pd.read_csv(PROCESSED_ROOT / "eda" / "sync_timing_map.csv")
    sync = sync[sync["group_project"].notna() & (sync["group_project"] != "Group/project")]
    sync_lookup: dict[str, float] = {
        str(r["group_project"]).upper(): float(r["processed_video_start_unix_ms"])
        for _, r in sync.iterrows()
        if pd.notna(r.get("processed_video_start_unix_ms"))
    }
    color_pid = build_color_pid_lookup(vmap)

    # ── Load modality data ────────────────────────────────────────────────────
    print("Loading EDA...", end=" ", flush=True)
    eda_all = load_all_eda(vmap, sync_lookup)
    print(f"{len(eda_all)} groups")

    print("Loading SCR...", end=" ", flush=True)
    scr_all = load_all_scr(vmap, sync_lookup)
    print(f"{len(scr_all)} groups")

    print("Loading EmoNet...", end=" ", flush=True)
    emo_all = load_all_emonet(sync_lookup)
    print(f"{len(emo_all)} groups")

    print("Loading Gaze...", end=" ", flush=True)
    gaze = load_gaze()
    gaze_groups = set(gaze["group_project"].unique())
    print(f"{len(gaze_groups)} groups")

    # ── Load + deduplicate annotations ───────────────────────────────────────
    anno = pd.read_excel(ANNO_PATH)
    anno["start_sec"] = anno["Start Time"].apply(hms_to_sec)
    anno["end_sec"]   = anno["End Time"].apply(hms_to_sec)
    anno["duration"]  = anno["end_sec"] - anno["start_sec"]
    anno["speaker_pid"] = anno.apply(
        lambda r: color_pid.get(str(r["Group"]).upper(), {}).get(
            str(r["Speaker"]).strip(), ""), axis=1
    )

    events = (anno
              .drop_duplicates(subset=["Group", "Start Time", "Deliberation"])
              .reset_index(drop=True))
    print(f"\nAnnotation events (after dedup): {len(events)}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    feature_rows = []
    for _, ev in events.iterrows():
        gp          = str(ev["Group"]).upper()
        start_sec   = float(ev["start_sec"])
        speaker_pid = str(ev["speaker_pid"])

        row: dict = {
            "group":        gp,
            "start_sec":    round(start_sec, 2),
            "end_sec":      round(float(ev["end_sec"]), 2),
            "duration_s":   round(float(ev["duration"]), 2),
            "deliberation": ev["Deliberation"],
            "speaker":      ev["Speaker"],
            "speaker_pid":  speaker_pid,
            "dialogue":     str(ev.get("Dialogue", ""))[:80],
            # pre/post window config
            "pre_window":   f"[{pre_lo},{pre_hi})",
            "post_window":  f"[{post_lo},{post_hi})",
        }

        # EDA (Autumn Room 1 only)
        pid_eda = eda_all.get(gp, {})
        row.update(extract_eda_features(
            start_sec, pid_eda, speaker_pid,
            pre_lo, pre_hi, post_lo, post_hi
        ))

        # SCR peaks + HR
        pid_scr = scr_all.get(gp, {})
        row.update(extract_scr_features(
            start_sec, pid_scr, speaker_pid,
            pre_lo, pre_hi, post_lo, post_hi
        ))

        # EmoNet
        pid_emo = emo_all.get(gp, {})
        row.update(extract_emo_features(
            start_sec, pid_emo, speaker_pid,
            pre_lo, pre_hi, post_lo, post_hi
        ))

        # Gaze
        if gp in gaze_groups:
            row.update(extract_gaze_features(
                start_sec, gaze, gp,
                pre_lo, pre_hi, post_lo, post_hi
            ))
        else:
            # Fill gaze columns with NaN for Spring sessions without gaze
            for pid in ["p1", "p2", "p3"]:
                for phase in ["pre", "post"]:
                    for suffix in ["laptop_s", "peer_s", "laptop_frac", "peer_frac"]:
                        row[f"gaze_{pid}_{phase}_{suffix}"] = float("nan")

        feature_rows.append(row)

    df = pd.DataFrame(feature_rows)

    # ── Coverage summary ─────────────────────────────────────────────────────
    print("\n=== Coverage (non-NaN fraction per modality) ===")
    eda_cols  = [c for c in df.columns if c.startswith("eda_")]
    scr_cols  = [c for c in df.columns if c.startswith("scr_")]
    emo_cols  = [c for c in df.columns if c.startswith("emo_")]
    gaze_cols = [c for c in df.columns if c.startswith("gaze_")]
    print(f"EDA  columns ({len(eda_cols)}):  "
          f"{df[eda_cols].notna().any(axis=1).sum()}/{len(df)} events have ≥1 EDA feature")
    print(f"SCR  columns ({len(scr_cols)}):  "
          f"{df[scr_cols].notna().any(axis=1).sum()}/{len(df)} events have ≥1 SCR feature")
    print(f"EmoNet columns ({len(emo_cols)}): "
          f"{df[emo_cols].notna().any(axis=1).sum()}/{len(df)} events have ≥1 EmoNet feature")
    print(f"Gaze columns ({len(gaze_cols)}): "
          f"{df[gaze_cols].notna().any(axis=1).sum()}/{len(df)} events have ≥1 gaze feature")

    print("\n=== Deliberation label breakdown ===")
    print(df["deliberation"].value_counts(dropna=False).to_string())

    print("\n=== EDA speaker delta by Deliberation (labeled Socio-emo events) ===")
    socioemo = df[df["deliberation"].isin([
        "Positive socioemotional interaction",
        "Regulate group emo-mo",
        "Negative socioemotional interaction",
    ]) & df["eda_speaker_delta"].notna()]
    if not socioemo.empty:
        print(socioemo.groupby("deliberation")["eda_speaker_delta"].agg(["mean","count"]).to_string())

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = PROCESSED_ROOT / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_matrix.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path.relative_to(ROOT)}  ({len(df)} rows × {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
