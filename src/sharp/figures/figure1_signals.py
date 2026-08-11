#!/usr/bin/env python3
"""
sharp.figures.figure1_signals

Paper Figure 1: representative Negative SSRL event showing cross-modal
conflict. Three panels, all raw/minimally processed:
  (a) EmoNet facial affect at 30fps (valence/arousal scores per video frame)
  (b) Tobii gaze fixation raster (event bars per participant, coloured by target)
  (c) Shimmer GSR3+ raw skin conductance in µS at 128 Hz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from ..dataset import load_sharp_dataset, WIN_SEC, LABELS_3CLASS
from ..config import GAZE_PATH, ROOT, ANALYSIS_DIR

WINDOW_SEC = 30   # seconds before/after event

# Flat/candy palette (matches crossmodal_conflict_evidence.png's CLASS_COLORS)
BLUE   = "#3498db"   # Peter River
ORANGE = "#f39c12"   # Orange
RED    = "#e74c3c"   # Alizarin
GREEN  = "#2ecc71"   # Emerald
YELLOW = "#f1c40f"   # Sun Flower
PINK   = "#fd79a8"   # Flamingo
GRAY   = "#95a5a6"   # Concrete

PEER_COLORS = {
    "Yellow": YELLOW,
    "Green":  GREEN,
    "Pink":   PINK,
}
TARGET_COLORS = {
    "Laptop": BLUE,
    "Pink":   PINK,
    "Green":  GREEN,
    "Yellow": YELLOW,
    "Other":  "#dfe6e9",
}


# ── event selection ───────────────────────────────────────────────────────────

def _conflict_score(s):
    eda_delta = s["eda"][0].numpy()[30:].mean() - s["eda"][0].numpy()[:30].mean()
    aro_delta = s["emonet"][1].numpy()[30:].mean() - s["emonet"][1].numpy()[:30].mean()
    return eda_delta - aro_delta


# ── raw data loaders ──────────────────────────────────────────────────────────

def load_raw_eda(eda_raw_path, t0_ms, event_sec, window=WINDOW_SEC):
    """
    Load truly raw Shimmer GSR3+ skin conductance (µS) for ±window seconds.
    Shimmer CSV format: row 0 = 'sep=\t', row 1 = column names, row 2 = units, row 3+ = data.
    """
    path = ROOT / eda_raw_path
    # read header row to get column names, skip sep= and units rows
    df = pd.read_csv(path, sep="\t", skiprows=[0, 2], header=0, low_memory=False)
    # find timestamp and EDA columns dynamically (device code varies per participant)
    ts_col  = next(c for c in df.columns if "TimestampSync_Unix" in c)
    eda_col = next(c for c in df.columns if "GSR_Skin_Conductance_CAL" in c)
    df[ts_col]  = pd.to_numeric(df[ts_col],  errors="coerce")
    df[eda_col] = pd.to_numeric(df[eda_col], errors="coerce")
    df = df.dropna(subset=[ts_col, eda_col])
    df["task_sec"] = (df[ts_col] - t0_ms) / 1000.0
    seg = df[(df["task_sec"] >= event_sec - window) &
             (df["task_sec"] <  event_sec + window)]
    t_rel = seg["task_sec"].values - event_sec
    return t_rel, seg[eda_col].values


def load_emonet_raw(date_str, pid, event_sec, window=WINDOW_SEC):
    """
    Load EmoNet frame-level output (30 fps) for ±window seconds around event_sec.
    Returns (t_rel, valence, arousal); both set to NaN where valence == -2 (no face detected).
    """
    path = (ROOT / "raw_data/education/2022 02 combined emotion 1s"
            / "Oct_Combined/autumn" / f"{date_str}_trigger"
            / f"{date_str}_{pid}.xlsx")
    df = pd.read_excel(path, header=None,
                       names=["emotion", "valence", "arousal", "_", "marker"])
    frame0  = int((event_sec - window) * 30)
    n_frames = int(window * 2 * 30)
    seg = df.iloc[frame0: frame0 + n_frames].reset_index(drop=True)
    t_rel = (seg.index.values.astype(float) / 30.0) - window
    val = seg["valence"].values.astype(float)
    aro = seg["arousal"].values.astype(float)
    val[val <= -1.5] = np.nan   # -2.0 == no detection sentinel
    aro[aro <= -1.5] = np.nan
    return t_rel, val, aro


def load_gaze_events(gaze_df, group, event_sec, window=WINDOW_SEC):
    """Return gaze fixation events clipped to the ±window window, in relative time."""
    sub = gaze_df[
        (gaze_df["group_project"] == group) &
        (gaze_df["start_seconds"]  < event_sec + window) &
        (gaze_df["end_seconds"]    > event_sec - window)
    ].copy()
    sub["rel_start"] = (sub["start_seconds"] - event_sec).clip(-window, window)
    sub["rel_end"]   = (sub["end_seconds"]   - event_sec).clip(-window, window)
    return sub


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset...")
    ds   = load_sharp_dataset(season="autumn", require_eda=True)
    gaze = pd.read_csv(GAZE_PATH)
    gaze["group_project"] = gaze["group_project"].str.upper()

    # D3G1 t=1108: Positive SSRL event. Window centred on the SSRL annotation
    # start. P3/Yellow has 100% face detection over the full 60s window (no
    # gaps) and clearly visible arousal/valence dynamics (selected by
    # scanning all events x participants for full detection + highest
    # arousal std; see scan_fig1_candidates.py).
    FIXED_GROUP = "D3G1"
    FIXED_T     = 1108.26

    sample     = next(s for s in ds._samples if s["group"] == FIXED_GROUP
                      and abs(float(s["start_sec"]) - FIXED_T) < 1.0)
    group      = sample["group"]
    event_sec  = float(sample["start_sec"])
    event_label = LABELS_3CLASS[int(sample["label"])].split()[0]  # Negative/Positive/Regulate
    print(f"Selected group={group}  t={event_sec:.0f}s  label={event_label}")

    # lookup tables
    vmap = pd.read_csv(ROOT / "processed_data/eda/video_timing_map.csv")
    vmap["group_project"] = vmap["group_project"].str.upper()
    sync = pd.read_csv(ROOT / "processed_data/eda/sync_timing_map.csv")
    sync["group_project"] = sync["group_project"].str.upper()

    pmap    = vmap[vmap["group_project"] == group].copy()
    t0_ms   = float(sync[sync["group_project"] == group]["processed_video_start_unix_ms"].iloc[0])
    date_str = pmap["date_str"].iloc[0]
    pid_list = sorted(pmap["p_id"].unique())
    print(f"date={date_str}  t0_ms={t0_ms:.0f}  participants={pid_list}")

    # ── load raw EDA per participant ──────────────────────────────────────────
    eda_traces = {}
    for _, row in pmap.iterrows():
        epath = row.get("eda_raw_path")
        if pd.isna(epath):
            continue
        pid       = row["p_id"]
        peer_color = str(row.get("participant_identification", "")).strip()
        t_rel, eda_raw = load_raw_eda(epath, t0_ms, event_sec)
        eda_traces[pid] = (peer_color, t_rel, eda_raw)
        print(f"  EDA {pid} ({peer_color}): {len(t_rel)} samples, "
              f"range [{eda_raw.min():.2f}, {eda_raw.max():.2f}] µS")

    # ── load EmoNet 30fps per participant ─────────────────────────────────────
    emo_traces = {}
    for _, row in pmap.iterrows():
        pid = row["p_id"]
        peer_color = str(row.get("participant_identification", "")).strip()
        try:
            t_rel, val, aro = load_emonet_raw(date_str, pid, event_sec)
            det_rate = (~np.isnan(val)).mean()
            emo_traces[pid] = (peer_color, t_rel, val, aro)
            print(f"  EmoNet {pid} ({peer_color}): detection rate {det_rate:.0%}")
        except Exception as e:
            print(f"  EmoNet {pid}: skipped ({e})")

    # ── pick single best participant to show across all panels ───────────────
    def _pid_conflict_score(pid):
        if pid not in eda_traces or pid not in emo_traces:
            return -999
        _, t_eda, eda_raw = eda_traces[pid]
        _, _, val, aro = emo_traces[pid]
        det = (~np.isnan(aro)).mean()
        if det < 0.5:
            return -999
        mid = len(t_eda) // 2
        eda_rise = eda_raw[mid:].mean() - eda_raw[:mid].mean()
        return eda_rise

    focal_pid = "P3"  # P3/Yellow: 100% face detection, no gaps
    focal_peer, _, _ = eda_traces[focal_pid]
    print(f"Focal participant: {focal_pid} ({focal_peer})")

    # ── load gaze events ──────────────────────────────────────────────────────
    gaze_events = load_gaze_events(gaze, group, event_sec)
    gaze_focal  = gaze_events[gaze_events["p_id"] == focal_pid]

    # ── figure layout ─────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})
    fig, axes = plt.subplots(4, 1, figsize=(9, 10.5),
                             gridspec_kw={"hspace": 0.62, "height_ratios": [1.3, 1.3, 0.9, 2]})

    SSRL_KW  = dict(color=RED,      linestyle=":",  linewidth=1.8, alpha=0.8, zorder=5)
    GRID_KW  = dict(alpha=0.22, linewidth=0.6)
    TITLE_KW = dict(loc="left", fontsize=10, fontweight="bold", pad=4)
    XLIM     = (-WINDOW_SEC, WINDOW_SEC)

    _, t_emo, val, aro = emo_traces[focal_pid]
    det_rate = (~np.isnan(val)).mean()
    gap_mask = np.isnan(val)

    def _shade_gaps(ax):
        if not gap_mask.any():
            return
        in_gap, gap_start = False, t_emo[0]
        for i, is_gap in enumerate(gap_mask):
            if is_gap and not in_gap:
                gap_start, in_gap = t_emo[i], True
            elif not is_gap and in_gap:
                ax.axvspan(gap_start, t_emo[i], color="#dfe6e9",
                           alpha=0.4, linewidth=0, zorder=0)
                in_gap = False
        if in_gap:
            ax.axvspan(gap_start, t_emo[-1], color="#dfe6e9",
                       alpha=0.4, linewidth=0, zorder=0)

    # ─ (a) EmoNet: Valence at 30fps ───────────────────────────────────────────
    ax = axes[0]
    _shade_gaps(ax)
    ax.plot(t_emo, np.ma.array(val, mask=np.isnan(val)),
            linewidth=1.2, color=BLUE, label="Valence")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(0, **SSRL_KW)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Valence")
    ax.set_title("Facial Valence", **TITLE_KW)
    ssrl_xf = 0.5 + 0.01
    ax.text(ssrl_xf, 0.97, f"↓ {event_label} SSRL event", color=RED, fontsize=8, ha="left", va="top", transform=ax.transAxes)
    ax.grid(**GRID_KW)
    ax.set_xlim(*XLIM)

    # ─ (b) EmoNet: Arousal at 30fps ───────────────────────────────────────────
    ax = axes[1]
    _shade_gaps(ax)
    ax.plot(t_emo, np.ma.array(aro, mask=np.isnan(aro)),
            linewidth=1.2, color=ORANGE, label="Arousal")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(0, **SSRL_KW)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Arousal")
    ax.set_title("Facial Arousal", **TITLE_KW)
    ax.grid(**GRID_KW)
    ax.set_xlim(*XLIM)

    # ─ (c) Gaze raster — focal participant only (single row) ─────────────────
    ax = axes[2]
    bar_h = 0.28
    for _, evt in gaze_focal.iterrows():
        target = str(evt.get("Gazed_entity", "Other")).strip()
        col = TARGET_COLORS.get(target, TARGET_COLORS["Other"])
        w   = evt["rel_end"] - evt["rel_start"]
        if w > 0:
            ax.barh(0, w, left=evt["rel_start"], height=bar_h,
                    color=col, alpha=0.88, linewidth=0)

    ax.axvline(0, **SSRL_KW)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.9)
    ax.set_ylabel("Fixation\ntarget")
    legend_patches = [mpatches.Patch(color=TARGET_COLORS[t], label=t)
                      for t in ["Laptop", "Pink", "Green", "Yellow"]]
    ax.legend(handles=legend_patches, loc="upper right",
              frameon=False, fontsize=7.5, ncol=4)
    ax.set_title("Gaze State", **TITLE_KW)
    ax.grid(axis="x", **GRID_KW)
    ax.set_xlim(*XLIM)

    # ─ (d) Raw EDA at 128 Hz — focal participant ─────────────────────────────
    ax = axes[3]
    _, t_eda, eda_raw = eda_traces[focal_pid]
    plot_col = PEER_COLORS.get(focal_peer, GRAY)
    ax.plot(t_eda, eda_raw, linewidth=1.0, color=plot_col)
    ax.axvline(0, **SSRL_KW)
    ax.set_ylabel("Skin Conductance (μS)")
    ax.set_xlabel("Time relative to SSRL event onset (s)")
    ax.set_title("Raw Electrodermal Activity (EDA)", **TITLE_KW)
    ax.grid(**GRID_KW)
    ax.set_xlim(*XLIM)

    out_dir = ANALYSIS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure1.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
