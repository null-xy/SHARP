"""
sharp.features.gaze_transitions

Compute gaze transition features for each event window and append to feature_matrix.csv.

Features per participant (P1/P2/P3) × phase (pre/post/delta):
  gaze_{pid}_{phase}_switch_count  - total gaze-target switches
  gaze_{pid}_{phase}_lp_switch     - Laptop <-> Peer (Pink/Green/Yellow) switches only
  gaze_{pid}_{phase}_mean_dur_s    - mean single gaze event duration (seconds)

Plus speaker average (the annotated speaker's values) and group mean (P1/P2/P3).

Windows: pre = [start_sec-30, start_sec), post = [start_sec, start_sec+30)
Source: processed_data/gaze/gaze_events.csv
Coverage: 9 of 10 groups (SD4G1 has no gaze data → NaN)
"""

import pandas as pd
import numpy as np

from ..config import FM_PATH, GAZE_PATH, WINDOW_SEC
from ..utils import load_feature_matrix, load_gaze

WINDOW   = WINDOW_SEC
GAZE_CSV = str(GAZE_PATH)  # kept for reference; actual load uses load_gaze()

PEER_ENTITIES = {"Pink", "Green", "Yellow"}
LAPTOP_ENTITY = {"Laptop"}


def gaze_transition_features(gaze_pid: pd.DataFrame, win_start: float, win_end: float) -> dict:
    """Compute switch_count, lp_switch, mean_dur_s for one participant in one window."""
    mask = (gaze_pid["end_seconds"] > win_start) & (gaze_pid["start_seconds"] < win_end)
    rows = gaze_pid[mask].copy().sort_values("start_seconds").reset_index(drop=True)

    n = len(rows)
    if n == 0:
        return {"switch_count": 0, "lp_switch": 0, "mean_dur_s": float("nan")}

    # durations clipped to window
    clipped_start = rows["start_seconds"].clip(lower=win_start)
    clipped_end   = rows["end_seconds"].clip(upper=win_end)
    dur = (clipped_end - clipped_start).clip(lower=0)
    mean_dur = dur.mean() if n > 0 else float("nan")

    # consecutive gaze-target switches
    entities = rows["Gazed_entity"].tolist()
    switch_count = sum(1 for a, b in zip(entities, entities[1:]) if a != b)

    # laptop ↔ peer transitions only
    lp_switch = 0
    for a, b in zip(entities, entities[1:]):
        if a == b:
            continue
        a_is_laptop = a in LAPTOP_ENTITY
        a_is_peer   = a in PEER_ENTITIES
        b_is_laptop = b in LAPTOP_ENTITY
        b_is_peer   = b in PEER_ENTITIES
        if (a_is_laptop and b_is_peer) or (a_is_peer and b_is_laptop):
            lp_switch += 1

    return {
        "switch_count": switch_count,
        "lp_switch":    lp_switch,
        "mean_dur_s":   round(mean_dur, 3),
    }


def main():
    gaze = load_gaze()[["group_project", "p_id", "Gazed_entity", "start_seconds", "end_seconds"]]
    fm = load_feature_matrix()

    # Index gaze by (group, pid)
    gaze_idx: dict[tuple, pd.DataFrame] = {}
    for (grp, pid), df in gaze.groupby(["group_project", "p_id"]):
        gaze_idx[(grp, pid)] = df

    gaze_groups = set(gaze["group_project"].unique())

    rows_out = []
    for _, ev in fm.iterrows():
        group      = str(ev["group"]).upper()
        s          = float(ev["start_sec"])
        spk_pid    = str(ev.get("speaker_pid", "")).strip()
        has_gaze   = group in gaze_groups

        row: dict = {}
        pid_feats: dict[str, dict] = {}

        for pid in ("P1", "P2", "P3"):
            for phase, (lo, hi) in (("pre", (s - WINDOW, s)), ("post", (s, s + WINDOW))):
                if has_gaze and (group, pid) in gaze_idx:
                    f = gaze_transition_features(gaze_idx[(group, pid)], lo, hi)
                else:
                    f = {"switch_count": float("nan"), "lp_switch": float("nan"),
                         "mean_dur_s": float("nan")}
                pkey = f"{pid}_{phase}"
                pid_feats[pkey] = f
                p = pid.lower()
                for feat, val in f.items():
                    row[f"gaze_{p}_{phase}_{feat}"] = val

        # delta
        for pid in ("P1", "P2", "P3"):
            p = pid.lower()
            for feat in ("switch_count", "lp_switch", "mean_dur_s"):
                pre_val  = pid_feats[f"{pid}_pre"][feat]
                post_val = pid_feats[f"{pid}_post"][feat]
                if pd.notna(pre_val) and pd.notna(post_val):
                    row[f"gaze_{p}_delta_{feat}"] = round(post_val - pre_val, 3)
                else:
                    row[f"gaze_{p}_delta_{feat}"] = float("nan")

        # speaker mean
        for phase in ("pre", "post", "delta"):
            for feat in ("switch_count", "lp_switch", "mean_dur_s"):
                if spk_pid and f"{spk_pid}_{phase}" in pid_feats:
                    val = pid_feats[f"{spk_pid}_{phase}"][feat]
                elif spk_pid in ("P1", "P2", "P3"):
                    p = spk_pid.lower()
                    col = f"gaze_{p}_{phase}_{feat}"
                    val = row.get(col, float("nan"))
                else:
                    val = float("nan")
                row[f"gaze_speaker_{phase}_{feat}"] = val if pd.notna(val) else float("nan")

        # group mean (P1/P2/P3 average)
        for phase in ("pre", "post", "delta"):
            for feat in ("switch_count", "lp_switch", "mean_dur_s"):
                vals = []
                for pid in ("P1", "P2", "P3"):
                    p = pid.lower()
                    col = f"gaze_{p}_{phase}_{feat}"
                    v = row.get(col, float("nan"))
                    if pd.notna(v):
                        vals.append(v)
                row[f"gaze_group_{phase}_{feat}"] = round(np.mean(vals), 3) if vals else float("nan")

        rows_out.append(row)

    new_cols = pd.DataFrame(rows_out, index=fm.index)

    for c in new_cols.columns:
        if c in fm.columns:
            fm.drop(columns=[c], inplace=True)

    fm = pd.concat([fm, new_cols], axis=1)
    fm.to_csv(FM_PATH, index=False)

    print(f"Added {len(new_cols.columns)} gaze transition columns to feature_matrix ({len(fm)} events).")

    # coverage summary
    n_nonnull = fm["gaze_p1_pre_switch_count"].notna().sum()
    print(f"gaze_p1_pre_switch_count non-null: {n_nonnull}/{len(fm)}")
    print("\nswitch_count_pre distribution (P1):")
    print(fm["gaze_p1_pre_switch_count"].value_counts().sort_index().head(10).to_string())
    print("\nMean switch_count_pre by deliberation class:")
    print(fm.groupby("deliberation")["gaze_group_pre_switch_count"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
