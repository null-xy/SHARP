#!/usr/bin/env python3
"""
sharp.dataset

PyTorch Dataset for SHARP event-level multimodal time series.

One sample = one SSRL annotation event.
Each modality is a 60-second window (pre[-30,0) + post[0,30)) at 1 Hz → 60 points.

Modalities:
  eda    : (1, 60)  or  (3, 60)  — z-scored EDA phasic, group mean or P1/P2/P3
  emonet : (2, 60)  or  (6, 60)  — [valence, arousal], group mean or P1/P2/P3×2
  gaze   : (2, 60)  or  (6, 60)  — [laptop_frac, peer_frac], group mean or P1/P2/P3×2

Labels (3-class Socio-emo deliberation):
  0 = Negative socioemotional interaction
  1 = Positive socioemotional interaction
  2 = Regulate group emo-mo

Usage
-----
from sharp.dataset import load_sharp_dataset
ds = load_sharp_dataset()
sample = ds[0]   # dict with 'eda','emonet','gaze','label','eda_valid','gaze_valid',...

# Stratified CV splits
from sharp.dataset import get_cv_splits
for fold, (train_idx, val_idx) in enumerate(get_cv_splits(ds)):
    ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import LeaveOneGroupOut

from .config import ROOT, ANNO_PATH, GAZE_PATH
from .utils import hms_to_sec, build_color_pid_lookup

PROCESSED_ROOT  = ROOT / "processed_data"
EMO_DIR         = PROCESSED_ROOT / "education" / "emotion"
GAZE_CSV        = GAZE_PATH
TEXT_SENT_DIR   = PROCESSED_ROOT / "analysis" / "text_features" / "sentiment_timeseries"

LABELS_3CLASS = [
    "Negative socioemotional interaction",
    "Positive socioemotional interaction",
    "Regulate group emo-mo",
]
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS_3CLASS)}

PEER_ENTITIES = {"Pink", "Green", "Yellow"}
WIN_SEC       = 60      # total window length in seconds
PRE_START     = -30     # window start relative to event


# ── low-level helpers (hms_to_sec, build_color_pid_lookup imported from sharp.utils) ──


# ── EDA loading ───────────────────────────────────────────────────────────────

def _load_eda_series(
    vmap: pd.DataFrame, sync_lookup: dict[str, float]
) -> dict[str, dict[str, pd.Series]]:
    """
    Returns {group: {pid: Series(index=task_sec_int, values=z_eda)}}.
    Only groups with processed EDA paths are populated.
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
            mu, sd = df["EDA_Phasic"].mean(), df["EDA_Phasic"].std()
            df["z"]      = (df["EDA_Phasic"] - mu) / sd if sd > 0 else 0.0
            df["second"] = df["task_sec"].round().astype(int)
            pid_series[str(row["p_id"])] = df.groupby("second")["z"].mean()
        if pid_series:
            result[gp] = pid_series
    return result


def _extract_eda_window(
    pid_series: dict[str, pd.Series],
    origin: int,
    individual: bool,
    lag_s: int = 0,
) -> tuple[np.ndarray, bool]:
    """
    Returns (array, valid).
      individual=False → (1, 60)  group mean
      individual=True  → (3, 60)  P1/P2/P3 rows (NaN→0 for missing participant)
    lag_s: temporal offset in seconds (positive = look at EDA later, compensating SCR delay).
    """
    secs = np.arange(origin + PRE_START + lag_s, origin + PRE_START + lag_s + WIN_SEC)
    streams = []
    for pid in ["P1", "P2", "P3"]:
        s = pid_series.get(pid)
        if s is None:
            streams.append(np.zeros(WIN_SEC))
        else:
            arr = np.array([s.get(t, np.nan) for t in secs])
            arr = np.nan_to_num(arr, nan=0.0)
            streams.append(arr)

    arr3 = np.stack(streams, axis=0)  # (3, 60)
    # Check both that a file was loaded AND that the window has real signal.
    # D6G1 / D5G1-late events load a file but all timestamps fall outside the
    # recorded range → nan_to_num fills zeros → std ≈ 0.
    valid = (any(pid_series.get(p) is not None for p in ["P1", "P2", "P3"])
             and float(arr3.std()) > 0.05)

    if individual:
        return arr3, valid
    # group mean (only average participants that exist)
    n = sum(pid_series.get(p) is not None for p in ["P1", "P2", "P3"])
    group = arr3.sum(axis=0, keepdims=True) / max(n, 1)  # (1, 60)
    return group, valid


# ── EmoNet loading ────────────────────────────────────────────────────────────

def _load_emonet_series(
    group_to_date: dict[str, str],
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Returns {group: {pid: DataFrame(index=second, cols=[valence,arousal])}}.
    valence <= -1.5 = face-not-detected → set to NaN → forward-filled then 0-filled.
    """
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for gp, date_str in group_to_date.items():
        pid_dfs: dict[str, pd.DataFrame] = {}
        for pid in ["P1", "P2", "P3"]:
            path = EMO_DIR / f"{date_str}_{pid}_1s.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=["second", "valence", "arousal"])
            df.loc[df["valence"] <= -1.5, ["valence", "arousal"]] = np.nan
            df["detected"] = df["valence"].notna().astype(float)   # 1=face detected, before ffill
            # forward-fill short gaps, then 0 for remaining NaN
            df[["valence","arousal"]] = (df[["valence","arousal"]]
                                          .ffill(limit=5)
                                          .fillna(0.0))
            df = df.set_index("second")
            pid_dfs[pid] = df
        if pid_dfs:
            result[gp] = pid_dfs
    return result


def _extract_emonet_window(
    pid_dfs: dict[str, pd.DataFrame],
    origin: int,
    individual: bool,
    return_mask: bool = False,
) -> np.ndarray:
    """
    individual=False → (2, 60)  group mean [valence, arousal]
                    → (3, 60)  + detection mask channel when return_mask=True
    individual=True  → (6, 60)  P1_val, P1_aro, P2_val, P2_aro, P3_val, P3_aro
    """
    secs = np.arange(origin + PRE_START, origin + PRE_START + WIN_SEC)
    streams_val, streams_aro, streams_det = [], [], []
    for pid in ["P1", "P2", "P3"]:
        df = pid_dfs.get(pid)
        if df is None:
            streams_val.append(np.zeros(WIN_SEC))
            streams_aro.append(np.zeros(WIN_SEC))
            streams_det.append(np.zeros(WIN_SEC))
        else:
            val = np.array([df["valence"].get(t, 0.0) for t in secs])
            aro = np.array([df["arousal"].get(t, 0.0) for t in secs])
            streams_val.append(val)
            streams_aro.append(aro)
            if return_mask and "detected" in df.columns:
                det = np.array([df["detected"].get(t, 0.0) for t in secs])
            else:
                det = np.ones(WIN_SEC)
            streams_det.append(det)

    if individual:
        # Interleave: P1_val, P1_aro, P2_val, P2_aro, P3_val, P3_aro
        rows = []
        for v, a in zip(streams_val, streams_aro):
            rows.extend([v, a])
        return np.stack(rows, axis=0)  # (6, 60)
    # Group mean
    n = sum(pid_dfs.get(p) is not None for p in ["P1","P2","P3"])
    mean_val = np.stack(streams_val).sum(axis=0) / max(n, 1)
    mean_aro = np.stack(streams_aro).sum(axis=0) / max(n, 1)
    if return_mask:
        mean_det = np.stack(streams_det).sum(axis=0) / max(n, 1)
        return np.stack([mean_val, mean_aro, mean_det], axis=0)  # (3, 60)
    return np.stack([mean_val, mean_aro], axis=0)  # (2, 60)


# ── Gaze loading ──────────────────────────────────────────────────────────────

def _load_gaze_df() -> pd.DataFrame:
    gaze = pd.read_csv(GAZE_CSV, usecols=[
        "group_project", "p_id", "Gazed_entity", "start_seconds", "end_seconds"
    ])
    gaze["group_project"] = gaze["group_project"].str.upper()
    return gaze


def _gaze_per_second(
    gaze: pd.DataFrame, group: str, pid: str, origin: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (laptop_secs, peer_secs) arrays of length WIN_SEC.
    Each value = seconds spent gazing at entity during that 1-s slot.
    Vectorised: computes all 60-bin overlaps in one numpy broadcast (N×60).
    """
    sub = gaze[(gaze["group_project"] == group) & (gaze["p_id"] == pid)]
    laptop = np.zeros(WIN_SEC)
    peer   = np.zeros(WIN_SEC)
    if sub.empty:
        return laptop, peer

    win_start = float(origin + PRE_START)
    bins = np.arange(WIN_SEC + 1, dtype=np.float64) + win_start   # (61,)

    s = sub["start_seconds"].values[:, None]   # (N, 1)
    e = sub["end_seconds"].values[:, None]     # (N, 1)
    # Overlap of each fixation event with each 1-second bin  →  (N, 60)
    ov = np.maximum(0.0, np.minimum(e, bins[1:]) - np.maximum(s, bins[:-1]))

    is_lap  = (sub["Gazed_entity"] == "Laptop").values
    is_peer = sub["Gazed_entity"].isin(PEER_ENTITIES).values

    if is_lap.any():
        laptop = ov[is_lap].sum(axis=0)
    if is_peer.any():
        peer = ov[is_peer].sum(axis=0)
    return laptop, peer


def _extract_gaze_window(
    gaze: pd.DataFrame, group: str, origin: int, individual: bool
) -> tuple[np.ndarray, bool]:
    """
    individual=False → (2, 60)  group mean [laptop_frac, peer_frac]
    individual=True  → (6, 60)  P1_laptop, P1_peer, P2_laptop, P2_peer, P3_laptop, P3_peer
    """
    pids_in_group = gaze[gaze["group_project"] == group]["p_id"].unique()
    valid = len(pids_in_group) > 0

    streams_lap, streams_peer = [], []
    for pid in ["P1", "P2", "P3"]:
        if pid in pids_in_group:
            lap, peer = _gaze_per_second(gaze, group, pid, origin)
        else:
            lap = peer = np.zeros(WIN_SEC)
        streams_lap.append(lap)
        streams_peer.append(peer)

    if individual:
        rows = []
        for l, p in zip(streams_lap, streams_peer):
            rows.extend([l, p])
        return np.stack(rows, axis=0), valid  # (6, 60)

    n = len(pids_in_group)
    mean_lap  = np.stack(streams_lap).sum(axis=0) / max(n, 1)
    mean_peer = np.stack(streams_peer).sum(axis=0) / max(n, 1)
    return np.stack([mean_lap, mean_peer], axis=0), valid  # (2, 60)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SHARPEventDataset(Dataset):
    """
    Parameters
    ----------
    individual : bool
        If True, return per-participant streams (P1/P2/P3 separately).
        If False (default), return group-mean streams.
    labels : list[str] | None
        Subset of LABELS_3CLASS to include. Default: all 3.
    load_text : bool
        If True, load per-second sentiment timeseries (1,60) from
        text_features/sentiment_timeseries/{group}_{t0}.npz.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        eda_all:    dict[str, dict[str, pd.Series]],
        emonet_all: dict[str, dict[str, pd.DataFrame]],
        gaze:       pd.DataFrame,
        individual: bool = False,
        load_text:  bool = False,
        require_eda: bool = False,
        eda_lag_s: int = 0,
        emonet_miss_mask: bool = False,
    ):
        self.events      = events.reset_index(drop=True)
        self.eda_all     = eda_all
        self.emonet_all  = emonet_all
        self.gaze        = gaze
        self.individual  = individual
        self.load_text   = load_text
        self.require_eda = require_eda
        self.eda_lag_s   = eda_lag_s
        self.emonet_miss_mask = emonet_miss_mask

        # Pre-compute all samples and cache as tensors
        self._samples: list[dict] = []
        self._build_cache()

    def _build_cache(self) -> None:
        n = len(self.events)
        print(f"Building dataset cache ({n} events"
              f"{', require_eda' if self.require_eda else ''})...", flush=True)
        for i, (_, ev) in enumerate(self.events.iterrows()):
            if i % 10 == 0 and i > 0:
                print(f"  {i}/{n} scanned, {len(self._samples)} kept...", flush=True)
            gp     = str(ev["group"]).upper()
            origin = int(round(float(ev["start_sec"])))
            label  = int(ev["label"])

            # EDA
            pid_eda = self.eda_all.get(gp, {})
            if pid_eda:
                eda_arr, eda_valid = _extract_eda_window(pid_eda, origin, self.individual, lag_s=self.eda_lag_s)
            else:
                ch = 3 if self.individual else 1
                eda_arr, eda_valid = np.zeros((ch, WIN_SEC)), False

            # Skip events without valid EDA if require_eda is set
            if self.require_eda and not eda_valid:
                continue

            # EmoNet
            pid_emo = self.emonet_all.get(gp, {})
            emonet_arr = _extract_emonet_window(pid_emo, origin, self.individual, return_mask=self.emonet_miss_mask)

            # Gaze
            gaze_arr, gaze_valid = _extract_gaze_window(
                self.gaze, gp, origin, self.individual
            )

            # Text sentiment timeseries (optional)
            text_arr  = None
            text_valid = False
            if self.load_text:
                npz_path = TEXT_SENT_DIR / f"{gp}_{origin}.npz"
                if npz_path.exists():
                    text_arr   = np.load(npz_path)["sentiment"].astype(np.float32)
                    text_valid = True
                else:
                    text_arr = np.zeros((1, WIN_SEC), dtype=np.float32)

            sample = {
                "eda":        torch.from_numpy(eda_arr).float(),
                "emonet":     torch.from_numpy(emonet_arr).float(),
                "gaze":       torch.from_numpy(gaze_arr).float(),
                "label":      torch.tensor(label, dtype=torch.long),
                "eda_valid":  eda_valid,
                "gaze_valid": gaze_valid,
                "group":      gp,
                "start_sec":  float(ev["start_sec"]),
                "deliberation": str(ev["deliberation"]),
            }
            if self.load_text:
                sample["text"]       = torch.from_numpy(text_arr).float()
                sample["text_valid"] = text_valid
            self._samples.append(sample)
        print(f"  Done. ({len(self._samples)}/{n} events cached)", flush=True)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]

    @property
    def labels(self) -> np.ndarray:
        return np.array([s["label"].item() for s in self._samples])

    @property
    def eda_shape(self) -> tuple:
        return tuple(self._samples[0]["eda"].shape)

    @property
    def emonet_shape(self) -> tuple:
        return tuple(self._samples[0]["emonet"].shape)

    @property
    def gaze_shape(self) -> tuple:
        return tuple(self._samples[0]["gaze"].shape)


# ── public API ────────────────────────────────────────────────────────────────

def load_sharp_dataset(
    individual: bool = False,
    labels: Optional[list[str]] = None,
    load_text: bool = False,
    season: Optional[str] = None,
    require_eda: bool = False,
    eda_lag_s: int = 0,
    emonet_miss_mask: bool = False,
) -> SHARPEventDataset:
    """
    Load the full SHARP event dataset.

    Parameters
    ----------
    individual : bool
        Return per-participant (3×) streams instead of group means.
    labels : list[str] | None
        Filter to a subset of LABELS_3CLASS.
    season : str | None
        "autumn" → only D*G* groups; "spring" → only SD*G* groups; None → all.
    require_eda : bool
        If True, drop events where eda_valid=False after building the cache.
        Use to enforce a clean EDA-complete subset (e.g. Autumn n=29).

    Returns
    -------
    SHARPEventDataset
    """
    if labels is None:
        labels = LABELS_3CLASS

    # Infrastructure
    vmap = pd.read_csv(PROCESSED_ROOT / "eda" / "video_timing_map.csv")
    sync = pd.read_csv(PROCESSED_ROOT / "eda" / "sync_timing_map.csv")
    sync = sync[sync["group_project"].notna() & (sync["group_project"] != "Group/project")]

    sync_lookup: dict[str, float] = {
        str(r["group_project"]).upper(): float(r["processed_video_start_unix_ms"])
        for _, r in sync.iterrows()
        if pd.notna(r.get("processed_video_start_unix_ms"))
    }
    group_to_date: dict[str, str] = {}
    for _, r in sync.iterrows():
        g = str(r["group_project"]).upper()
        if g not in group_to_date:
            group_to_date[g] = str(r["date_str"])

    color_pid = build_color_pid_lookup(vmap)

    # Annotations
    anno = pd.read_excel(ANNO_PATH)
    anno["start_sec"] = anno["Start Time"].apply(hms_to_sec)
    anno["end_sec"]   = anno["End Time"].apply(hms_to_sec)
    anno["group"]     = anno["Group"].str.upper()
    anno["speaker_pid"] = anno.apply(
        lambda r: color_pid.get(str(r["Group"]).upper(), {}).get(
            str(r["Speaker"]).strip(), ""), axis=1
    )
    anno["label"]        = anno["Deliberation"].map(LABEL_TO_INT)
    anno["deliberation"] = anno["Deliberation"]

    events = (anno[anno["Deliberation"].isin(labels)]
              .drop_duplicates(subset=["Group", "Start Time", "Deliberation"])
              .dropna(subset=["label"])
              .reset_index(drop=True))

    if season == "autumn":
        events = events[~events["group"].str.startswith("SD")].reset_index(drop=True)
    elif season == "spring":
        events = events[events["group"].str.startswith("SD")].reset_index(drop=True)
    print(f"Events: {len(events)}")
    print(events["Deliberation"].value_counts().to_string())
    print()

    # Load modality data
    print("Loading EDA...", end=" ", flush=True)
    eda_all = _load_eda_series(vmap, sync_lookup)
    print(f"{len(eda_all)} groups with EDA", flush=True)

    print("Loading EmoNet...", end=" ", flush=True)
    emonet_all = _load_emonet_series(group_to_date)
    print(f"{len(emonet_all)} groups with EmoNet", flush=True)

    print("Loading Gaze...", end=" ", flush=True)
    gaze = _load_gaze_df()
    print(f"{gaze['group_project'].nunique()} groups with Gaze", flush=True)
    print()

    if load_text:
        n_text = sum(
            1 for ev in events.itertuples()
            if (TEXT_SENT_DIR / f"{str(ev.group).upper()}_{int(round(float(ev.start_sec)))}.npz").exists()
        )
        print(f"Text sentiment timeseries: {n_text}/{len(events)} events found")
    ds = SHARPEventDataset(events, eda_all, emonet_all, gaze,
                           individual=individual, load_text=load_text,
                           require_eda=require_eda,
                           eda_lag_s=eda_lag_s,
                           emonet_miss_mask=emonet_miss_mask)
    return ds


def get_cv_splits(
    ds: SHARPEventDataset,
    n_splits: int = 5,      # unused (LOGO is group-determined), kept for API compat
    random_state: int = 42, # unused
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-One-Group-Out splits. Each fold holds out one group entirely."""
    logo   = LeaveOneGroupOut()
    groups = np.array([s["group"] for s in ds._samples])
    y      = ds.labels
    return list(logo.split(np.zeros(len(y)), y, groups))


# ── EDA mean imputation (mirrors SVM's SimpleImputer) ────────────────────────

def impute_eda_fold(ds: "SHARPEventDataset", train_idx: np.ndarray) -> dict:
    """
    Fill invalid EDA samples with the mean EDA of valid training samples.
    Operates in-place on ds._samples; returns {idx: original_tensor} for restoration.
    Mirrors SVM's SimpleImputer(strategy='mean') fitted on the training fold.
    """
    invalid = [i for i in range(len(ds)) if not ds._samples[i]["eda_valid"]]
    if not invalid:
        return {}
    valid_eda = [ds._samples[i]["eda"] for i in train_idx
                 if ds._samples[i]["eda_valid"]]
    mean_eda = (torch.stack(valid_eda).mean(0) if valid_eda
                else torch.zeros_like(ds._samples[0]["eda"]))
    saved = {i: ds._samples[i]["eda"].clone() for i in invalid}
    for i in invalid:
        ds._samples[i]["eda"] = mean_eda
    return saved


def restore_eda(ds: "SHARPEventDataset", saved: dict) -> None:
    """Restore EDA tensors after imputation (call after each fold)."""
    for i, orig in saved.items():
        ds._samples[i]["eda"] = orig


# ── quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = load_sharp_dataset(individual=False)

    print(f"\nDataset size : {len(ds)}")
    print(f"EDA shape    : {ds.eda_shape}")
    print(f"EmoNet shape : {ds.emonet_shape}")
    print(f"Gaze shape   : {ds.gaze_shape}")

    # Check first few samples
    for i in range(min(3, len(ds))):
        s = ds[i]
        print(f"\n[{i}] {s['group']}  t={s['start_sec']:.0f}s  "
              f"label={s['label'].item()}  ({s['deliberation'][:30]})")
        print(f"  eda    range: [{s['eda'].min():.3f}, {s['eda'].max():.3f}]  "
              f"valid={s['eda_valid']}")
        print(f"  emonet range: [{s['emonet'].min():.3f}, {s['emonet'].max():.3f}]")
        print(f"  gaze   range: [{s['gaze'].min():.3f}, {s['gaze'].max():.3f}]  "
              f"valid={s['gaze_valid']}")

    # CV splits
    splits = get_cv_splits(ds)
    print(f"\n5-fold CV splits:")
    for fold, (tr, va) in enumerate(splits):
        y = ds.labels
        print(f"  Fold {fold}: train={len(tr)}  val={len(va)}  "
              f"val_dist={np.bincount(y[va]).tolist()}")

    # EDA coverage
    eda_valid = sum(s["eda_valid"] for s in ds._samples)
    gaze_valid = sum(s["gaze_valid"] for s in ds._samples)
    print(f"\nCoverage: EDA {eda_valid}/{len(ds)}  Gaze {gaze_valid}/{len(ds)}")
