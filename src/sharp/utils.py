"""
sharp.utils — shared helper functions for the SHARP analysis modules

Provides functions only; it does not export any paths or constants.
Paths and constants are always imported from sharp.config:
    from sharp.config import ANNO_PATH, VMAP_PATH, WINDOW_SEC, ...
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score as _sk_f1

# Import paths privately (underscore-prefixed) so they don't leak as module attributes
from .config import (
    ANNO_PATH  as _ANNO_PATH,
    VMAP_PATH  as _VMAP_PATH,
    FM_PATH    as _FM_PATH,
    UTT_PATH   as _UTT_PATH,
    GAZE_PATH  as _GAZE_PATH,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set a global deterministic seed. Call at the start of each (seed, fold): set_seed(seed * 1000 + fold)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_ckpt_epochs(fixed_epochs: int, interval: int = 10) -> list[int]:
    """Checkpoint every `interval` epochs; the final epoch is always included. Returns a sorted list."""
    return sorted(set(range(interval, fixed_epochs + 1, interval)) | {fixed_epochs})


def eval_fold_dl(model, test_dl, device: str) -> tuple[list, list]:
    """Run inference on test_dl, returning (y_true, y_pred) lists. Leaves the caller's train/eval mode unchanged."""
    model.eval()
    with torch.no_grad():
        ob    = {k: v.to(device) for k, v in next(iter(test_dl)).items()}
        preds = model(ob).argmax(1).cpu().numpy()
        y_o   = ob["label"].cpu().numpy()
    model.train()
    return y_o.tolist(), preds.tolist()


class CheckpointTracker:
    """Accumulates pooled-OOF predictions per checkpoint epoch across folds/seeds,
    and computes the resulting macro-F1 curve.

    Usage (inside an ablation loop):
        tracker = CheckpointTracker(make_ckpt_epochs(100))
        for seed in range(n_seeds):
            tracker.reset_seed()
            for fold, (tr, va) in enumerate(splits):
                set_seed(seed * 1000 + fold)
                *ret, ckpt_preds = run_fold(..., ckpt_epochs=tracker.ckpt_epochs)
                tracker.add_fold(ckpt_preds)
            tracker.commit_seed()
        means, stds, best_ep = tracker.summary(label=tag)
    """

    def __init__(self, ckpt_epochs: list[int]) -> None:
        self.ckpt_epochs: list[int] = sorted(ckpt_epochs)
        self._fold_yt:  dict[int, list] = {ep: [] for ep in self.ckpt_epochs}
        self._fold_yp:  dict[int, list] = {ep: [] for ep in self.ckpt_epochs}
        self._seed_f1s: dict[int, list[float]] = {ep: [] for ep in self.ckpt_epochs}

    def reset_seed(self) -> None:
        """Call at the start of each seed; clears the per-fold accumulators."""
        self._fold_yt = {ep: [] for ep in self.ckpt_epochs}
        self._fold_yp = {ep: [] for ep in self.ckpt_epochs}

    def add_fold(self, ckpt_preds: dict[int, tuple[list, list]]) -> None:
        """Append one fold's checkpoint predictions to the accumulators."""
        for ep in self.ckpt_epochs:
            if ep in ckpt_preds:
                yt, yp = ckpt_preds[ep]
                self._fold_yt[ep].extend(yt)
                self._fold_yp[ep].extend(yp)

    def commit_seed(self) -> None:
        """Call once all folds for a seed have finished; computes and stores that seed's pooled-OOF F1."""
        for ep in self.ckpt_epochs:
            yt, yp = self._fold_yt[ep], self._fold_yp[ep]
            if yt:
                self._seed_f1s[ep].append(
                    float(_sk_f1(yt, yp, average="macro", zero_division=0))
                )

    def best_epoch(self) -> int:
        """Return the epoch with the highest mean F1."""
        means = {ep: float(np.mean(v)) for ep, v in self._seed_f1s.items() if v}
        return max(means, key=means.get) if means else self.ckpt_epochs[-1]

    def summary(self, label: str = "") -> tuple[dict[int, float], dict[int, float], int]:
        """Print the epoch curve, returning (means, stds, best_ep)."""
        means = {ep: float(np.mean(v)) for ep, v in self._seed_f1s.items() if v}
        stds  = {ep: float(np.std(v))  for ep, v in self._seed_f1s.items() if v}
        best  = max(means, key=means.get) if means else self.ckpt_epochs[-1]
        curve = "  ".join(f"ep{ep:03d}:{means[ep]:.3f}" for ep in self.ckpt_epochs if ep in means)
        tag   = f"[{label}]" if label else ""
        print(f"  curves: {curve}  {tag}")
        return means, stds, best


# ── Time conversion ────────────────────────────────────────────────────────────

def hms_to_sec(s) -> float:
    """
    HH:MM:SS[.ffffff] -> seconds.
    Also tolerates Excel misreading MM:SS as HH:MM:SS. Returns float('nan') on failure.
    """
    try:
        h, m, rest = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)
    except Exception:
        return float("nan")


# ── Color -> participant mapping ────────────────────────────────────────────────

def build_color_pid_lookup(vmap: pd.DataFrame) -> dict[str, dict[str, str]]:
    """
    Build a {group_project: {color: p_id}} mapping.
    Color assignment varies by group, so this must be read dynamically from
    vmap rather than hardcoded.
    """
    lookup: dict[str, dict[str, str]] = {}
    rows = vmap[vmap["p_id"].notna() & vmap["participant_identification"].notna()]
    for _, row in rows.iterrows():
        gp    = str(row["group_project"]).upper()
        pid   = str(row["p_id"])
        color = str(row["participant_identification"]).strip()
        lookup.setdefault(gp, {})[color] = pid
    return lookup


def color_to_pid(group: str, color: str, color_pid: dict[str, dict[str, str]]) -> str:
    """Given a group and color, return the p_id (empty string if not found)."""
    return color_pid.get(str(group).upper(), {}).get(str(color).strip(), "")


# ── Standard data loaders ────────────────────────────────────────────────────────

def load_vmap() -> pd.DataFrame:
    return pd.read_csv(_VMAP_PATH)


def load_feature_matrix() -> pd.DataFrame:
    return pd.read_csv(_FM_PATH)


def load_annotations() -> pd.DataFrame:
    """Load the SSRL annotation xlsx and append start_sec / end_sec columns
    (integer task_sec, on the same zero-point as the other signals)."""
    anno = pd.read_excel(_ANNO_PATH)
    anno["start_sec"] = anno["Start Time"].apply(hms_to_sec)
    anno["end_sec"]   = anno["End Time"].apply(hms_to_sec)
    return anno.dropna(subset=["start_sec"]).reset_index(drop=True)


def load_utterances() -> pd.DataFrame:
    return pd.read_csv(_UTT_PATH)


def load_gaze() -> pd.DataFrame:
    gaze = pd.read_csv(_GAZE_PATH)
    gaze["group_project"] = gaze["group_project"].str.upper()
    return gaze


# ── Window helpers ────────────────────────────────────────────────────────────

def pre_post_bounds(start_sec: float, window: float) -> tuple[tuple, tuple]:
    """Return ((pre_lo, pre_hi), (post_lo, post_hi)). `window` must be passed
    explicitly (from config.WINDOW_SEC)."""
    return (start_sec - window, start_sec), (start_sec, start_sec + window)
