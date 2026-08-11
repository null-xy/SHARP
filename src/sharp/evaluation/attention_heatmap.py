#!/usr/bin/env python3
"""
sharp.evaluation.attention_heatmap

Cross-modal attention weight extraction and visualization (paper Figure 5,
generated together with sharp.figures.figure5_attention_weights).

Model: CrossModalAttentionNet trained on all events (season-filtered),
       then attention_weights() is called per event.

Usage:
  python -m sharp.evaluation.attention_heatmap --season autumn           # All3 (EDA+EmoNet+Gaze)
  python -m sharp.evaluation.attention_heatmap --season autumn --text    # All3+Text (4 modalities)

Three outputs
─────────────
1. class_attention_matrix.png
   3-panel figure (Negative / Positive / Regulate).
   Each panel is an N×N heatmap (N=3 or 4): row=query, col=key modality.
   Cell value = mean cross-attention selectivity (higher = more focused).

2. negative_event_heatmaps.png
   One row per Negative event.
   Each row shows all cross-attention weight matrices (8×8 token grids).

3. modality_attention_by_class.png
   Bar chart: per-class mean attention weight for each directed pair.

Outputs: processed_data/analysis/attention_heatmap_{season}[_text]/
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")
from ..dataset import load_sharp_dataset, LABELS_3CLASS

from ..config import ANALYSIS_DIR, ROOT

LABEL_NAMES = ["Negative", "Positive", "Regulate"]
ALL_MOD_LABELS = {"eda": "EDA", "emonet": "EmoNet", "gaze": "Gaze", "text": "Text"}

FIXED_EPOCHS = 100   # fixed budget, consistent with sharp.models.cross_modal_attention's evaluation protocol
N_XAI_SEEDS = 10    # average attention weights across multiple seeds


# ── Model (a standalone copy of sharp.models.cross_modal_attention's
#    architecture, with an added attention_weights() method for extracting
#    the weights this script visualizes) ───────────────────────────────────

class _LN1d(nn.Module):
    """LayerNorm for (B, C, T) Conv1d outputs: normalises C at each time step."""
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TokenEncoder(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int = 64, num_tokens: int = 8):
        super().__init__()
        mid = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch,  mid,       kernel_size=5, stride=2, padding=2),
            _LN1d(mid), nn.ReLU(),
            nn.Conv1d(mid,    embed_dim, kernel_size=3, stride=2, padding=1),
            _LN1d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(num_tokens),
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x).transpose(1, 2))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x    = x + h
        x    = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                              batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        return q + h


class CrossModalAttentionNet(nn.Module):
    _IN_CH = {"eda": 1, "emonet": 2, "gaze": 2, "text": 1}

    def __init__(self, active_modalities: list[str],
                 embed_dim: int = 64, num_tokens: int = 8,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.active     = active_modalities
        self.num_tokens = num_tokens

        self.encoders = nn.ModuleDict({
            m: TokenEncoder(self._IN_CH[m], embed_dim, num_tokens)
            for m in active_modalities
        })
        self.self_attn = nn.ModuleDict({
            m: TransformerBlock(embed_dim, num_heads, dropout)
            for m in active_modalities
        })
        self.cross_attn = nn.ModuleDict({
            f"{m1}_from_{m2}": CrossAttentionBlock(embed_dim, num_heads, dropout)
            for m1 in active_modalities for m2 in active_modalities if m1 != m2
        })
        n = len(active_modalities)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * n),
            nn.Linear(embed_dim * n, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 3),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        tokens  = {m: self.encoders[m](batch[m]) for m in self.active}
        tokens  = {m: self.self_attn[m](tokens[m]) for m in self.active}
        updated = {m: tokens[m] for m in self.active}
        for m1 in self.active:
            for m2 in self.active:
                if m1 != m2:
                    updated[m1] = self.cross_attn[f"{m1}_from_{m2}"](
                        updated[m1], tokens[m2]
                    )
        pooled = torch.cat([updated[m].mean(dim=1) for m in self.active], dim=1)
        return self.classifier(pooled)

    def attention_weights(self, batch: dict) -> dict[str, torch.Tensor]:
        """Return per-pair cross-attention weight matrices (B, T_q, T_kv)."""
        tokens  = {m: self.encoders[m](batch[m]) for m in self.active}
        tokens  = {m: self.self_attn[m](tokens[m]) for m in self.active}
        weights: dict[str, torch.Tensor] = {}
        for m1 in self.active:
            for m2 in self.active:
                if m1 != m2:
                    blk = self.cross_attn[f"{m1}_from_{m2}"]
                    _, w = blk.attn(
                        blk.norm_q(tokens[m1]),
                        blk.norm_kv(tokens[m2]),
                        blk.norm_kv(tokens[m2]),
                    )
                    weights[f"{m1}←{m2}"] = w.detach()   # (B, T_q, T_kv)
        return weights


# ── Training helpers ──────────────────────────────────────────────────────────

def make_collate(active_modalities: list[str]):
    keys = active_modalities + ["label"]
    def collate(batch: list[dict]) -> dict:
        return {k: torch.stack([b[k] for b in batch]) for k in keys}
    return collate

def class_weights_tensor(y: np.ndarray, device="cpu") -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(float)
    w = counts.sum() / (3 * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_all(ds, active_modalities: list[str], device: str) -> CrossModalAttentionNet:
    """Train CrossModalAttentionNet on all events for a fixed 100-epoch budget.
    No early stopping; matches sharp.models.cross_modal_attention's training protocol."""
    y    = ds.labels
    cw   = class_weights_tensor(y, device)
    crit = nn.CrossEntropyLoss(weight=cw)

    model = CrossModalAttentionNet(active_modalities).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FIXED_EPOCHS)
    collate_fn = make_collate(active_modalities)
    dl    = DataLoader(ds, batch_size=16, shuffle=True,
                       collate_fn=collate_fn, drop_last=False)

    for ep in range(FIXED_EPOCHS):
        model.train()
        for xb in dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            opt.zero_grad()
            loss = crit(model(xb), xb["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    return model


# ── Attention extraction ──────────────────────────────────────────────────────

def extract_all_weights(model: CrossModalAttentionNet, ds, device: str,
                        pair_keys: list[str], active_modalities: list[str]
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      focus_w   (N, P)     — mean(max per query row): attention selectivity score
      entropy_w (N, P)     — mean Shannon entropy per query row (lower=more selective)
      token_w   (N, P, 8, 8) — full token attention grid per pair per event
    """
    N   = len(ds)
    T   = 8
    P   = len(pair_keys)
    eps = 1e-9
    focus_w   = np.zeros((N, P), dtype=np.float32)
    entropy_w = np.zeros((N, P), dtype=np.float32)
    token_w   = np.zeros((N, P, T, T), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(N):
            sample = ds[i]
            batch  = {k: sample[k].unsqueeze(0).to(device) for k in active_modalities}
            wdict  = model.attention_weights(batch)
            for j, key in enumerate(pair_keys):
                w = wdict[key].cpu().numpy()[0]
                token_w[i, j]   = w
                focus_w[i, j]   = w.max(axis=-1).mean()
                H = -(w * np.log(w + eps)).sum(axis=-1)
                entropy_w[i, j] = H.mean()

    return focus_w, entropy_w, token_w


# ── Plot helpers ──────────────────────────────────────────────────────────────

def build_matrix(scalar_w_class: np.ndarray, modalities: list[str],
                 pairs: list[tuple]) -> np.ndarray:
    """Build N×N attention matrix (diagonal=NaN) from scalar weights."""
    n = len(modalities)
    M = np.full((n, n), np.nan)
    for k, (m1, m2) in enumerate(pairs):
        i = modalities.index(m1)
        j = modalities.index(m2)
        M[i, j] = scalar_w_class[k]
    return M


def plot_class_attention_matrix(scalar_w: np.ndarray, y: np.ndarray,
                                modalities: list[str], pairs: list[tuple],
                                path: Path):
    """3-panel N×N heatmap, one per class."""
    n = len(modalities)
    mod_labels_short = [ALL_MOD_LABELS[m] for m in modalities]
    fig, axes = plt.subplots(1, 3, figsize=(4 * n, 3.5))
    vmin = np.nanmin(scalar_w)
    vmax = np.nanmax(scalar_w)

    for c, (ax, cls_name) in enumerate(zip(axes, LABEL_NAMES)):
        idx = np.where(y == c)[0]
        M   = build_matrix(scalar_w[idx].mean(axis=0), modalities, pairs)
        im  = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(mod_labels_short, fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(mod_labels_short, fontsize=9)
        ax.set_title(f"{cls_name}\n(n={len(idx)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Key modality (attended to)", fontsize=8)
        ax.set_ylabel("Query modality", fontsize=8)
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                            fontsize=8, color="black" if M[i,j] < 0.6*vmax else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Cross-modal Attention Selectivity by Deliberation Class\n"
                 "Cell = mean(max attention per query row); higher = more focused on key modality",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path.relative_to(ROOT)}")


def plot_negative_event_heatmaps(token_w: np.ndarray, neg_idx: np.ndarray,
                                  ds, pairs: list[tuple], path: Path):
    """One row per Negative event; P columns = P cross-attn pairs (8×8 tokens)."""
    n_neg = len(neg_idx)
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_neg, n_pairs, figsize=(2.2 * n_pairs, 2.2 * n_neg))
    if n_neg == 1:
        axes = axes[np.newaxis, :]

    pair_titles = [f"{ALL_MOD_LABELS[m1]}\n←{ALL_MOD_LABELS[m2]}"
                   for m1, m2 in pairs]
    vmin, vmax = token_w[neg_idx].min(), token_w[neg_idx].max()

    for row, ei in enumerate(neg_idx):
        sample = ds[ei]
        ev_label = f"{sample['group']}\n@{sample['start_sec']:.0f}s"
        for col, (ax, ptitle) in enumerate(zip(axes[row], pair_titles)):
            im = ax.imshow(token_w[ei, col], vmin=vmin, vmax=vmax,
                           cmap="Blues", aspect="auto")
            if row == 0:
                ax.set_title(ptitle, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(ev_label, fontsize=7, rotation=0,
                              labelpad=60, va="center")
        plt.colorbar(im, ax=axes[row, -1], fraction=0.046, pad=0.04)

    plt.suptitle("Token-level Attention Maps — Negative Events\n"
                 "Row=query token (8), Col=key token (8); brighter=higher weight",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path.relative_to(ROOT)}")


def plot_bar_by_class(scalar_w: np.ndarray, y: np.ndarray,
                      pairs: list[tuple], path: Path):
    """Bar chart: mean attention weight per pair × class."""
    pair_keys  = [f"{ALL_MOD_LABELS[m1]}←{ALL_MOD_LABELS[m2]}" for m1, m2 in pairs]
    x     = np.arange(len(pair_keys))
    width = 0.25
    colors = {"Negative": "#d62728", "Positive": "#2ca02c", "Regulate": "#1f77b4"}

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(pair_keys)), 4))
    for c, (cls_name, col) in enumerate(colors.items()):
        idx  = np.where(y == c)[0]
        vals = scalar_w[idx].mean(axis=0)
        errs = scalar_w[idx].std(axis=0)
        ax.bar(x + (c - 1) * width, vals, width,
               label=f"{cls_name} (n={len(idx)})",
               color=col, alpha=0.8,
               yerr=errs, capsize=3, error_kw={"linewidth": 0.8})

    short_keys = pair_keys
    ax.set_xticks(x)
    ax.set_xticklabels(short_keys, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Attention selectivity (mean max per query row)", fontsize=9)
    ax.set_title("Cross-modal Attention Selectivity by Deliberation Class\n"
                 "Error bars = ±1 SD across events  |  0.125=uniform, 1.0=fully focused",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path.relative_to(ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(season: str | None = None, use_text: bool = False):
    suffix = ("_" + season if season else "") + ("_text" if use_text else "")
    out_dir = ANALYSIS_DIR / f"attention_heatmap{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    active_modalities = ["eda", "emonet", "gaze"] + (["text"] if use_text else [])
    pairs    = [(m1, m2) for m1 in active_modalities
                          for m2 in active_modalities if m1 != m2]
    pair_keys = [f"{m1}←{m2}" for m1, m2 in pairs]

    ds = load_sharp_dataset(individual=False, load_text=use_text, season=season,
                            require_eda=True)
    y  = ds.labels
    mod_str = "+".join(ALL_MOD_LABELS[m] for m in active_modalities)
    print(f"Dataset: {len(ds)} events  |  class dist: {np.bincount(y).tolist()}")
    print(f"Modalities: {mod_str}\n")

    N   = len(ds)
    P   = len(pair_keys)
    T   = 8
    focus_acc   = np.zeros((N, P), dtype=np.float64)
    entropy_acc = np.zeros((N, P), dtype=np.float64)
    token_acc   = np.zeros((N, P, T, T), dtype=np.float64)

    print(f"Training CrossModalAttentionNet ({mod_str}) on all events "
          f"({N_XAI_SEEDS} seeds, averaged)...")
    for seed in range(N_XAI_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_all(ds, active_modalities, device)
        fw, ew, tw = extract_all_weights(model, ds, device, pair_keys, active_modalities)
        focus_acc   += fw
        entropy_acc += ew
        token_acc   += tw
        print(f"  Seed {seed}: done")

    focus_w   = (focus_acc   / N_XAI_SEEDS).astype(np.float32)
    entropy_w = (entropy_acc / N_XAI_SEEDS).astype(np.float32)
    token_w   = (token_acc   / N_XAI_SEEDS).astype(np.float32)
    print(f"Averaged over {N_XAI_SEEDS} seeds.\n")
    print(f"  focus_w shape: {focus_w.shape}   (N × {len(pair_keys)} pairs)\n")

    # Save raw scores
    focus_cols   = [f"focus_{k}"   for k in pair_keys]
    entropy_cols = [f"entropy_{k}" for k in pair_keys]
    df = pd.DataFrame(
        np.concatenate([focus_w, entropy_w], axis=1),
        columns=focus_cols + entropy_cols,
    )
    df.insert(0, "label", [LABEL_NAMES[yi] for yi in y])
    for col in ["group", "start_sec", "deliberation"]:
        df.insert(list(df.columns).index("label"), col,
                  [ds[i][col] for i in range(len(ds))])
    df.to_csv(out_dir / "attention_weights_raw.csv", index=False)
    print(f"Saved: {(out_dir/'attention_weights_raw.csv').relative_to(ROOT)}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\nAttention selectivity (mean max-per-row) by class:")
    for c, cls_name in enumerate(LABEL_NAMES):
        idx = np.where(y == c)[0]
        row = focus_w[idx].mean(axis=0)
        print(f"  {cls_name:10s}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in zip(pair_keys, row)))

    print("\nAttention entropy (lower=more selective) by class:")
    for c, cls_name in enumerate(LABEL_NAMES):
        idx = np.where(y == c)[0]
        row = entropy_w[idx].mean(axis=0)
        print(f"  {cls_name:10s}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in zip(pair_keys, row)))

    # ── Figures ───────────────────────────────────────────────────────────────
    neg_idx = np.where(y == 0)[0]

    plot_class_attention_matrix(focus_w, y, active_modalities, pairs,
                                out_dir / "class_attention_matrix.png")
    plot_negative_event_heatmaps(token_w, neg_idx, ds, pairs,
                                 out_dir / "negative_event_heatmaps.png")
    plot_bar_by_class(focus_w, y, pairs,
                      out_dir / "modality_attention_by_class.png")

    # ── Key finding summary ───────────────────────────────────────────────────
    print("\n=== Key Findings (Selectivity: Negative − Positive) ===")
    neg_mean = focus_w[neg_idx].mean(axis=0)
    pos_idx  = np.where(y == 1)[0]
    pos_mean = focus_w[pos_idx].mean(axis=0)
    diffs    = neg_mean - pos_mean

    for k, d in sorted(zip(pair_keys, diffs), key=lambda x: -abs(x[1])):
        direction = "↑ Neg more selective" if d > 0 else "↓ Neg less selective"
        print(f"  {k:20s}  Δ={d:+.4f}  {direction}")

    print(f"\nOutputs: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["autumn", "spring"], default=None)
    ap.add_argument("--text",   action="store_true",
                    help="Include Text sentiment as 4th modality")
    args = ap.parse_args()
    main(season=args.season, use_text=args.text)
