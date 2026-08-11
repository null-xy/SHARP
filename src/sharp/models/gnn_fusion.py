#!/usr/bin/env python3
"""
sharp.models.gnn_fusion

Graph-based fusion models for 3-class SSRL event recognition (paper
Table 4, "graph-based fusion" rows: GNN1-Mean, GNN2-Mean, Participant-Attn,
Serial-GNN-Attn, Parallel-GNN-Attn). This is the paper's third fusion
family, alongside Early Fusion (sharp.models.early_fusion) and Cross-Modal
Attention (sharp.models.cross_modal_attention): instead of a single group-mean signal
per modality, each event is represented as a small graph over the 3 group
members, to test whether modeling participant-to-participant interaction
explicitly helps beyond aggregating to the group mean.

Each event is a fully connected graph:
  Nodes  : P1, P2, P3  (the 3 group members)
  Edges  : fully connected (6 directed edges; each person attends to every other)
  Message: each node aggregates its 2 neighbours' embeddings (mean aggregation)

Node features = per-participant EDA (1ch) + EmoNet (2ch) + Gaze (2ch) x 60 s

Architecture (3 nodes only, so this is plain PyTorch rather than a graph
library like PyG):
  1. NodeEncoder  (Conv1d, weights shared across participants)
       (B x 3, 5, 60) -> (B x 3, node_dim)
  2. GraphConvLayer x {0, 1, 2}
       mean-aggregation + linear update + LayerNorm + ReLU
  3. Readout: mean-pool or concat the 3 node embeddings
  4. MLP classifier -> 3 classes

Ablations (see the ABLATIONS list below): No-Graph, GNN1-Mean, GNN2-Mean,
CrossAttn (= Participant-Attn in the paper), Serial-GNN-Attn,
Parallel-GNN-Attn, NoGraph-Concat, GNN1-Concat. The paper reports the five
GNN1-Mean / GNN2-Mean / CrossAttn / Serial-GNN-Attn / Parallel-GNN-Attn
rows; the remaining three (No-Graph, NoGraph-Concat, GNN1-Concat) are
additional ablations exploring the readout design (mean-pool vs.
concatenation) that are not part of the paper's main comparison table.

Protocol: Leave-One-Group-Out cross-validation (5 folds), pooled-OOF
macro-F1, 10 random seeds, fixed 100-epoch training budget with
checkpoints saved every 10 epochs.

Outputs: processed_data/analysis/gnn/
  results_gnn.csv   - per-ablation x per-fold F1 / accuracy
  summary_gnn.csv   - mean +/- std
  curves_best.png   - training curves for the best-performing GNN config
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")
from ..dataset import load_sharp_dataset, get_cv_splits, impute_eda_fold, restore_eda
from ..utils import set_seed, make_ckpt_epochs, CheckpointTracker, eval_fold_dl

from ..config import ANALYSIS_DIR, ROOT
out_dir = ANALYSIS_DIR / "gnn"

NODE_DIM     = 48       # embedding dim per participant node
MAX_EPOCHS   = 200
PATIENCE     = 30
LABEL_NAMES  = ["Negative", "Positive", "Regulate"]


# ── Node encoder ──────────────────────────────────────────────────────────────

class _LN1d(nn.Module):
    """LayerNorm for (B, C, T) Conv1d outputs: normalises C at each time step."""
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class NodeEncoder(nn.Module):
    """
    Encode one participant's (5, 60) tensor → node_dim embedding.
    Three separate Conv1d streams (EDA=1ch, EmoNet=2ch, Gaze=2ch) are
    encoded independently then concatenated.
    Weights are shared across P1/P2/P3 (applied as batch dimension).
    """
    _STREAMS = {"eda": (0, 1), "emonet": (1, 3), "gaze": (3, 5)}

    def __init__(self, node_dim: int = 48, mid_ch: int = 32):
        super().__init__()
        self.convs = nn.ModuleDict()
        in_ch_map  = {"eda": 1, "emonet": 2, "gaze": 2}
        for name, in_ch in in_ch_map.items():
            self.convs[name] = nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, kernel_size=5, stride=2, padding=2),
                _LN1d(mid_ch), nn.ReLU(),
                nn.Conv1d(mid_ch, mid_ch, kernel_size=3, stride=2, padding=1),
                _LN1d(mid_ch), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),   # (B, mid_ch, 1)
                nn.Flatten(),              # (B, mid_ch)
            )
        self.proj = nn.Sequential(
            nn.Linear(mid_ch * 3, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, 60) — B participants from any batch
        parts = []
        for name, (s, e) in self._STREAMS.items():
            parts.append(self.convs[name](x[:, s:e, :]))
        return self.proj(torch.cat(parts, dim=1))   # (B, node_dim)


# ── GNN layer (mean aggregation, 3-node fully connected) ─────────────────────

class GraphConvLayer(nn.Module):
    """
    One round of mean-aggregation message passing on a 3-node fully connected graph.
    h_new[i] = LayerNorm( h[i] + ReLU( Linear([h[i] || mean_j≠i(h[j]) ]) ) )
    """
    def __init__(self, node_dim: int):
        super().__init__()
        self.update = nn.Linear(node_dim * 2, node_dim)
        self.norm   = nn.LayerNorm(node_dim)
        self.drop   = nn.Dropout(0.2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, 3, node_dim)
        total = h.sum(dim=1, keepdim=True)              # (B, 1, node_dim)
        agg   = (total - h) / 2.0                       # mean of the other 2 nodes
        msg   = self.drop(torch.relu(self.update(
            torch.cat([h, agg], dim=-1)                  # (B, 3, 2*node_dim)
        )))
        return self.norm(h + msg)                        # residual + LayerNorm


# ── Full GNN model ────────────────────────────────────────────────────────────

def _encode_participants(batch: dict, encoder: nn.Module) -> torch.Tensor:
    """Shared forward prefix: concat EDA/EmoNet/Gaze -> NodeEncoder -> (B, 3, node_dim)."""
    B = batch["eda"].shape[0]
    eda    = batch["eda"].unsqueeze(2)               # (B, 3, 1, 60)
    emonet = batch["emonet"].view(B, 3, 2, 60)
    gaze   = batch["gaze"].view(B, 3, 2, 60)
    x = torch.cat([eda, emonet, gaze], dim=2)        # (B, 3, 5, 60)
    return encoder(x.view(B * 3, 5, 60)).view(B, 3, -1)  # (B, 3, node_dim)


class GroupGNN(nn.Module):
    """GNN fusion: NodeEncoder -> GraphConvLayer x n -> readout -> MLP.

    readout modes:
        "mean"     - fixed mean-pool, 48-dim  (no projection, discards participant identity)
        "concat"   - concat [P1,P2,P3], 144-dim (no projection)
        "proj-{d}" - concat 144-dim -> Linear(144, d)  (learnable capacity control)
    """
    def __init__(self, n_gnn_layers: int = 2, node_dim: int = NODE_DIM,
                 readout: str = "concat"):
        super().__init__()
        self.encoder    = NodeEncoder(node_dim)
        self.gnn_layers = nn.ModuleList(
            [GraphConvLayer(node_dim) for _ in range(n_gnn_layers)]
        )
        self._readout = readout

        if readout == "mean":
            self.proj = None
            in_dim = node_dim
        elif readout == "concat":
            self.proj = None
            in_dim = node_dim * 3
        elif readout.startswith("proj-"):
            proj_dim = int(readout.split("-")[1])
            self.proj = nn.Sequential(
                nn.LayerNorm(node_dim * 3),
                nn.Linear(node_dim * 3, proj_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            in_dim = proj_dim
        else:
            raise ValueError(f"Unknown readout mode: {readout!r}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 3),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        h = _encode_participants(batch, self.encoder)  # (B, 3, node_dim)
        for layer in self.gnn_layers:
            h = layer(h)
        if self._readout == "mean":
            graph_emb = h.mean(dim=1)
        elif self._readout == "concat":
            graph_emb = h.view(h.shape[0], -1)
        else:  # proj-d
            graph_emb = self.proj(h.view(h.shape[0], -1))
        return self.classifier(graph_emb)


class GroupCrossAttn(nn.Module):
    """NodeEncoder -> (optional GNN layers) -> MHA -> mean-pool -> MLP.

    n_gnn_layers=0 : pure CrossAttn (baseline)
    n_gnn_layers=1 : Serial-GNN-Attn (GNN smooths first, then MHA; ablation arm)
    """
    def __init__(self, node_dim: int = NODE_DIM, num_heads: int = 4,
                 n_gnn_layers: int = 0):
        super().__init__()
        self.encoder    = NodeEncoder(node_dim)
        self.gnn_layers = nn.ModuleList(
            [GraphConvLayer(node_dim) for _ in range(n_gnn_layers)]
        )
        self.norm       = nn.LayerNorm(node_dim)
        self.attn       = nn.MultiheadAttention(node_dim, num_heads,
                                                dropout=0.1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 3),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        h = _encode_participants(batch, self.encoder)    # (B, 3, node_dim)
        for layer in self.gnn_layers:
            h = layer(h)
        h_n = self.norm(h)
        h_a, _ = self.attn(h_n, h_n, h_n)
        h   = h + h_a                                    # residual
        return self.classifier(h.mean(dim=1))


class ParallelGNNAttn(nn.Module):
    """GNN and CrossAttn in parallel: both branches start from the raw h0,
    then are combined with a learned gate.

    h = h0 + sigmoid(alpha) * (GNN(h0) - h0) + sigmoid(beta) * Attn(norm(h0))

    Key difference from Serial-GNN-Attn:
    - GNN and Attn run independently in parallel, not one after the other
    - Attn sees the raw per-node differences, not the GNN's already-smoothed representation
    - learnable gate: the model decides how much each branch should contribute
    - alpha=beta=0 initially -> sigmoid=0.5, so both branches start on equal footing
    - lower MHA/classifier dropout: the parallel design stacks less dropout than the serial one
    """
    def __init__(self, node_dim: int = NODE_DIM, num_heads: int = 4):
        super().__init__()
        self.encoder = NodeEncoder(node_dim)
        self.gnn     = GraphConvLayer(node_dim)
        self.norm    = nn.LayerNorm(node_dim)
        self.attn    = nn.MultiheadAttention(node_dim, num_heads,
                                             dropout=0.0, batch_first=True)
        self.alpha   = nn.Parameter(torch.tensor(0.0))   # gate: GNN delta
        self.beta    = nn.Parameter(torch.tensor(0.0))   # gate: Attn output
        self.classifier = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        h0 = _encode_participants(batch, self.encoder)    # (B, 3, node_dim)

        h_g = self.gnn(h0) - h0                           # GNN delta (message-passing contribution only)

        h_n = self.norm(h0)
        h_a, _ = self.attn(h_n, h_n, h_n)                # Attn operates on raw h0

        a = torch.sigmoid(self.alpha)
        b = torch.sigmoid(self.beta)
        h = h0 + a * h_g + b * h_a
        return self.classifier(h.mean(dim=1))


# ── Training helpers ──────────────────────────────────────────────────────────

def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch])
            for k in ["eda", "emonet", "gaze", "label"]}

def class_weights_tensor(y: np.ndarray, device="cpu") -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(float)
    w = counts.sum() / (3 * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def run_fold(
    ds, train_idx: np.ndarray, val_idx: np.ndarray,
    model_fn, device: str,
    fixed_epochs: int = 100,
    lr: float = 3e-4, wd: float = 1e-2, batch_size: int = 16,
    ckpt_epochs: list[int] | None = None,
    ckpt_save_dir: Path | None = None,
) -> tuple:
    """Train for a fixed number of epochs on the full training fold.
    No early stopping; cosine annealing provides LR decay.
    val_idx (test fold) is evaluated once after training completes.
    """
    saved_eda = impute_eda_fold(ds, train_idx)

    y_tr  = ds.labels[train_idx]
    cw    = class_weights_tensor(y_tr, device)
    crit  = nn.CrossEntropyLoss(weight=cw)

    model = model_fn().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fixed_epochs)

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=batch_size,
                          shuffle=True, collate_fn=collate, drop_last=False)
    test_dl  = DataLoader(Subset(ds, val_idx),   batch_size=len(val_idx),
                          shuffle=False, collate_fn=collate)

    train_losses = []
    ckpt_set   = set(ckpt_epochs) if ckpt_epochs else set()
    ckpt_preds: dict[int, tuple[list, list]] = {}

    for epoch in range(fixed_epochs):
        model.train()
        ep_loss = 0.0
        for xb in train_dl:
            xb = {k: v.to(device) for k, v in xb.items()}
            opt.zero_grad()
            loss = crit(model(xb), xb["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb["label"])
        sched.step()
        train_losses.append(ep_loss / len(train_idx))
        if (epoch + 1) in ckpt_set:
            ckpt_preds[epoch + 1] = eval_fold_dl(model, test_dl, device)
            if ckpt_save_dir is not None:
                ckpt_save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(),
                           ckpt_save_dir / f"ep{epoch+1:03d}.pt")

    # Final evaluation on test fold.
    y_o, preds_list = eval_fold_dl(model, test_dl, device)
    preds = np.array(preds_list)
    y_o_arr = np.array(y_o)
    test_f1  = f1_score(y_o_arr, preds, average="macro", zero_division=0)
    test_acc = (preds == y_o_arr).mean()

    restore_eda(ds, saved_eda)
    return test_f1, test_acc, train_losses, [], y_o, preds_list, ckpt_preds


# ── Ablation configs ──────────────────────────────────────────────────────────

# Readout capacity ablation matrix:
#   rows  = message-passing depth (0 / 1 layer GNN)
#   cols  = readout bottleneck dim (mean-pool | proj-24 | proj-48 | proj-96 | concat-144)
#
# Key question: what is the optimal group-representation capacity given n=29?
# "mean" is a fixed aggregation; "proj-d" is learnable but capacity-controlled.
# Complete readout ablation matrix:
#
#              | mean-pool (0 params) | proj-48 (LN+Linear+ReLU) | proj-96 | concat-144 |
#   No-GNN     |  No-Graph            | NoGraph-Proj-48           | ...     | NoGraph-Concat |
#   GNN-1L     |  GNN1-Mean  ←KEY    | GNN1-Proj-48              | ...     | GNN1-Concat    |
#   GNN-2L     |  GNN2-Mean  ←KEY    | —                         | —       | —              |
#   CrossAttn  |  (mean-pool, dynamic MHA)                                                   |
#
# GNN1-Mean / GNN2-Mean are the critical ablations:
#   No-Graph mean-pool vs GNN1-Mean = effect of 1 GNN layer, with readout held constant.
ABLATIONS = [
    # ── mean-pool (0 readout params) ──
    ("No-Graph",           lambda: GroupGNN(n_gnn_layers=0, readout="mean")),
    ("GNN1-Mean",          lambda: GroupGNN(n_gnn_layers=1, readout="mean")),
    ("GNN2-Mean",          lambda: GroupGNN(n_gnn_layers=2, readout="mean")),
    # ── Attn variants ──
    ("CrossAttn",          lambda: GroupCrossAttn(n_gnn_layers=0)),    # pure MHA, baseline
    ("Serial-GNN-Attn",    lambda: GroupCrossAttn(n_gnn_layers=1)),    # serial (negative ablation)
    ("Parallel-GNN-Attn",  lambda: ParallelGNNAttn()),                 # parallel gated (proposed)
    # ── projection (static bottleneck, all worse than mean-pool) ──
    ("NoGraph-Concat",     lambda: GroupGNN(n_gnn_layers=0, readout="concat")),
    ("GNN1-Concat",        lambda: GroupGNN(n_gnn_layers=1, readout="concat")),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(season: str | None = None, n_seeds: int = 10) -> None:
    out_dir = ANALYSIS_DIR / ("gnn" + (f"_{season}" if season else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Seeds: {n_seeds}\n")

    ds = load_sharp_dataset(individual=True, season=season, require_eda=True)
    s0 = ds[0]
    print(f"Individual shapes — EDA:{tuple(s0['eda'].shape)}  "
          f"EmoNet:{tuple(s0['emonet'].shape)}  Gaze:{tuple(s0['gaze'].shape)}\n")

    result_rows: list[dict] = []
    seed_rows:   list[dict] = []
    epoch_rows:  list[dict] = []
    curves_best: dict | None = None
    best_mean_f1 = -1.0

    for tag, model_fn in ABLATIONS:
        seed_f1s:    list[float] = []
        seed0_curves: list     = []
        tracker = CheckpointTracker(make_ckpt_epochs(100))

        for seed in range(n_seeds):
            splits = get_cv_splits(ds, random_state=seed)
            seed_y_true: list[int] = []
            seed_y_pred: list[int] = []
            tracker.reset_seed()
            for fold, (tr_idx, va_idx) in enumerate(splits):
                set_seed(seed * 1000 + fold)
                ckpt_dir = out_dir / "checkpoints" / tag / f"seed{seed:02d}_fold{fold}"
                f1, acc, tl, vl, y_true, y_pred, ckpt_preds = run_fold(
                    ds, tr_idx, va_idx, model_fn, device,
                    ckpt_epochs=tracker.ckpt_epochs,
                    ckpt_save_dir=ckpt_dir,
                )
                seed_y_true.extend(y_true)
                seed_y_pred.extend(y_pred)
                result_rows.append({
                    "ablation": tag, "seed": seed, "fold": fold,
                    "val_f1":  round(f1, 4),
                    "val_acc": round(acc, 4),
                })
                if seed == 0:
                    seed0_curves.append((tl, vl))
                tracker.add_fold(ckpt_preds)
            seed_f1 = float(f1_score(seed_y_true, seed_y_pred, average="macro", zero_division=0))
            seed_f1s.append(seed_f1)
            seed_rows.append({"ablation": tag, "seed": seed, "seed_f1": round(seed_f1, 4)})
            tracker.commit_seed()

        means, stds, best_ep = tracker.summary(label=tag)
        mean_f1 = means[best_ep]
        std_f1  = stds.get(best_ep, 0.0)
        print(f"{tag:12s}  F1={mean_f1:.3f}±{std_f1:.3f}  best@ep{best_ep:03d}")
        for ep, m in means.items():
            epoch_rows.append({"ablation": tag, "epoch": ep, "f1_mean": round(m, 4),
                                "f1_std": round(stds.get(ep, 0.0), 4)})

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            curves_best  = {"tag": tag, "folds": seed0_curves}

    # ── Save results ──────────────────────────────────────────────────────────
    df_res = pd.DataFrame(result_rows)
    df_res.to_csv(out_dir / "results_gnn.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_dir / "results_gnn_per_seed.csv", index=False)

    summary = (df_res.groupby("ablation")[["val_f1","val_acc"]]
               .agg(["mean","std"]).round(4))
    summary.columns = ["f1_mean","f1_std","acc_mean","acc_std"]
    summary = summary.sort_values("f1_mean", ascending=False)
    summary.to_csv(out_dir / "summary_gnn.csv")
    pd.DataFrame(epoch_rows).to_csv(out_dir / "results_gnn_by_epoch.csv", index=False)
    print(f"\n=== Summary ===")
    print(summary.to_string())

    # ── Training curves for best GNN config ───────────────────────────────────
    if curves_best:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for fi, (tl, vl) in enumerate(curves_best["folds"]):
            axes[0].plot(range(len(tl)), tl, alpha=0.6, label=f"Fold {fi}")
            if vl:
                axes[1].plot(range(len(vl)), vl, alpha=0.6, label=f"Fold {fi}")
        for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
            ax.set_xlabel("Epoch"); ax.set_ylabel("CrossEntropy")
            ax.set_title(f"{curves_best['tag']} — {title}")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        plt.suptitle(f"Training curves — best GNN: {curves_best['tag']}"
                     f"  (mean F1={best_mean_f1:.3f})", fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "curves_best.png", dpi=150)
        plt.close()
        print(f"Saved: {(out_dir/'curves_best.png').relative_to(ROOT)}")

    print(f"\nOutputs: {out_dir.relative_to(ROOT)}")
    print(f"Best GNN F1: {best_mean_f1:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["autumn", "spring"], default=None)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    main(season=args.season, n_seeds=args.seeds)
