#!/usr/bin/env python3
"""
sharp.models.cross_modal_attention

Cross-Modal Attention fusion model for 3-class SSRL event recognition.
This is the paper's primary model (Section 4.3, "Cross-Modal Attention"):
it is the source of the CrossModal Attn numbers in Table 3
(modality-combination ablation) and Table 4 (fusion-strategy comparison),
and of the confusion matrix (Figure 4, via sharp.evaluation.confusion_matrix)
and attention-weight interpretability analysis (Figure 5, via
sharp.evaluation.attention_heatmap).

Architecture:
  3 modality-specific TokenEncoders (Conv1d -> 8 tokens x 64-dim each)
  -> per-modality self-attention (TransformerBlock)
  -> pairwise cross-attention: each modality queries every other modality
     (6 directed CrossAttnBlocks for 3 modalities)
  -> mean-pool -> concat -> MLP classifier

Ablation: all 7 non-empty subsets of {EDA, EmoNet, Gaze}.

Outputs: processed_data/analysis/attn_fusion/
  results_attn.csv   - per-ablation x per-fold F1 / accuracy
  summary_attn.csv   - mean +/- std
  curves_best.png    - training curves for the best-performing modality subset
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")
from ..dataset import load_sharp_dataset, get_cv_splits, impute_eda_fold, restore_eda
from ..utils import set_seed, make_ckpt_epochs, CheckpointTracker, eval_fold_dl

from ..config import ANALYSIS_DIR, ROOT
out_dir = ANALYSIS_DIR / "attn_fusion"


# ── Model components ──────────────────────────────────────────────────────────

class _LN1d(nn.Module):
    """LayerNorm for (B, C, T) Conv1d outputs: normalises C at each time step."""
    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TokenEncoder(nn.Module):
    """
    (B, in_ch, 60) → (B, num_tokens, embed_dim)
    Two Conv1d layers (stride=2) halve the sequence: 60→30→15,
    then AdaptiveAvgPool1d(num_tokens) → fixed-size token sequence.
    """
    def __init__(self, in_ch: int, embed_dim: int = 64, num_tokens: int = 8):
        super().__init__()
        mid = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch,  mid,       kernel_size=5, stride=2, padding=2),
            _LN1d(mid), nn.ReLU(),
            nn.Conv1d(mid,    embed_dim, kernel_size=3, stride=2, padding=1),
            _LN1d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(num_tokens),   # (B, embed_dim, num_tokens)
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x).transpose(1, 2)   # (B, num_tokens, embed_dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Standard pre-norm self-attention block."""
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
    """q from modality A, k/v from modality B. Residual added to A."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                              batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                r: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        if r is not None:
            h = h * r.unsqueeze(1)   # (B,1,1) broadcasts over (B, T, dim)
        return q + h


class ReliabilityEstimator(nn.Module):
    """
    Estimates per-sample modality reliability from representation-level
    physiological-facial discordance (DCRG: Discordance-Conditioned Reliability Gating).

    Inputs:  h_emonet, h_eda — token sequences post-self-attention (B, T, dim)
    Outputs: r_face (B,1), r_eda (B,1), D_repr (B,1)

    D_repr = 1 - cosine_sim(pool(h_emonet), pool(h_eda))
    Training signal: BCE(r_face, exp(-D_repr))  →  high discordance → low r_face
    """
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, 32), nn.ReLU(),
            nn.Linear(32, 2), nn.Sigmoid(),
        )

    def forward(self, h_emonet: torch.Tensor, h_eda: torch.Tensor):
        pf  = h_emonet.mean(dim=1)                                        # (B, dim)
        pe  = h_eda.mean(dim=1)                                           # (B, dim)
        cos = F.cosine_similarity(pf, pe, dim=-1).unsqueeze(-1)           # (B, 1)
        D   = (1.0 - cos).clamp(0.0, 2.0)                                 # (B, 1)
        r   = self.mlp(torch.cat([pf, pe, D], dim=-1))                    # (B, 2)
        return r[:, 0:1], r[:, 1:2], D                                    # r_face, r_eda, D


class CrossModalAttentionNet(nn.Module):
    """
    Full pairwise cross-attention fusion for n_mod modalities.
    For n_mod=3: 6 cross-attention blocks (each of 3 modalities queries 2 others).
    For n_mod=4: 12 cross-attention blocks (EDA+EmoNet+Gaze+Text).
    """
    _IN_CH = {"eda": 1, "emonet": 2, "gaze": 2, "text": 1}

    def __init__(
        self,
        active_modalities: list[str],
        embed_dim:  int   = 64,
        num_tokens: int   = 8,
        num_heads:  int   = 4,
        dropout:    float = 0.1,
        use_dcrg:   bool  = False,
        in_ch_overrides: dict | None = None,
    ):
        super().__init__()
        self.active     = active_modalities
        self.embed_dim  = embed_dim
        self.num_tokens = num_tokens

        in_ch = dict(self._IN_CH)
        if in_ch_overrides:
            in_ch.update(in_ch_overrides)

        # Per-modality token encoders
        self.encoders = nn.ModuleDict({
            m: TokenEncoder(in_ch[m], embed_dim, num_tokens)
            for m in active_modalities
        })

        # Per-modality self-attention
        self.self_attn = nn.ModuleDict({
            m: TransformerBlock(embed_dim, num_heads, dropout)
            for m in active_modalities
        })

        # Pairwise cross-attention: m1 queries m2 for all m1≠m2
        self.cross_attn = nn.ModuleDict({
            f"{m1}_from_{m2}": CrossAttentionBlock(embed_dim, num_heads, dropout)
            for m1 in active_modalities
            for m2 in active_modalities
            if m1 != m2
        })

        # Classifier input is always embed_dim regardless of modality count.
        # Modality representations are mean-pooled before classification so that
        # parameter count does not grow with n_mod (avoids overfitting on this small dataset).
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )

        # DCRG: only active when both emonet and eda are present
        self.use_dcrg = (
            use_dcrg
            and "emonet" in active_modalities
            and "eda" in active_modalities
        )
        if self.use_dcrg:
            self.reliability = ReliabilityEstimator(embed_dim)

    def forward(self, batch: dict,
                return_reliability: bool = False) -> torch.Tensor | tuple:
        tokens = {m: self.encoders[m](batch[m]) for m in self.active}
        tokens = {m: self.self_attn[m](tokens[m]) for m in self.active}

        r_face = r_eda = D_repr = None
        if self.use_dcrg:
            r_face, r_eda, D_repr = self.reliability(tokens["emonet"], tokens["eda"])

        updated = {m: tokens[m] for m in self.active}
        for m1 in self.active:
            for m2 in self.active:
                if m1 != m2:
                    r = None
                    if self.use_dcrg:
                        r = r_face if m2 == "emonet" else (r_eda if m2 == "eda" else None)
                    updated[m1] = self.cross_attn[f"{m1}_from_{m2}"](
                        updated[m1], tokens[m2], r=r
                    )

        # Mean over modalities → fixed (B, embed_dim) regardless of n_mod
        pooled = torch.stack([updated[m].mean(dim=1) for m in self.active], dim=1).mean(dim=1)
        logits = self.classifier(pooled)
        if return_reliability:
            return logits, r_face, r_eda, D_repr
        return logits

    def attention_weights(self, batch: dict) -> dict[str, torch.Tensor]:
        """Return cross-attention weight matrices (for inspection)."""
        tokens = {m: self.encoders[m](batch[m]) for m in self.active}
        tokens = {m: self.self_attn[m](tokens[m]) for m in self.active}
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
                    weights[f"{m1}←{m2}"] = w.detach()
        return weights


# ── Training helpers ──────────────────────────────────────────────────────────

def class_weights(y: np.ndarray, n_classes: int = 3, device="cpu") -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def collate(batch: list[dict]) -> dict:
    tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    return {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}


def batch_discordance(xb: dict) -> torch.Tensor:
    """
    Compute per-sample EDA-vs-arousal discordance from raw input tensors.
    Window: 0:30 = pre-onset, 30:60 = post-onset.
    Returns D (B,), z-scored within the batch.
    """
    eda_delta = xb["eda"][:, 0, 30:].mean(-1) - xb["eda"][:, 0, :30].mean(-1)
    ar_delta  = xb["emonet"][:, 0, 30:].mean(-1) - xb["emonet"][:, 0, :30].mean(-1)
    def bz(x):
        return (x - x.mean()) / (x.std() + 1e-6)
    return (bz(eda_delta) - bz(ar_delta)).abs()   # (B,)


def run_fold(
    ds, train_idx, val_idx, active, device,
    embed_dim=64, num_tokens=8, num_heads=4,
    fixed_epochs=100,
    lr=3e-4, wd=1e-2, batch_size=16,
    use_dcrg=False, lambda_rel=0.3,
    use_dgmd=False, dgmd_base=0.15, dgmd_scale=0.35,
    ckpt_epochs: list[int] | None = None,
    ckpt_save_dir: Path | None = None,
    in_ch_overrides: dict | None = None,
):
    """Train for a fixed number of epochs on the full training fold.
    No early stopping; cosine annealing provides LR decay.
    val_idx (test fold) is evaluated once after training completes.
    """
    saved_eda = impute_eda_fold(ds, train_idx) if "eda" in active else {}

    y_train = ds.labels[train_idx]
    cw      = class_weights(y_train, device=device)
    crit    = nn.CrossEntropyLoss(weight=cw)

    model = CrossModalAttentionNet(
        active, embed_dim=embed_dim,
        num_tokens=num_tokens, num_heads=num_heads,
        use_dcrg=use_dcrg,
        in_ch_overrides=in_ch_overrides,
    ).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fixed_epochs)

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=batch_size,
                          shuffle=True, collate_fn=collate, drop_last=False)
    test_dl  = DataLoader(Subset(ds, val_idx), batch_size=len(val_idx),
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
            if use_dgmd and "emonet" in active and "eda" in active:
                D = batch_discordance(xb)
                p_mask = dgmd_base + dgmd_scale * torch.sigmoid(D - D.mean())
                mask = (torch.rand_like(p_mask) < p_mask).view(-1, 1, 1)
                xb["emonet"] = xb["emonet"] * (~mask).float()

            if model.use_dcrg:
                logits, r_face, _, D_r = model(xb, return_reliability=True)
                loss = crit(logits, xb["label"])
                target_r = torch.exp(-D_r.detach())
                loss = loss + lambda_rel * F.binary_cross_entropy(r_face, target_r)
            else:
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

ALL_MOD   = ["eda", "emonet", "gaze"]
ALL_MOD_4 = ["eda", "emonet", "gaze", "text"]

# Original 3-modality ablations
ABLATIONS = [
    combo
    for r in range(1, 4)
    for combo in itertools.combinations(ALL_MOD, r)
]
# Extra 4-modality ablations (text as 4th)
ABLATIONS_TEXT = [
    ("emonet", "gaze", "text"),
    ("eda", "emonet", "gaze", "text"),
]

LABELS_MAP = {
    ("eda",):                          "EDA",
    ("emonet",):                       "EmoNet",
    ("gaze",):                         "Gaze",
    ("eda", "emonet"):                 "EDA+EmoNet",
    ("eda", "gaze"):                   "EDA+Gaze",
    ("emonet", "gaze"):                "EmoNet+Gaze",
    ("eda", "emonet", "gaze"):         "All3",
    ("emonet", "gaze", "text"):        "EmoNet+Gaze+Text",
    ("eda", "emonet", "gaze", "text"): "All3+Text",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(season: str | None = None, n_seeds: int = 10) -> None:
    out_dir = ANALYSIS_DIR / ("attn_fusion" + (f"_{season}" if season else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Seeds: {n_seeds}\n")

    ds = load_sharp_dataset(individual=False, load_text=True, season=season, require_eda=True)
    has_text = any(s.get("text") is not None for s in ds._samples)
    print(f"Text modality loaded: {has_text}\n")

    result_rows: list[dict] = []
    seed_rows:   list[dict] = []
    epoch_rows:  list[dict] = []
    curves_best:  dict | None = None
    best_mean_f1 = -1.0

    all_combos = ABLATIONS + (ABLATIONS_TEXT if has_text else [])

    for combo in all_combos:
        tag    = LABELS_MAP.get(combo, "+".join(combo))
        active = list(combo)
        if "text" in active and not has_text:
            continue
        seed_f1s:   list[float] = []
        seed0_tl:   list       = []
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
                    ds, tr_idx, va_idx, active, device,
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
                    seed0_tl.append((tl, vl))
                tracker.add_fold(ckpt_preds)
            seed_f1 = float(f1_score(seed_y_true, seed_y_pred, average="macro", zero_division=0))
            seed_f1s.append(seed_f1)
            seed_rows.append({"ablation": tag, "seed": seed, "seed_f1": round(seed_f1, 4)})
            tracker.commit_seed()

        means, stds, best_ep = tracker.summary(label=tag)
        mf1 = means[best_ep]
        sf1 = stds.get(best_ep, 0.0)
        print(f"{tag:25s}  F1={mf1:.3f}±{sf1:.3f}  best@ep{best_ep:03d}")
        for ep, m in means.items():
            epoch_rows.append({"ablation": tag, "epoch": ep, "f1_mean": round(m, 4),
                                "f1_std": round(stds.get(ep, 0.0), 4)})

        if mf1 > best_mean_f1:
            best_mean_f1 = mf1
            curves_best  = {"tag": tag, "folds": seed0_tl}

    # ── Results tables ────────────────────────────────────────────────────────
    df_res  = pd.DataFrame(result_rows)
    df_res.to_csv(out_dir / "results_attn.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_dir / "results_attn_per_seed.csv", index=False)

    summary = (df_res.groupby("ablation")[["val_f1","val_acc"]]
               .agg(["mean","std"]).round(4))
    summary.columns = ["f1_mean","f1_std","acc_mean","acc_std"]
    summary = summary.sort_values("f1_mean", ascending=False)
    summary.to_csv(out_dir / "summary_attn.csv")
    pd.DataFrame(epoch_rows).to_csv(out_dir / "results_attn_by_epoch.csv", index=False)

    print(f"\n=== Summary (sorted by F1) ===")
    print(summary.to_string())

    # ── Training curves ───────────────────────────────────────────────────────
    if curves_best:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for fi, (tl, vl) in enumerate(curves_best["folds"]):
            axes[0].plot(tl, alpha=0.7, label=f"Fold {fi}")
            if vl:
                axes[1].plot(vl, alpha=0.7, label=f"Fold {fi}")
        for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
            ax.set_xlabel("Epoch"); ax.set_ylabel("CrossEntropy")
            ax.set_title(f"{curves_best['tag']} — {title}")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        plt.suptitle(
            f"Attention Fusion — best: {curves_best['tag']}  "
            f"(mean F1={best_mean_f1:.3f})",
            fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(out_dir / "curves_best.png", dpi=150)
        plt.close()
        print(f"Saved: {(out_dir/'curves_best.png').relative_to(ROOT)}")

    print(f"\nOutputs: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=["autumn", "spring"], default=None)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    main(season=args.season, n_seeds=args.seeds)
