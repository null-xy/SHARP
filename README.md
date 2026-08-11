<div align="center">

# SHARP

### Multimodal Analysis for Socially Shared Regulation in Collaborative Learning

*Code release — 4DMR Workshop, IJCAI-ECAI 2026 (under review)*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/paper-under%20review-lightgrey)]()

</div>

## Overview

This repository contains the analysis and modeling code for our study of
**Socially Shared Regulation of Learning (SSRL)**: episodes where a small
group of students collaboratively regulates their cognition, motivation,
or emotion during a task. Using the MSSRL dataset, we combine three
synchronized modalities recorded around each annotated SSRL event —
**electrodermal activity (EDA)**, **facial affect**, and **eye gaze** —
to (1) statistically characterize how these signals diverge across event
types, and (2) benchmark several multimodal fusion strategies for
classifying each event as **Negative** (socio-emotional conflict),
**Positive** (collaborative coordination), or **Regulate** (explicit
joint regulation).

This code produces every table and figure in the paper. The underlying
sensor dataset is not redistributed here — see
[Data availability](#data-availability).

## Results

Fusion-strategy comparison on the retained n=29 event subset
(Leave-One-Group-Out CV, pooled out-of-fold predictions, 10-seed mean ± std):

| Model | Macro-F1 | Accuracy |
|---|:---:|:---:|
| SVM (hand-crafted features) | 0.548 | 0.690 |
| Early Fusion | 0.347 ± 0.090 | 0.466 ± 0.152 |
| GNN2-Mean | 0.375 ± 0.070 | 0.576 ± 0.085 |
| Participant-Attn | 0.423 ± 0.044 | 0.586 ± 0.071 |
| GNN1-Mean | 0.440 ± 0.098 | 0.590 ± 0.095 |
| Serial-GNN-Attn | 0.465 ± 0.082 | 0.614 ± 0.089 |
| Parallel-GNN-Attn | 0.453 ± 0.068 | 0.624 ± 0.055 |
| **Cross-Modal Attention** (ours) | **0.592 ± 0.080** | **0.738 ± 0.077** |

See the paper for the full unimodal and modality-combination ablations
(Tables 2–3), the confusion matrix (Figure 4), and the cross-modal
attention interpretability analysis (Figure 5).

## Installation

```bash
git clone https://github.com/null-xy/SHARP.git
cd SHARP
pip install -e .
```

Requires **Python 3.9+**. `requirements.txt` lists the same pinned
dependency versions if you prefer `pip install -r requirements.txt`
instead of installing `sharp` as a package.

## Repository structure

```
src/sharp/
├── config.py, utils.py, dataset.py   # paths/constants, shared helpers, PyTorch Dataset
├── processing/    # raw_data/ -> processed_data/ ETL (needs raw data, see below)
├── features/      # event-level feature matrix + gaze features
├── stats/         # Section 3.4: EDA deltas, synchrony, normality-driven tests, discordance
├── models/        # Section 4: SVM, Early Fusion, Cross-Modal Attention, GNN fusion
├── evaluation/     # Section 5: metrics tables, confusion matrix, attention weights
└── figures/       # Figure 1, 2, 5 generation
run_pipeline.py     # runs (or lists) the full pipeline in order
```

## Reproducing the results

All deep-learning results use Leave-One-Group-Out cross-validation
(one fold per retained student group), pooled out-of-fold predictions,
10 random seeds, and a fixed 100-epoch training budget with no
checkpoint selection. Every module is runnable as `python -m <module>`;
`run_pipeline.py` runs (or just prints) the full sequence in the correct
order:

```bash
python run_pipeline.py            # run everything, in order
python run_pipeline.py --list     # print the command sequence without running it
python run_pipeline.py --stage models   # run one stage only
```

For example, training the paper's main model directly:

```bash
python -m sharp.models.cross_modal_attention --season autumn
```

Model training (`sharp.models.*`) is the expensive step; the evaluation
and figure modules reuse the checkpoints those runs save under
`processed_data/analysis/*/checkpoints/` rather than retraining.

## Data availability

The raw recordings (360° classroom video, audio, EDA, eye-gaze) come
from the MSSRL dataset, collected from minors (high-school students)
under an ethics-approved protocol and data use agreement. **Raw sensor
data is not included in this repository** and cannot be publicly
redistributed. `sharp.processing` is included for pipeline transparency
— to show exactly how `raw_data/` becomes `processed_data/` — not as
something runnable without access to the original dataset. Processed,
de-identified event-level data is available on reasonable request.

## Citation

The paper is currently under review. Citation details will be added
once it is published.
