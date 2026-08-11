# SHARP: Multimodal Analysis for Socially Shared Regulation in Collaborative Learning

Code release accompanying the paper *"Multimodal analysis for Socially
Shared Regulation in Collaborative Learning"* (submitted to the 4DMR
Workshop, IJCAI-ECAI 2026). This repository contains the analysis and
modeling code used to produce every table and figure in the paper. The
underlying multimodal sensor dataset (MSSRL) is not redistributed here;
see [Data availability](#data-availability) below.

The paper studies **Socially Shared Regulation of Learning (SSRL)**:
episodes where a small group of students collaboratively regulates their
cognition, motivation, or emotion during a task. Each annotated SSRL
event is labeled Negative (socio-emotional conflict), Positive
(collaborative coordination), or Regulate (explicit joint regulation).
We combine three synchronized modalities recorded around each event
(electrodermal activity, facial affect, and eye gaze) and compare
several fusion strategies for classifying the event's regulation type,
alongside a statistical characterization of how the three modalities
behave differently across classes.

## Repository layout

```
pyproject.toml, requirements.txt
run_pipeline.py             # runs (or lists) every stage below, in order
CODE_REQUIREMENTS.md        # detailed paper-section -> module -> output mapping
src/sharp/
├── config.py, utils.py, dataset.py   # shared configuration, helpers, PyTorch Dataset
├── processing/              # raw-data ETL package (only runnable with raw_data/, see below)
│   └── build_processed_data.py       # Stage 0 entry point: raw_data/ -> processed_data/
├── features/
│   ├── build_feature_matrix.py       # Stage 1: builds the event-level feature matrix
│   └── gaze_transitions.py           # gaze switching features
├── stats/                   # Section 3.4 statistical characterization
│   ├── eda_event_analysis.py         # EDA pre/post delta statistics
│   ├── event_synchrony.py            # inter-personal EDA covariation (pairwise correlation)
│   ├── normality_audit.py            # normality-driven test selection (t-test vs Wilcoxon)
│   ├── discordance.py                # joint EDA-facial change pattern (Fisher-Freeman-Halton)
│   └── correlation_kruskal.py        # feature-vs-class Kruskal-Wallis tests
├── models/                  # Section 4 benchmark models (Table 4)
│   ├── svm_baseline.py               # SVM baseline
│   ├── early_fusion.py               # Early Fusion baseline
│   ├── cross_modal_attention.py      # Cross-Modal Attention, the paper's main model
│   └── gnn_fusion.py                 # 5 graph-based fusion variants
├── evaluation/               # Section 5 metrics tables and confusion matrix/attention analysis
│   ├── epoch100_metrics.py           # Precision/Recall/Accuracy for Table 4
│   ├── modality_ablation_metrics.py  # Precision/Recall/Accuracy for Tables 2 and 3
│   ├── confusion_matrix.py           # Figure 4
│   └── attention_heatmap.py          # attention-weight extraction for Figure 5
└── figures/
    ├── figure1_signals.py
    ├── figure2_crossmodal_evidence.py
    └── figure5_attention_weights.py
```

This package replaces an earlier, internal version of this code that was
organized as a flat directory of numbered scripts (`01_...py`, `07_...py`,
`15_...py`, ...) reflecting this project's internal pipeline ordering —
common in research codebases, but not a real Python package, and it
required file-path-based `importlib` hacks anywhere one numbered script
needed to import from another (`import` statements can't start with a
digit). Every module here now has a plain descriptive name and is
imported normally (`from sharp.models.cross_modal_attention import
...`); `run_pipeline.py` and the table below are where the *order*
information that used to live in the filenames now lives.
`CODE_REQUIREMENTS.md` documents, section by section, exactly which
module produced which paper result, and what was intentionally left out
of this release and why (it still refers to the old numbered names in
its change history, since that's what was true when it was written).

## Setup

```bash
pip install -e .
```

This installs the `sharp` package (editable, from `src/`) along with its
pinned dependencies from `pyproject.toml`. `requirements.txt` lists the
same pinned versions if you'd rather `pip install -r requirements.txt`
into an environment you manage yourself without installing `sharp` as a
package.

Python 3.9+ is required (the code uses `from __future__ import
annotations` and PEP 604-style type hints; `sharp.processing` additionally
uses the PEP 584 `dict | dict` merge operator, which needs 3.9+ even at
runtime).

## Data availability

The raw multimodal recordings (360-degree classroom video, audio,
electrodermal activity, eye-gaze) come from the MSSRL dataset, collected
from minors (high-school students) under an ethics-approved protocol and
data use agreement. **Raw sensor data is not included in this
repository** and cannot be publicly redistributed. `sharp.processing`
(and its entry point, `sharp.processing.build_processed_data`) is
included for pipeline transparency — to show exactly how `raw_data/` was
turned into `processed_data/` — not as something you can run without
access to the original dataset. Everything downstream of
`processed_data/` (feature extraction, statistics, model training) is
runnable once you have that directory in place at the repository root
(as a sibling of `src/`), either by obtaining the raw dataset and running
the pipeline yourself, or by requesting the processed, de-identified
event-level outputs directly from the authors.

## Reproducing the paper's results

All deep-learning results use **Leave-One-Group-Out (LOGO) cross-validation**
(one fold per retained student group), **pooled out-of-fold predictions**
across folds before computing metrics, and are averaged over **10 random
seeds** with a fixed 100-epoch training budget (no checkpoint selection).
See `CODE_REQUIREMENTS.md` for the exact module that produced each table
and figure. Every module can be run as `python -m <module>`; the full
sequence is:

```bash
python run_pipeline.py            # runs every stage below, in order
python run_pipeline.py --list     # print the same command sequence without running it
python run_pipeline.py --stage models   # run just one stage (features/stats/models/evaluation/figures)
```

which is equivalent to, in order:

```bash
# Statistical characterization (Section 3.4)
python -m sharp.stats.eda_event_analysis
python -m sharp.stats.event_synchrony
python -m sharp.stats.normality_audit
python -m sharp.stats.discordance
python -m sharp.stats.correlation_kruskal

# Benchmark models (Section 4, Tables 2-4) -- the expensive step
python -m sharp.models.svm_baseline
python -m sharp.models.early_fusion
python -m sharp.models.cross_modal_attention --season autumn   # main model
python -m sharp.models.gnn_fusion --season autumn              # 5 graph-based variants

# Metrics tables and figures (Section 5), reused from the checkpoints saved above
python -m sharp.evaluation.epoch100_metrics
python -m sharp.evaluation.modality_ablation_metrics
python -m sharp.evaluation.confusion_matrix       # Figure 4
python -m sharp.evaluation.attention_heatmap --season autumn   # attention weights for Figure 5
python -m sharp.figures.figure1_signals
python -m sharp.figures.figure2_crossmodal_evidence
python -m sharp.figures.figure5_attention_weights
```

Model training (`sharp.models.*`) is the expensive step; the evaluation
and figure modules reuse the checkpoints those runs save to
`processed_data/analysis/*/checkpoints/` rather than retraining.

## Citation

The paper is currently under review. Citation details will be added once
it is published.
