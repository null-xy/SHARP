#!/usr/bin/env python3
"""
run_pipeline.py

Orchestrates the analysis pipeline in the order the paper's results depend
on each other. This replaces the numeric-filename-prefix convention that
the code used before it was reorganized into the `sharp` package (`01_`,
`07_`, `15_`, ... -> descriptive module names under `src/sharp/`) with an
explicit, single place that documents (and can run) that order.

Requires `processed_data/` to already exist -- see the top-level README's
"Data availability" section for how to obtain or build it, and
`sharp.processing.build_processed_data` for the (non-runnable-without-raw-
data) stage that builds it from `raw_data/`.

Usage:
    python run_pipeline.py                 # run every stage, in order
    python run_pipeline.py --stage stats    # run only one stage
    python run_pipeline.py --list           # print the stage plan and exit
    python run_pipeline.py --dry-run        # print the commands without running them

Each stage is run as `python -m <module>` in a subprocess, exactly as
described in the README, so this script and the README can never drift
out of sync with what actually gets executed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# (stage name, [module invocations in that stage])
# "models" is the expensive step (trains 4 model families x up to 7
# modality-ablations x 10 seeds x 5 LOGO folds each); everything after it
# only reads the checkpoints those runs save to processed_data/analysis/*/checkpoints/.
STAGES: list[tuple[str, list[list[str]]]] = [
    ("features", [
        ["sharp.features.build_feature_matrix"],
        ["sharp.features.gaze_transitions"],
    ]),
    ("stats", [
        ["sharp.stats.eda_event_analysis"],
        ["sharp.stats.event_synchrony"],
        ["sharp.stats.normality_audit"],
        ["sharp.stats.discordance"],
        ["sharp.stats.correlation_kruskal"],
    ]),
    ("models", [
        ["sharp.models.svm_baseline"],
        ["sharp.models.early_fusion"],
        ["sharp.models.cross_modal_attention", "--season", "autumn"],
        ["sharp.models.gnn_fusion", "--season", "autumn"],
    ]),
    ("evaluation", [
        ["sharp.evaluation.epoch100_metrics"],
        ["sharp.evaluation.modality_ablation_metrics"],
        ["sharp.evaluation.confusion_matrix"],
        ["sharp.evaluation.attention_heatmap", "--season", "autumn"],
    ]),
    ("figures", [
        ["sharp.figures.figure1_signals"],
        ["sharp.figures.figure2_crossmodal_evidence"],
        ["sharp.figures.figure5_attention_weights"],
    ]),
]


def iter_commands(stage_filter: str | None):
    for stage, commands in STAGES:
        if stage_filter and stage != stage_filter:
            continue
        for args in commands:
            yield stage, [sys.executable, "-m", *args]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=[s for s, _ in STAGES], default=None,
                     help="Run only this stage instead of the full pipeline.")
    ap.add_argument("--list", action="store_true",
                     help="Print the stage/command plan and exit.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print each command instead of running it.")
    args = ap.parse_args()

    commands = list(iter_commands(args.stage))
    if not commands:
        print(f"No stage named {args.stage!r}. Choices: {[s for s, _ in STAGES]}")
        sys.exit(1)

    if args.list or args.dry_run:
        for stage, cmd in commands:
            print(f"[{stage}] {' '.join(cmd)}")
        if args.list:
            return

    if args.dry_run:
        return

    for stage, cmd in commands:
        print(f"\n=== [{stage}] {' '.join(cmd)} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStage '{stage}' failed ({' '.join(cmd)}), stopping.")
            sys.exit(result.returncode)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
