"""
sharp.config — SHARP project configuration

All paths and analysis constants are defined here in one place. To change
a path or a parameter, edit only this file.

Usage:
    from sharp.config import ROOT, FM_PATH, WINDOW_SEC, TARGET_LABELS, ...
    from sharp.utils import load_feature_matrix, load_vmap, ...   # helper functions
"""
from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project root
#
# This file lives at <repo_root>/src/sharp/config.py, so the repo root is
# three levels up: config.py -> sharp/ -> src/ -> <repo_root>. `raw_data/`
# and `processed_data/` are expected as siblings of `src/` at that root.
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Raw data paths (raw_data/) — NOT included in this repository, see README
# ─────────────────────────────────────────────────────────────────────────────
RAW_ROOT = ROOT / "raw_data"

# EDA
EDA_RAW_ROOT       = RAW_ROOT / "SHARP_EDA"
EDA_COMBINED_CSV   = {
    "autumn": EDA_RAW_ROOT / "CombinedEDA" / "syncedcombinedallSHARPAutumn.csv",
    "spring": EDA_RAW_ROOT / "CombinedEDA" / "syncedcombinedallSHARPSpring.csv",
}
EDA_RAW_FILES_ROOT = EDA_RAW_ROOT / "SHARP_EDA_RawFiles"

# Host Sheets (device <-> participant <-> color mapping)
HOST_SHEETS_ROOT = RAW_ROOT / "SHARP_Host Sheets - Timepoints"
HOST_SHEETS = {
    "autumn": HOST_SHEETS_ROOT / "Host Sheets SHARP Autumn.xlsx",
    "spring": HOST_SHEETS_ROOT / "Host Sheets SHARP Spring.xlsx",
}

# Trigger timepoints (CT/ET, HH:MM:SS format)
EDUCATION_ROOT = RAW_ROOT / "education"
TIMEPOINT_XLSX = {
    "autumn": EDUCATION_ROOT / "0 important tables" / "time point Oct.xlsx",
    "spring": EDUCATION_ROOT / "0 important tables" / "time point May.xlsx",
}

# EmoNet trigger frame numbers (each participant's CT/ET frame index, 30 fps)
TRIGGER_ROOT = {
    "autumn": EDUCATION_ROOT / "2022 02 combined emotion 1s" / "Oct_Combined" / "autumn",
    "spring": EDUCATION_ROOT / "2022 02 combined emotion 1s" / "May_Combined" / "spring",
}

# EmoNet per-second raw xlsx (per group/participant)
EMOTION_1S_RAW_ROOT = {
    "autumn": EDUCATION_ROOT / "2022 02 combined emotion 1s" / "Oct_Combined" / "autumn 4 emotion 1s",
    "spring": EDUCATION_ROOT / "2022 02 combined emotion 1s" / "May_Combined" / "spring emotion4 1s",
}

# SSRL event annotations (speaker x type x start/end time)
ANNO_PATH = (
    EDUCATION_ROOT / "0 important tables" / "audio" / "SHARP_QualitativeCode_Socio-emo.xlsx"
)

# Gaze data (GAZE_XLSX = raw xlsx; GAZE_PATH = processed events CSV)
GAZE_XLSX = RAW_ROOT / "Final Gaze data_SHARP - to share.xlsx"

# Transcripts
TRANSCRIPTION_ROOT = RAW_ROOT / "SHARP_Transcription"

# Self-report surveys
SELFREPORTS_ROOT = RAW_ROOT / "SHARP_Selfreports" / "en"

# Kinect skeleton (trigger-window velocity)
KINECT_TRIGGER_ROOT = {
    "autumn": EDUCATION_ROOT / "Kinect_trigger" / "Oct_trigger_skeleton",
    "spring": EDUCATION_ROOT / "Kinect_trigger" / "May_trigger_skeleton",
}

# ─────────────────────────────────────────────────────────────────────────────
# Processed data paths (processed_data/)
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_ROOT = ROOT / "processed_data"

# EDA
VMAP_PATH            = PROCESSED_ROOT / "eda" / "video_timing_map.csv"
SYNC_MAP_PATH        = PROCESSED_ROOT / "eda" / "sync_timing_map.csv"
TRIGGER_FRAMES_PATH  = PROCESSED_ROOT / "eda" / "trigger_emotion_frames.csv"
TRIGGER_OCT_PATH     = PROCESSED_ROOT / "eda" / "trigger_timepoints_oct.csv"
EDA_FILT_DIR         = PROCESSED_ROOT / "eda" / "filtered"
EDA_SCR_DIR          = PROCESSED_ROOT / "eda" / "scr"

# Facial affect (EmoNet, per-second CSV)
EMOTION_DIR = PROCESSED_ROOT / "education" / "emotion"

# Gaze
GAZE_PATH = PROCESSED_ROOT / "gaze" / "gaze_events.csv"

# Transcripts
UTT_PATH = PROCESSED_ROOT / "transcription" / "all_utterances.csv"

# Analysis outputs
ANALYSIS_DIR    = PROCESSED_ROOT / "analysis"
FM_PATH         = ANALYSIS_DIR / "feature_matrix.csv"
CORR_DIR        = ANALYSIS_DIR / "correlation"
SURVEY_DIR      = ANALYSIS_DIR / "survey"

# ─────────────────────────────────────────────────────────────────────────────
# Analysis parameters
# ─────────────────────────────────────────────────────────────────────────────

# Pre/post event window, in seconds (used by scripts 11/23/24/25/27 etc.)
WINDOW_SEC = 30

# Minimum valid points for a per-second coupling Pearson r
MIN_COUPLING_PAIRS = 10

# EmoNet video frame rate
TRIGGER_FPS = 30

# Participant IDs
PIDS = ["P1", "P2", "P3"]

# ─────────────────────────────────────────────────────────────────────────────
# SSRL class labels (must match the `deliberation` column in feature_matrix.csv
# and the SSRL annotation xlsx exactly)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_POS = "Positive socioemotional interaction"
CLASS_NEG = "Negative socioemotional interaction"
CLASS_REG = "Regulate group emo-mo"

# The 3 target classes used for classification (fixed order, for sklearn's
# LabelEncoder etc.)
TARGET_LABELS = [CLASS_NEG, CLASS_POS, CLASS_REG]

# Short name -> full name
CLASS_SHORT = {
    "Positive": CLASS_POS,
    "Negative": CLASS_NEG,
    "Regulate": CLASS_REG,
}
CLASS_SHORT_INV = {v: k for k, v in CLASS_SHORT.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Date-key mapping (ISO date <-> human-readable string <-> group ID)
# ─────────────────────────────────────────────────────────────────────────────
AUTUMN_DATE_KEYS = {
    "2021-10-05": "Oct_5",
    "2021-10-07": "Oct_7",
    "2021-10-08": "Oct_8",
    "2021-10-15": "Oct_15",
    "2021-10-21": "Oct_21",
    "2021-10-22": "Oct_22",
}
SPRING_DATE_KEYS = {
    "2021-05-07": "May_7",
    "2021-05-10": "May_10",
    "2021-05-14": "May_14",
    "2021-05-17": "May_17",
}
DATE_KEYS = {**AUTUMN_DATE_KEYS, **SPRING_DATE_KEYS}

# date_str -> group_project (Room 1 only)
DATE_TO_GROUP_R1 = {
    "Oct_5":  "D1G1",
    "Oct_7":  "D2G1",
    "Oct_8":  "D3G1",
    "Oct_15": "D4G1",
    "Oct_21": "D5G1",
    "Oct_22": "D6G1",
    "May_7":  "SD1G1",
    "May_10": "SD2G1",
    "May_14": "SD3G1",
    "May_17": "SD4G1",
}
GROUP_TO_DATE_R1 = {v: k for k, v in DATE_TO_GROUP_R1.items()}
