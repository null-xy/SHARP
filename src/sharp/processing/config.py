from __future__ import annotations

import re
from pathlib import Path

# config.py -> processing/ -> sharp/ -> src/ -> <repo_root>
ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "raw_data"
PROCESSED_ROOT = ROOT / "processed_data"

EDA_ROOT = RAW_ROOT / "SHARP_EDA"
TRANSCRIPTION_ROOT = RAW_ROOT / "SHARP_Transcription"
EDUCATION_ROOT = RAW_ROOT / "education"
HOST_SHEETS_ROOT = RAW_ROOT / "SHARP_Host Sheets - Timepoints"
GAZE_XLSX = RAW_ROOT / "Final Gaze data_SHARP - to share.xlsx"
PHYSIOLOGY_ROOT = EDUCATION_ROOT / "0 important tables" / "PhysiologicalData" / "PhysiologicalData"
TIMEPOINT_OCT_XLSX = EDUCATION_ROOT / "0 important tables" / "time point Oct.xlsx"
TIMING_AUDIOSTART_XLSX = HOST_SHEETS_ROOT / "AudioStarttime.xlsx"
TRIGGER_AUTUMN_ROOT = EDUCATION_ROOT / "2022 02 combined emotion 1s" / "Oct_Combined" / "autumn"
TRIGGER_SPRING_ROOT = EDUCATION_ROOT / "2022 02 combined emotion 1s" / "May_Combined" / "spring"
TRIGGER_FPS = 30

TRANSCRIPT_SOURCE_PRIORITY = [
    "uploaded",
    "uploaded 111020",
    "unsorted",
]

SRT_BLOCK_RE = re.compile(
    r"^\s*(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})\s*$"
)
LANG_TAG_RE = re.compile(r"^\[(?P<lang>[^\]]+)\]\s*")
EDA_DIR_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})(?:_Room(?P<room>\d))?", re.IGNORECASE)
EDA_HW_RE = re.compile(r"Shimmer_(?P<hardware>[A-Z0-9]+)_", re.IGNORECASE)
EMOTION_ORIGIN_RE = re.compile(
    r"(?:(?P<date>(?:Oct|May)_?\d{1,2}))?"
    r"(?P<pid>P\d)"
    r"(?:_(?:CT|ET\d))?.*?"
    r"origin_(?P<origin>\d+)_",
    re.IGNORECASE
)

AUTUMN_DATE_KEY_MAP = {
    "2021-10-05": "Oct_5",
    "2021-10-07": "Oct_7",
    "2021-10-08": "Oct_8",
    "2021-10-15": "Oct_15",
    "2021-10-21": "Oct_21",
    "2021-10-22": "Oct_22",
}
SPRING_DATE_KEY_MAP = {
    "2021-05-07": "May_7",
    "2021-05-10": "May_10",
    "2021-05-11": "May_14",  # Spring EDA dirname uses 2021-05-11 for what host sheet calls May_14
    "2021-05-14": "May_14",
    "2021-05-17": "May_17",
}
DATE_KEY_FROM_ISO = AUTUMN_DATE_KEY_MAP | SPRING_DATE_KEY_MAP
