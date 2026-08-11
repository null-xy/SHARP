from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATE_KEY_FROM_ISO, EDA_DIR_RE, EDA_HW_RE, EDA_ROOT


@dataclass
class TranscriptSource:
    session_id: str
    source_bucket: str
    source_path: Path


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_dir_session_info(dirname: str) -> tuple[str, str, str]:
    match = EDA_DIR_RE.search(dirname or "")
    if not match:
        return "", "", ""
    date_iso = match.group("date")
    room = match.group("room")
    room_label = f"Room{room}" if room else ""
    return date_iso, DATE_KEY_FROM_ISO.get(date_iso, ""), room_label


def normalize_date_key(value: str) -> str:
    match = re.fullmatch(r"(Oct|May)_(0?)(\d+)", value)
    if not match:
        return value
    return f"{match.group(1)}_{int(match.group(3))}"


def parse_hardware_id(filename: str) -> str:
    match = EDA_HW_RE.search(filename or "")
    return match.group("hardware").upper() if match else ""


def format_ms_as_utc_text(value: object) -> str:
    if pd.isna(value) or value == "":
        return ""
    ts = pd.to_datetime(float(value), unit="ms", utc=True)
    return ts.isoformat()


def extract_windows_basename(path_text: str) -> str:
    return (path_text or "").replace("\\", "/").rstrip("/").split("/")[-1]


def eda_raw_root_for_date(date_iso: str) -> Path:
    return EDA_ROOT / "SHARP_EDA_RawFiles" / ("SharpAutumn EDA" if date_iso.startswith("2021-10") else "SharpSpring EDA")


def normalize_header(value: object) -> str:
    text = str(value).strip().replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_device_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_timecode_to_seconds(value: object, fps: float = 30.0) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, time):
        # Heuristic: Excel often parses MM:SS as HH:MM:SS (e.g. 07:59 -> 07:59:00)
        # In this project, video offsets are usually < 1 hour.
        # If hour > 0 and second == 0, it's very likely MM:SS was parsed as HH:MM.
        if value.hour > 0 and value.second == 0:
            return float(value.hour * 60 + value.minute)
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "not played"}:
        return None
    text = text.replace(";", ":")
    text = re.sub(r"\s*\(.*?\)\s*", "", text)
    text = re.sub(r"(?<=\d):\.(?=\d)", ":", text)
    text = re.sub(r"(?<=:)\.(?=\d)", "", text)
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", text):
        hh, mm, ss = [int(part) for part in text.split(":")]
        # Apply the same heuristic for string-parsed times
        if hh > 0 and ss == 0:
            return float(hh * 60 + mm)
        return float(hh * 3600 + mm * 60 + ss)
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}:\d{1,2}", text):
        hh, mm, ss, ff = [int(part) for part in text.split(":")]
        return hh * 3600 + mm * 60 + ss + ff / fps
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}\.\d{1,2}", text):
        hh, mm, ss = text.split(":")
        sec_text, frame_text = ss.split(".")
        return int(hh) * 3600 + int(mm) * 60 + int(sec_text) + int(frame_text) / fps
    parsed = pd.to_timedelta(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.total_seconds()
    return None


from datetime import datetime
from zoneinfo import ZoneInfo
HOST_TIMEZONE = ZoneInfo("Europe/Helsinki")


def wall_clock_to_unix_ms(date_iso: str, time_str: object) -> int | str:
    """date_iso (YYYY-MM-DD) + HH:MM:SS wall-clock string → unix ms (Helsinki TZ)."""
    if not date_iso or pd.isna(time_str) or not str(time_str).strip():
        return ""
    text = str(time_str).strip().split("(")[0].strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    else:
        return ""
    try:
        y, m, d = (int(x) for x in date_iso.split("-"))
        dt = datetime(y, m, d, t.hour, t.minute, t.second, tzinfo=HOST_TIMEZONE)
        return int(dt.timestamp() * 1000)
    except (ValueError, OSError):
        return ""


def parse_realtime_timestamp(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    
    # Try parsing as naive datetime then localizing
    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        # If it already has timezone info, just convert to UTC
        if parsed.tzinfo is not None:
            return parsed.tz_convert("UTC")
        # Otherwise, assume it is Helsinki time and localize
        return parsed.tz_localize(HOST_TIMEZONE).tz_convert("UTC")
    except Exception:
        return None


def parse_host_sheet_date_room(value: str) -> tuple[str, str]:
    room_match = re.search(r"Room\s*(\d)", value, flags=re.IGNORECASE)
    date_match = re.search(r"(\d{1,2})\.(\d{1,2})", value)
    if not room_match or not date_match:
        return "", ""
    day = int(date_match.group(1))
    month = int(date_match.group(2))
    return f"2021-{month:02d}-{day:02d}", f"Room{room_match.group(1)}"
