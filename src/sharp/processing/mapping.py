from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from .common import (
    normalize_date_key,
    normalize_device_code,
    normalize_header,
    parse_host_sheet_date_room,
    parse_realtime_timestamp,
    parse_timecode_to_seconds,
)
from .config import (
    DATE_KEY_FROM_ISO,
    EDUCATION_ROOT,
    EMOTION_ORIGIN_RE,
    HOST_SHEETS_ROOT,
    PHYSIOLOGY_ROOT,
    PROCESSED_ROOT,
    RAW_ROOT,
    ROOT,
    TIMEPOINT_OCT_XLSX,
    TIMING_AUDIOSTART_XLSX,
    TRIGGER_AUTUMN_ROOT,
    TRIGGER_SPRING_ROOT,
    TRIGGER_FPS,
)


def build_host_device_schedule() -> list[dict]:
    rows: list[dict] = []
    workbooks = [
        ("autumn", HOST_SHEETS_ROOT / "Host Sheets SHARP Autumn.xlsx"),
        ("spring", HOST_SHEETS_ROOT / "Host Sheets SHARP Spring.xlsx"),
    ]
    for season, workbook in workbooks:
        if not workbook.exists():
            continue
        excel = pd.ExcelFile(workbook)
        for sheet_name in excel.sheet_names:
            if sheet_name == "Sync":
                continue
            raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
            if len(raw) < 4:
                continue
            header_row = raw.iloc[2].tolist()
            header_map = {normalize_header(col): idx for idx, col in enumerate(header_row)}
            current_date_room = ""
            for _, row in raw.iloc[3:].iterrows():
                first_value = row.iloc[0]
                if pd.notna(first_value) and str(first_value).strip():
                    current_date_room = str(first_value).strip()
                if not current_date_room or not any(ch.isdigit() for ch in current_date_room):
                    continue
                date_iso, room_label = parse_host_sheet_date_room(current_date_room)
                if not date_iso or not room_label:
                    continue
                date_str = DATE_KEY_FROM_ISO.get(date_iso, "")

                def cell(name: str) -> str:
                    idx = header_map.get(name)
                    if idx is None:
                        return ""
                    value = row.iloc[idx]
                    if pd.isna(value):
                        return ""
                    return str(value).strip()

                device_code = cell("shimmer device code")
                participant_identification = cell("participant identification")
                all_videos = cell("all videos") or cell("all videos ")
                if not device_code and not participant_identification and not all_videos:
                    continue
                rows.append(
                    {
                        "season": season,
                        "date_iso": date_iso,
                        "date_str": date_str,
                        "room_label": room_label,
                        "host_sheet": sheet_name,
                        "date_room_condition": current_date_room,
                        "name": cell("name"),
                        "pseudonym": cell("pseudonym") or cell("pseudonyms"),
                        "path": cell("path"),
                        "shimmer_device_code": device_code,
                        "participant_identification": participant_identification,
                        "all_videos": all_videos,
                        "task_start_time": cell("task start time"),
                        "start_video": cell("start video"),
                        "start_video_time": cell("start video time"),
                        "start_video_time_secs": cell("start video time secs"),
                        "time_ct": cell("time ct"),
                        "ct_video": cell("ct video"),
                        "ct_video_time": cell("ct video time"),
                        "ct_time_secs": cell("ct time secs"),
                        "ct_secs_since_task_start_time": cell("ct secs since task start time"),
                        "time_et1": cell("time et1"),
                        "et1_video": cell("et1 video"),
                        "et1_video_time": cell("et1 video time"),
                        "et1_video_time_secs": cell("et1 video time secs"),
                        "et1_time_since_task_start_time": cell("et1 time since task start time"),
                        "time_et2": cell("time et2"),
                        "et2_video": cell("et2 video"),
                        "et2_video_time": cell("et2 video time"),
                        "et2_video_time_secs": cell("et2 video time secs"),
                        "et2_time_since_task_start_time": cell("et2 time since task start time"),
                        "time_et3": cell("time et3"),
                        "et3_video": cell("et3 video"),
                        "et3_video_time": cell("et3 video time"),
                        "et3_video_time_secs": cell("et3 video time secs"),
                        "et3_time_since_task_start_time": cell("et3 time since task start task"),
                        "task_end_time": cell("task end time"),
                        "end_video": cell("end video"),
                        "end_video_time": cell("end video time"),
                    }
                )
    return rows


def build_emotion_origin_map() -> list[dict]:
    origin_rows: dict[tuple[str, str], dict] = {}
    base = EDUCATION_ROOT / "manual label accuracy and distribution" / "predidted trigger emotions"
    if not base.exists():
        return []
    for path in base.rglob("*.txt"):
        if path.name.startswith("~$"):
            continue
        
        match = EMOTION_ORIGIN_RE.search(path.name)
        if not match:
            continue

        # 1. Try to find date in parent folders (e.g. May_10/)
        date_str = ""
        for part in path.parts:
            if part.startswith(("Oct_", "May_")):
                date_str = normalize_date_key(part)
                break
        
        # 2. If date not in folders, try to extract from filename match
        if not date_str and match.group("date"):
            raw_date = match.group("date")
            # If the filename has something like May_07P1, normalize it
            # First, ensure underscore before day if missing
            if "_" not in raw_date:
                raw_date = re.sub(r"(\d+)", r"_\1", raw_date)
            date_str = normalize_date_key(raw_date)

        if not date_str:
            continue

        pid = match.group("pid").upper()
        origin_id = match.group("origin")
        
        # KEY FIX: If date_str looks like May_07P1_CT (due to bad extraction),
        # force it to just the date part.
        if len(date_str) > 6: # e.g. "May_07" is 6 chars
            sub_match = re.search(r"((?:Oct|May)_\d{1,2})", date_str, re.IGNORECASE)
            if sub_match:
                date_str = normalize_date_key(sub_match.group(1))

        key = (date_str, pid)
        origin_rows.setdefault(
            key,
            {
                "date_str": date_str,
                "p_id": pid,
                "origin_id": origin_id,
                "origin_token": f"origin_{origin_id}_",
                "example_source_path": str(path.relative_to(ROOT)),
            },
        )
    return [origin_rows[key] for key in sorted(origin_rows)]


def build_device_id_records() -> tuple[list[dict], dict[tuple[str, str], str]]:
    rows: list[dict] = []
    global_hardware_map: dict[tuple[str, str], str] = {}
    workbooks = [
        ("autumn", PHYSIOLOGY_ROOT / "Device ID Records - Autumn.xlsx"),
        ("spring", PHYSIOLOGY_ROOT / "Device ID Records - Spring.xlsx"),
    ]
    for season, workbook in workbooks:
        if not workbook.exists():
            continue
        raw = pd.read_excel(workbook)
        current_day = ""
        current_room = ""
        for _, row in raw.iterrows():
            day_value = row.get("Day")
            if pd.notna(day_value):
                if isinstance(day_value, (datetime, pd.Timestamp)):
                    current_day = day_value.strftime("%Y-%m-%d")
                else:
                    parsed = pd.to_datetime(day_value, errors="coerce")
                    current_day = parsed.strftime("%Y-%m-%d") if pd.notna(parsed) else str(day_value).strip()
            room_value = row.get("Room")
            if pd.notna(room_value):
                room_text = normalize_device_code(room_value)
                current_room = f"Room{room_text}" if room_text else ""
            table_value = row.get("Table [Colour/Number]")
            device_code = ""
            participant_identification = ""
            hardware_id = ""
            if season == "autumn":
                device_code = normalize_device_code(table_value)
                hardware_id = normalize_device_code(row.get("Shimmer")).upper()
            else:
                participant_identification = normalize_device_code(table_value)
                device_code = normalize_device_code(row.get("Shimmer"))
                hardware_id = normalize_device_code(row.get("Shimmer ID")).upper()
            
            if hardware_id and device_code:
                global_hardware_map[(season, device_code)] = hardware_id
                
            if not device_code and not hardware_id and not participant_identification:
                continue
            rows.append(
                {
                    "season": season,
                    "date_iso": current_day,
                    "date_str": DATE_KEY_FROM_ISO.get(current_day, ""),
                    "room_label": current_room,
                    "participant_identification": participant_identification,
                    "shimmer_device_code": device_code,
                    "hardware_id": hardware_id,
                    "source_path": str(workbook.relative_to(ROOT)),
                }
            )
    supplemental_rows = build_trigger_video_device_rows()
    base_exact_index = {
        (row["season"], row["date_iso"], row["room_label"], str(row["shimmer_device_code"])): row for row in rows
    }
    base_room_device_index = {
        (row["season"], row["room_label"], str(row["shimmer_device_code"])): row for row in rows
    }
    enriched_rows: list[dict] = []
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    supplemented_room_device_keys: set[tuple[str, str, str]] = set()
    for row in supplemental_rows:
        exact_key = (row["season"], row["date_iso"], row["room_label"], str(row["shimmer_device_code"]))
        base_row = base_exact_index.get(exact_key) or base_room_device_index.get(
            (row["season"], row["room_label"], str(row["shimmer_device_code"]))
        )
        hardware_id = base_row["hardware_id"] if base_row else ""
        if not hardware_id:
            hardware_id = global_hardware_map.get((row["season"], str(row["shimmer_device_code"])), "")
            
        supplemented_room_device_keys.add((row["season"], row["room_label"], str(row["shimmer_device_code"])))
        merged = {
            "season": row["season"],
            "date_iso": row["date_iso"],
            "date_str": row["date_str"],
            "room_label": row["room_label"],
            "participant_identification": row["participant_identification"],
            "shimmer_device_code": row["shimmer_device_code"],
            "hardware_id": hardware_id,
            "source_path": row["source_path"],
        }
        dedupe_key = (
            merged["season"],
            merged["date_iso"],
            merged["room_label"],
            str(merged["shimmer_device_code"]),
            merged["participant_identification"],
        )
        if dedupe_key not in seen_keys:
            seen_keys.add(dedupe_key)
            enriched_rows.append(merged)
    for row in rows:
        room_device_key = (row["season"], row["room_label"], str(row["shimmer_device_code"]))
        if room_device_key in supplemented_room_device_keys:
            continue
        dedupe_key = (
            row["season"],
            row["date_iso"],
            row["room_label"],
            str(row["shimmer_device_code"]),
            row["participant_identification"],
        )
        if dedupe_key not in seen_keys:
            seen_keys.add(dedupe_key)
            enriched_rows.append(row)
    return sorted(
        enriched_rows,
        key=lambda row: (
            row.get("season", ""),
            row.get("date_iso", ""),
            row.get("room_label", ""),
            str(row.get("shimmer_device_code", "")),
        ),
    ), global_hardware_map


def build_trigger_video_device_rows() -> list[dict]:
    rows: list[dict] = []
    workbooks = [
        ("autumn", PHYSIOLOGY_ROOT / "SHARP triggers and video files Autumn.xlsx"),
        ("spring", PHYSIOLOGY_ROOT / "SHARP triggers and video files Spring.xlsx"),
    ]
    for season, workbook in workbooks:
        if not workbook.exists():
            continue
        excel = pd.ExcelFile(workbook)
        for sheet_name in excel.sheet_names:
            if normalize_header(sheet_name) == "sync":
                continue
            raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
            if len(raw) < 4:
                continue
            header_row = raw.iloc[2].tolist()
            header_map = {normalize_header(col): idx for idx, col in enumerate(header_row)}
            current_date_room = ""
            for _, row in raw.iloc[3:].iterrows():
                first_value = row.iloc[0]
                if pd.notna(first_value) and str(first_value).strip():
                    current_date_room = str(first_value).strip()
                if not current_date_room or not any(ch.isdigit() for ch in current_date_room):
                    continue
                date_iso, room_label = parse_host_sheet_date_room(current_date_room)
                if not date_iso or not room_label:
                    continue

                def cell(name: str) -> str:
                    idx = header_map.get(name)
                    if idx is None:
                        return ""
                    value = row.iloc[idx]
                    if pd.isna(value):
                        return ""
                    return str(value).strip()

                device_code = cell("shimmer device code")
                participant_identification = cell("participant identification")
                if not device_code:
                    continue
                rows.append(
                    {
                        "season": season,
                        "date_iso": date_iso,
                        "date_str": DATE_KEY_FROM_ISO.get(date_iso, ""),
                        "room_label": room_label,
                        "participant_identification": participant_identification,
                        "shimmer_device_code": device_code,
                        "source_path": str(workbook.relative_to(ROOT)),
                    }
                )
    return rows


def build_device_record_indexes(
    device_id_rows: list[dict],
) -> tuple[dict[tuple[str, str, str, str], dict], dict[tuple[str, str, str], list[dict]]]:
    device_exact_index = {
        (row["season"], row["date_iso"], row["room_label"], str(row["shimmer_device_code"])): row for row in device_id_rows
    }
    device_room_index: dict[tuple[str, str, str], list[dict]] = {}
    for row in device_id_rows:
        key = (row["season"], row["room_label"], str(row["shimmer_device_code"]))
        device_room_index.setdefault(key, []).append(row)
    return device_exact_index, device_room_index


def resolve_hardware_from_device_records(
    host_row: dict,
    device_exact_index: dict[tuple[str, str, str, str], dict],
    device_room_index: dict[tuple[str, str, str], list[dict]],
    global_hardware_map: dict[tuple[str, str], str],
) -> tuple[str, str, str]:
    hardware_id = ""
    exact_key = (host_row["season"], host_row["date_iso"], host_row["room_label"], str(host_row["shimmer_device_code"]))
    exact_device_row = device_exact_index.get(exact_key)
    if exact_device_row and exact_device_row.get("hardware_id"):
        hardware_id = exact_device_row["hardware_id"]
        return (
            hardware_id,
            "ok",
            f"device_code {host_row['shimmer_device_code']} -> {hardware_id} via exact device record {exact_device_row['date_iso']}",
        )
    
    room_matches = device_room_index.get((host_row["season"], host_row["room_label"], str(host_row["shimmer_device_code"])), [])
    dated_matches = [row for row in room_matches if row.get("date_iso")]
    parsed_host_date = pd.to_datetime(host_row["date_iso"], errors="coerce")
    closest_row = None
    closest_days = None
    for candidate in dated_matches:
        parsed_candidate_date = pd.to_datetime(candidate["date_iso"], errors="coerce")
        if pd.isna(parsed_host_date) or pd.isna(parsed_candidate_date):
            continue
        delta_days = abs((parsed_candidate_date - parsed_host_date).days)
        if closest_days is None or delta_days < closest_days:
            closest_days = delta_days
            closest_row = candidate
            
    if closest_row is not None and closest_days is not None and closest_days <= 1 and closest_row.get("hardware_id"):
        hardware_id = closest_row["hardware_id"]
        return (
            hardware_id,
            "fallback_nearest_date_within_1_day",
            f"device_code {host_row['shimmer_device_code']} -> {hardware_id} via nearest device record dated {closest_row['date_iso']} ({closest_days} day difference)",
        )

    hardware_id = global_hardware_map.get((host_row["season"], str(host_row["shimmer_device_code"])), "")
    if hardware_id:
        return (
            hardware_id,
            "fallback_global_season_device_map",
            f"device_code {host_row['shimmer_device_code']} -> {hardware_id} via global season/device mapping",
        )

    return (
        "",
        "no_device_record_match",
        f"{host_row['season']} / {host_row['date_iso']} / {host_row['room_label']} / device_code {host_row['shimmer_device_code']} had no device-record match",
    )


def build_identity_links(
    host_rows: list[dict],
    emotion_origin_rows: list[dict],
    device_id_rows: list[dict],
    global_hardware_map: dict[tuple[str, str], str],
) -> list[dict]:
    out: list[dict] = []
    device_exact_index, device_room_index = build_device_record_indexes(device_id_rows)
    for origin_row in emotion_origin_rows:
        candidates = [
            row
            for row in host_rows
            if row.get("date_str") == origin_row["date_str"] and origin_row["origin_token"] in row.get("all_videos", "")
        ]
        if not candidates:
            out.append(
                {
                    "date_str": origin_row["date_str"],
                    "date_iso": "",
                    "room_label": "",
                    "p_id": origin_row["p_id"],
                    "origin_id": origin_row["origin_id"],
                    "origin_token": origin_row["origin_token"],
                    "participant_identification": "",
                    "shimmer_device_code": "",
                    "name": "",
                    "time_ct": "",
                    "host_origin_match_count": 0,
                    "candidate_room_labels": "",
                    "candidate_participant_colors": "",
                    "join_check_status": "no_host_origin_match",
                    "join_check_detail": f"{origin_row['p_id']} -> {origin_row['origin_token']} had no host-sheet match",
                    "example_source_path": origin_row["example_source_path"],
                }
            )
            continue
        room1_candidates = [row for row in candidates if row.get("room_label") == "Room1"]
        host_row = sorted(room1_candidates if room1_candidates else candidates, key=lambda row: (row.get("room_label", ""), row.get("participant_identification", "")))[0]
        join_check_status = "ok"
        join_check_detail = f"{origin_row['p_id']} -> {origin_row['origin_token']} -> {host_row['participant_identification']} via {host_row['room_label']}"
        if len(candidates) > 1:
            join_check_status = "ambiguous_origin_match_room1_preferred" if room1_candidates else "ambiguous_origin_match_first_selected"
            join_check_detail = (
                f"{origin_row['p_id']} -> {origin_row['origin_token']} matched {len(candidates)} host rows; "
                f"selected {host_row['room_label']} / {host_row['participant_identification']}"
            )
        hardware_id, hardware_join_status, hardware_join_detail = resolve_hardware_from_device_records(host_row, device_exact_index, device_room_index, global_hardware_map)
        out.append(
            {
                "date_str": origin_row["date_str"],
                "date_iso": host_row["date_iso"],
                "room_label": host_row["room_label"],
                "p_id": origin_row["p_id"],
                "origin_id": origin_row["origin_id"],
                "origin_token": origin_row["origin_token"],
                "participant_identification": host_row["participant_identification"],
                "shimmer_device_code": host_row["shimmer_device_code"],
                "hardware_id": hardware_id,
                "name": host_row["name"],
                "time_ct": host_row["time_ct"],
                "host_origin_match_count": len(candidates),
                "candidate_room_labels": ",".join(sorted({row["room_label"] for row in candidates if row.get("room_label")})),
                "candidate_participant_colors": ",".join(sorted({row["participant_identification"] for row in candidates if row.get("participant_identification")})),
                "join_check_status": join_check_status,
                "join_check_detail": join_check_detail,
                "hardware_join_status": hardware_join_status,
                "hardware_join_detail": hardware_join_detail,
                "example_source_path": origin_row["example_source_path"],
            }
        )
    return out


def build_audio_start_records() -> list[dict]:
    rows: list[dict] = []
    workbook = TIMING_AUDIOSTART_XLSX
    if not workbook.exists():
        return rows
    spring = pd.read_excel(workbook, sheet_name="Spring", header=None)
    for idx, value in spring.iloc[:, 0].items():
        parsed = parse_realtime_timestamp(value)
        if parsed is None:
            continue
        date_iso = parsed.tz_convert("Europe/Helsinki").strftime("%Y-%m-%d")
        rows.append(
            {
                "season": "spring",
                "date_iso": date_iso,
                "date_str": DATE_KEY_FROM_ISO.get(date_iso, ""),
                "record_key": f"spring_row_{idx + 1}",
                "audio_start_realtime": parsed.isoformat(),
                "audio_start_unix_ms": int(parsed.timestamp() * 1000),
                "source_path": str(workbook.relative_to(ROOT)),
            }
        )
    autumn = pd.read_excel(workbook, sheet_name="Autumn", header=None)
    for idx, row in autumn.iterrows():
        label = "" if pd.isna(row.iloc[0]) else str(row.iloc[0]).strip()
        parsed = parse_realtime_timestamp(row.iloc[1] if len(row) > 1 else None)
        if parsed is None:
            continue
        date_iso = parsed.tz_convert("Europe/Helsinki").strftime("%Y-%m-%d")
        rows.append(
            {
                "season": "autumn",
                "date_iso": date_iso,
                "date_str": DATE_KEY_FROM_ISO.get(date_iso, ""),
                "record_key": label or f"autumn_row_{idx + 1}",
                "audio_start_realtime": parsed.isoformat(),
                "audio_start_unix_ms": int(parsed.timestamp() * 1000),
                "source_path": str(workbook.relative_to(ROOT)),
            }
        )
    return rows


def build_sync_timing_rows() -> list[dict]:
    rows: list[dict] = []
    workbooks = [
        ("autumn", HOST_SHEETS_ROOT / "Host Sheets SHARP Autumn.xlsx"),
        ("spring", HOST_SHEETS_ROOT / "Host Sheets SHARP Spring.xlsx"),
    ]
    for season, workbook in workbooks:
        if not workbook.exists():
            continue
        raw = pd.read_excel(workbook, sheet_name="Sync", header=None)
        if raw.empty:
            continue
        headers = [normalize_header(value) for value in raw.iloc[0].tolist()]
        header_map = {name: idx for idx, name in enumerate(headers) if name}
        # Use a dict so that if the sheet has an "UPDATED" block (Spring), the later
        # rows overwrite the earlier ones, keeping only the revised values.
        group_rows: dict[str, dict] = {}
        for _, row in raw.iloc[1:].iterrows():
            group_project = ""
            group_idx = header_map.get("group/project")
            if group_idx is not None and pd.notna(row.iloc[group_idx]):
                group_project = str(row.iloc[group_idx]).strip()
            if not group_project or group_project.lower() in ("updated", "group/project"):
                continue

            def cell(name: str) -> str:
                idx = header_map.get(name)
                if idx is None:
                    return ""
                value = row.iloc[idx]
                if pd.isna(value):
                    return ""
                return str(value).strip()

            audio_start_raw = cell("audio-starttime(realtime)")
            audio_start_ts = parse_realtime_timestamp(audio_start_raw)
            date_iso = ""
            if audio_start_ts is not None:
                date_iso = audio_start_ts.tz_convert("Europe/Helsinki").strftime("%Y-%m-%d")
            mic_audio_start_raw = cell("mic audio start time")
            cut_task_start_raw = cell("cut task video start time")
            cut_task_end_raw = cell("cut task video end time")
            mic_audio_start_seconds = parse_timecode_to_seconds(mic_audio_start_raw)
            cut_task_start_seconds = parse_timecode_to_seconds(cut_task_start_raw)
            cut_task_end_seconds = parse_timecode_to_seconds(cut_task_end_raw)
            raw_video_start_ts = None
            processed_video_start_ts = None
            processed_video_end_ts = None
            if audio_start_ts is not None and mic_audio_start_seconds is not None:
                raw_video_start_ts = audio_start_ts - pd.to_timedelta(mic_audio_start_seconds, unit="s")
            if audio_start_ts is not None and cut_task_start_seconds is not None:
                processed_video_start_ts = audio_start_ts + pd.to_timedelta(cut_task_start_seconds, unit="s")
            if audio_start_ts is not None and cut_task_end_seconds is not None:
                processed_video_end_ts = audio_start_ts + pd.to_timedelta(cut_task_end_seconds, unit="s")
            group_rows[group_project] = {
                    "season": season,
                    "group_project": group_project,
                    "date_label": cell("date"),
                    "date_iso": date_iso,
                    "date_str": DATE_KEY_FROM_ISO.get(date_iso, ""),
                    "video": cell("video"),
                    "audio_start_realtime": audio_start_ts.isoformat() if audio_start_ts is not None else audio_start_raw,
                    "audio_start_unix_ms": int(audio_start_ts.timestamp() * 1000) if audio_start_ts is not None else "",
                    "mic_audio_start_time": mic_audio_start_raw,
                    "mic_audio_start_seconds": "" if mic_audio_start_seconds is None else mic_audio_start_seconds,
                    "mic_audio_start_frame": cell("mic audio start frame"),
                    "cut_task_video_start_time": cut_task_start_raw,
                    "cut_task_video_start_seconds": "" if cut_task_start_seconds is None else cut_task_start_seconds,
                    "cut_task_video_start_frame": cell("cut task video start frame"),
                    "cut_task_video_end_time": cut_task_end_raw,
                    "cut_task_video_end_seconds": "" if cut_task_end_seconds is None else cut_task_end_seconds,
                    "cut_task_video_end_frame": cell("cut task video end frame"),
                    "raw_video_start_realtime": raw_video_start_ts.isoformat() if raw_video_start_ts is not None else "",
                    "raw_video_start_unix_ms": int(raw_video_start_ts.timestamp() * 1000) if raw_video_start_ts is not None else "",
                    "processed_video_start_realtime": processed_video_start_ts.isoformat() if processed_video_start_ts is not None else "",
                    "processed_video_start_unix_ms": int(processed_video_start_ts.timestamp() * 1000) if processed_video_start_ts is not None else "",
                    "processed_video_end_realtime": processed_video_end_ts.isoformat() if processed_video_end_ts is not None else "",
                    "processed_video_end_unix_ms": int(processed_video_end_ts.timestamp() * 1000) if processed_video_end_ts is not None else "",
                    "note": cell("note"),
                    "source_path": str(workbook.relative_to(ROOT)),
                }
        rows.extend(group_rows.values())
    return rows


def build_oct_timepoint_rows() -> list[dict]:
    rows: list[dict] = []
    workbook = TIMEPOINT_OCT_XLSX
    if not workbook.exists():
        return rows
    raw = pd.read_excel(workbook, sheet_name=0, header=None)
    for col in range(0, raw.shape[1], 4):
        chunk = raw.iloc[:, col : col + 4]
        if chunk.shape[1] < 4:
            continue
        data_chunk = chunk.dropna(how="all")
        if data_chunk.empty:
            continue
        for _, row in data_chunk.iterrows():
            trigger_value = row.iloc[0]
            event_value = row.iloc[1]
            frame_value = row.iloc[2]
            time_value = row.iloc[3]
            if pd.isna(trigger_value) or pd.isna(event_value):
                continue
            trigger_id = str(trigger_value).strip().strip("'")
            event_label = str(event_value).strip().strip("'")
            if trigger_id.lower() in {"nan", "frame"} or event_label.lower() in {"nan", "time"}:
                continue
            rows.append(
                {
                    "season": "autumn",
                    "trigger_id": trigger_id,
                    "date_str": normalize_date_key(trigger_id.split("_trigger")[0]) if "_trigger" in trigger_id else "",
                    "event_label": event_label,
                    "frame": "" if pd.isna(frame_value) else frame_value,
                    "time_seconds": "" if pd.isna(time_value) else time_value,
                    "source_path": str(workbook.relative_to(ROOT)),
                }
            )
    return rows


def build_trigger_emotion_rows(identity_link_rows: list[dict]) -> list[dict]:
    identity_index = {
        (row["date_str"], row["p_id"]): row
        for row in identity_link_rows
        if row.get("date_str") and row.get("p_id")
    }

    p_file_re = re.compile(r"^(?P<date_str>(?:Oct|May)_\d+)_(?P<p_id>P\d)\.xlsx$", re.IGNORECASE)
    rows: list[dict] = []

    for season, trigger_root in [("autumn", TRIGGER_AUTUMN_ROOT), ("spring", TRIGGER_SPRING_ROOT)]:
        if not trigger_root.exists():
            continue
        for xlsx_path in sorted(trigger_root.rglob("*.xlsx")):
            if xlsx_path.name.startswith("~$"):
                continue
            m = p_file_re.fullmatch(xlsx_path.name)
            if not m:
                continue

            date_str = normalize_date_key(m.group("date_str"))
            p_id = m.group("p_id").upper()

            df = pd.read_excel(xlsx_path, header=None)
            label_col = df[4] if 4 in df.columns else pd.Series(dtype=object)
            events: dict[str, int] = {
                str(val).strip(): int(idx)
                for idx, val in label_col.items()
                if pd.notna(val)
            }

            def _frame(name: str, ev: dict = events) -> int | str:
                return ev.get(name, "")

            def _sec(name: str, ev: dict = events) -> float | str:
                f = ev.get(name)
                return "" if f is None else round(f / TRIGGER_FPS, 3)

            identity_row = identity_index.get((date_str, p_id), {})
            rows.append({
                "season": season,
                "date_str": date_str,
                "p_id": p_id,
                "origin_id": identity_row.get("origin_id", ""),
                "participant_identification": identity_row.get("participant_identification", ""),
                "hardware_id": identity_row.get("hardware_id", ""),
                "room_label": identity_row.get("room_label", ""),
                "identity_join_status": "ok" if identity_row else "no_identity_match",
                "total_frames": len(df),
                "total_seconds": round(len(df) / TRIGGER_FPS, 3),
                "start_frame": _frame("Start"),
                "ct_frame": _frame("CT"),
                "ct_second": _sec("CT"),
                "et1_frame": _frame("ET1"),
                "et1_second": _sec("ET1"),
                "et2_frame": _frame("ET2"),
                "et2_second": _sec("ET2"),
                "et3_frame": _frame("ET3"),
                "et3_second": _sec("ET3"),
                "end_frame": _frame("End"),
                "end_second": _sec("End"),
                "source_path": str(xlsx_path.relative_to(ROOT)),
            })

    return sorted(rows, key=lambda r: (r["season"], r["date_str"], r["p_id"]))


def build_video_timing_rows(
    host_rows: list[dict],
    identity_link_rows: list[dict],
    sync_timing_rows: list[dict],
    device_id_rows: list[dict],
    global_hardware_map: dict[tuple[str, str], str],
    eda_session_rows: list[dict] = None,  # pre-scanned EDA sessions, passed in by the caller
) -> list[dict]:
    rows: list[dict] = []
    device_exact_index, device_room_index = build_device_record_indexes(device_id_rows)

    # Build an index of processed EDA files: (hardware_id, date_str) -> path.
    # Prefer the largest (longest-recording) session so a brief test recording
    # doesn't win via last-match-wins.
    eda_proc_index: dict[tuple[str, str], str] = {}
    _eda_size: dict[tuple[str, str], int] = {}
    if eda_session_rows:
        for r in eda_session_rows:
            if r.get("variant") != "filtered":
                continue
            key = (str(r["hardware_id"]), r["date_str"])
            size = int(r.get("n_bytes") or 0)
            if key not in _eda_size or size > _eda_size[key]:
                _eda_size[key] = size
                eda_proc_index[key] = r["source_path"]

    identity_index = {
        (
            row.get("date_iso", ""),
            row.get("room_label", ""),
            row.get("participant_identification", ""),
        ): row
        for row in identity_link_rows
        if row.get("date_iso") and row.get("room_label") and row.get("participant_identification")
    }
    sync_index: dict[tuple[str, str, str], list[dict]] = {}
    for row in sync_timing_rows:
        group_project = row.get("group_project", "")
        match = re.search(r"G(\d)$", group_project)
        room_label = f"Room{match.group(1)}" if match else ""
        key = (row.get("season", ""), row.get("date_iso", ""), room_label)
        sync_index.setdefault(key, []).append(row)

    def parse_host_video_offset_seconds(value: object) -> float | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "not played":
            return None
        cleaned = re.sub(r"\s*\(.*?\)\s*", "", text)
        if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", cleaned):
            mm, ss, ff = [int(part) for part in cleaned.split(":")]
            return mm * 60 + ss + ff / 30.0
        return parse_timecode_to_seconds(cleaned)

    def first_nonempty_number(*values: object) -> float | None:
        for value in values:
            if pd.isna(value) or value in {"", None}:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                parsed = parse_host_video_offset_seconds(value)
                if parsed is not None:
                    return parsed
                td = pd.to_timedelta(str(value), errors="coerce")
                if pd.notna(td):
                    return td.total_seconds()
        return None

    for host_row in host_rows:
        error_notes = []
        participant_identification = host_row.get("participant_identification", "")
        if not participant_identification:
            continue
            
        identity_row = identity_index.get(
            (host_row.get("date_iso", ""), host_row.get("room_label", ""), participant_identification)
        )
        
        hardware_id = ""
        eda_raw_path = ""
        if identity_row and identity_row.get("hardware_id"):
            hardware_id = identity_row["hardware_id"]
        else:
            hardware_id, status, detail = resolve_hardware_from_device_records(host_row, device_exact_index, device_room_index, global_hardware_map)
            if not hardware_id:
                error_notes.append(f"EDA Hardware mapping failed: {status}")

        # Try to locate the raw EDA physical file
        if hardware_id:
            raw_base = RAW_ROOT / "SHARP_EDA" / "SHARP_EDA_RawFiles"
            if host_row["season"] == "autumn":
                date_dir = f"{host_row['date_iso']}_{host_row['room_label'].replace(' ', '')}Unix"
                target_dir = raw_base / "SharpAutumn EDA" / date_dir
                eda_raw_files = list(target_dir.glob(f"*Shimmer_{hardware_id}*"))
                if eda_raw_files:
                    eda_raw_path = str(eda_raw_files[0].relative_to(ROOT))
            else:
                target_dir = raw_base / "SharpSpring EDA"
                eda_raw_files = list(target_dir.rglob(f"*{host_row['date_iso']}*Shimmer_{hardware_id}*"))
                if eda_raw_files:
                    eda_raw_path = str(eda_raw_files[0].relative_to(ROOT))
            
            if not eda_raw_path:
                error_notes.append("EDA Raw file not found in Shimmer folders")

        sync_matches = sync_index.get((host_row.get("season", ""), host_row.get("date_iso", ""), host_row.get("room_label", "")), [])
        sync_row = sync_matches[0] if len(sync_matches) == 1 else {}
        sync_status = "ok" if len(sync_matches) == 1 else ("missing_sync_match" if not sync_matches else "ambiguous_sync_match")
        if sync_status != "ok":
            error_notes.append(f"Sync timing error: {sync_status}")

        start_video_offset_seconds = first_nonempty_number(host_row.get("start_video_time_secs"), host_row.get("start_video_time"))
        processed_video_start_unix_ms = sync_row.get("processed_video_start_unix_ms", "") if sync_row else ""
        if processed_video_start_unix_ms in {"", None} or pd.isna(processed_video_start_unix_ms):
            processed_video_start_unix_ms = ""
        if processed_video_start_unix_ms != "" and start_video_offset_seconds is not None:
            emotion_relative_zero_unix_ms = int(float(processed_video_start_unix_ms))
            start_video_clip_anchor_unix_ms = int(float(processed_video_start_unix_ms) - start_video_offset_seconds * 1000)
        else:
            emotion_relative_zero_unix_ms = ""
            start_video_clip_anchor_unix_ms = ""
        
        # Fill in the emotion-data source path
        emo_raw_path = identity_row.get("example_source_path", "N/A") if identity_row else "N/A"
        if emo_raw_path == "N/A":
            error_notes.append("Emotion origin file not mapped to any Room/Color")

        date_iso = host_row.get("date_iso", "")
        rows.append(
            {
                "season": host_row.get("season", ""),
                "date_iso": date_iso,
                "date_str": host_row.get("date_str", ""),
                "room_label": host_row.get("room_label", ""),
                "group_project": sync_row.get("group_project", "") if sync_row else "",
                "sync_match_status": sync_status,
                "name": host_row.get("name", ""),
                "participant_identification": participant_identification,
                "shimmer_device_code": host_row.get("shimmer_device_code", ""),
                "hardware_id": hardware_id,
                "p_id": identity_row.get("p_id", "") if identity_row else "",
                "origin_id": identity_row.get("origin_id", "") if identity_row else "",
                "eda_raw_path": eda_raw_path,
                "eda_processed_path": eda_proc_index.get((str(hardware_id), host_row.get("date_str", "")), "N/A"),
                "emotion_raw_path": emo_raw_path,
                "start_video": host_row.get("start_video", ""),
                "start_video_time": host_row.get("start_video_time", ""),
                "processed_video_start_unix_ms": processed_video_start_unix_ms,
                "raw_video_start_unix_ms": sync_row.get("raw_video_start_unix_ms", "") if sync_row else "",
                "emotion_relative_zero_unix_ms": emotion_relative_zero_unix_ms,
                "ct_unix_ms":  "",
                "et1_unix_ms": "",
                "et2_unix_ms": "",
                "et3_unix_ms": "",
                "mapping_notes": "; ".join(error_notes),
                "source_path": host_row.get("path", ""),
            }
        )
    return rows


def patch_trigger_unix_ms(
    video_timing_rows: list[dict],
    trigger_emotion_rows: list[dict],
) -> None:
    """Fill ct/et1/et2/et3_unix_ms using frame-derived seconds + video start.

    Source of truth:
      {x}_unix_ms = emotion_relative_zero_unix_ms + {x}_second * 1000

    emotion_relative_zero_unix_ms traces back to AudioStarttime.xlsx via the
    Sync sheet (sub-ms accuracy). {x}_second comes from trigger frame xlsx
    (frame / 30 fps, ±1/30 s). This replaces the host-sheet wall-clock approach
    which had minute-level precision and known errors (e.g. Oct_22 off by 34 min).
    """
    trig_lookup: dict[tuple[str, str], dict] = {
        (str(r["date_str"]), str(r["p_id"])): r
        for r in trigger_emotion_rows
        if r.get("date_str") and r.get("p_id")
    }

    for row in video_timing_rows:
        zero_v = row.get("emotion_relative_zero_unix_ms")
        if not zero_v and zero_v != 0:
            continue
        try:
            zero_ms = float(zero_v)
        except (TypeError, ValueError):
            continue

        key = (str(row.get("date_str", "")), str(row.get("p_id", "")))
        trig_row = trig_lookup.get(key)
        if trig_row is None:
            continue

        for trig in ("ct", "et1", "et2", "et3"):
            sec_v = trig_row.get(f"{trig}_second")
            if sec_v is None or str(sec_v).strip() in ("", "nan"):
                continue
            try:
                row[f"{trig}_unix_ms"] = int(zero_ms + float(sec_v) * 1000)
            except (TypeError, ValueError):
                pass
