from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import (
    eda_raw_root_for_date,
    extract_windows_basename,
    format_ms_as_utc_text,
    parse_dir_session_info,
    parse_hardware_id,
)
from .config import EDA_ROOT, PROCESSED_ROOT, ROOT


def read_raw_shimmer_preview(src: Path, preview_rows: int = 5) -> pd.DataFrame:
    rows: list[list[str]] = []
    with src.open(encoding="utf-8-sig", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r").strip()
            if not line:
                continue
            normalized = line.strip('"')
            if normalized == "sep=\t":
                continue
            parts = normalized.split("\t")
            if parts and parts[-1] == "":
                parts = parts[:-1]
            rows.append(parts)
            if len(rows) >= preview_rows + 1:
                break
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data_rows = rows[1:]
    df = pd.DataFrame(data_rows, columns=header)
    df.insert(0, "Device_ID", parse_hardware_id(src.name))
    return df


def build_combined_eda_manifest_entries(source_summary_dir: Path, example_dir: Path) -> list[dict]:
    manifest: list[dict] = []
    combined_sources = [
        ("autumn", EDA_ROOT / "CombinedEDA" / "syncedcombinedallSHARPAutumn.csv"),
        ("spring", EDA_ROOT / "CombinedEDA" / "syncedcombinedallSHARPSpring.csv"),
    ]
    for season, src in combined_sources:
        df_preview = pd.read_csv(src, nrows=5)
        header = list(df_preview.columns)
        metadata_path = source_summary_dir / f"{season}_combined_metadata.json"
        preview_path = example_dir / f"{season}_combined_head.csv"
        df_preview.to_csv(preview_path, index=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "season": season,
                    "source_path": str(src.relative_to(ROOT)),
                    "source_size_bytes": src.stat().st_size,
                    "column_count": len(header),
                    "columns": header,
                    "preview_path": str(preview_path.relative_to(ROOT)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest.append(
            {
                "dataset_type": "combined_metadata",
                "season": season,
                "session_id": "",
                "variant": "",
                "source_path": str(src.relative_to(ROOT)),
                "processed_path": str(metadata_path.relative_to(ROOT)),
                "row_count": "",
                "first_timestamp": "",
                "last_timestamp": "",
                "column_count": len(header),
                "columns": ",".join(header),
            }
        )
    return manifest


def build_raw_preview_artifacts(session_id: str, filename: str, dirname: str, example_dir: Path) -> tuple[str, str, int, str]:
    raw_unix_src = eda_raw_root_for_date(parse_dir_session_info(dirname)[0]) / extract_windows_basename(dirname) / filename
    raw_unix_source_path = str(raw_unix_src.relative_to(ROOT)) if raw_unix_src.exists() else ""
    raw_preview_path = ""
    raw_column_count = 0
    raw_columns = ""
    if raw_unix_src.exists():
        raw_preview_df = read_raw_shimmer_preview(raw_unix_src, preview_rows=5)
        if not raw_preview_df.empty:
            raw_preview_file = example_dir / f"raw_unix_session_{session_id.zfill(2)}_head.csv"
            raw_preview_df.to_csv(raw_preview_file, index=False)
            raw_preview_path = str(raw_preview_file.relative_to(ROOT))
            raw_column_count = len(raw_preview_df.columns)
            raw_columns = ",".join(raw_preview_df.columns)
    return raw_unix_source_path, raw_preview_path, raw_column_count, raw_columns


def build_processed_eda_session_entry(src: Path, variant: str, source_summary_dir: Path, example_dir: Path) -> tuple[dict, dict]:
    session_id = src.stem
    df_preview = pd.read_csv(src, nrows=5)
    header = list(df_preview.columns)
    usecols = [c for c in ["AdjustedTime", "TimestampSync_Unix", "Filename", "Dirname", "fileID"] if c in header]
    df_meta = pd.read_csv(src, usecols=usecols, nrows=5)
    metadata_path = source_summary_dir / f"{variant}_session_{session_id.zfill(2)}.json"
    preview_path = example_dir / f"{variant}_session_{session_id.zfill(2)}_head.csv"
    df_preview.to_csv(preview_path, index=False)
    filename = str(df_meta["Filename"].dropna().iloc[0]) if "Filename" in df_meta and not df_meta["Filename"].dropna().empty else ""
    dirname = str(df_meta["Dirname"].dropna().iloc[0]) if "Dirname" in df_meta and not df_meta["Dirname"].dropna().empty else ""
    file_id = str(df_meta["fileID"].dropna().iloc[0]) if "fileID" in df_meta and not df_meta["fileID"].dropna().empty else ""
    date_iso, date_str, room_label = parse_dir_session_info(dirname)
    hardware_id = parse_hardware_id(filename)
    raw_unix_source_path, raw_preview_path, raw_column_count, raw_columns = build_raw_preview_artifacts(session_id, filename, dirname, example_dir)
    adjusted_start = df_meta["AdjustedTime"].min() if "AdjustedTime" in df_meta else ""
    ts_start = df_meta["TimestampSync_Unix"].min() if "TimestampSync_Unix" in df_meta else ""
    metadata_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "variant": variant,
                "source_path": str(src.relative_to(ROOT)),
                "source_size_bytes": src.stat().st_size,
                "column_count": len(header),
                "columns": header,
                "preview_path": str(preview_path.relative_to(ROOT)),
                "raw_unix_source_path": raw_unix_source_path,
                "raw_preview_path": raw_preview_path,
                "raw_column_count": raw_column_count,
                "raw_columns": raw_columns.split(",") if raw_columns else [],
                "date_iso": date_iso,
                "date_str": date_str,
                "room_label": room_label,
                "hardware_id": hardware_id,
                "fileID": file_id,
                "filename": filename,
                "dirname": dirname,
                "adjusted_time_start_ms": adjusted_start,
                "adjusted_time_end_ms": "",
                "adjusted_time_start_utc": format_ms_as_utc_text(adjusted_start),
                "adjusted_time_end_utc": "",
                "timestampsync_unix_start_ms": ts_start,
                "timestampsync_unix_end_ms": "",
                "timestampsync_unix_start_utc": format_ms_as_utc_text(ts_start),
                "timestampsync_unix_end_utc": "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    n_bytes = src.stat().st_size
    return (
        {
            "dataset_type": "per_session_metadata",
            "season": "",
            "session_id": session_id,
            "variant": variant,
            "source_path": str(src.relative_to(ROOT)),
            "processed_path": str(metadata_path.relative_to(ROOT)),
            "row_count": "",
            "first_timestamp": adjusted_start,
            "last_timestamp": "",
            "column_count": len(header),
            "columns": ",".join(header),
        },
        {
            "session_id": int(session_id),
            "variant": variant,
            "date_iso": date_iso,
            "date_str": date_str,
            "season": "autumn" if date_iso.startswith("2021-10") else ("spring" if date_iso.startswith("2021-05") else ""),
            "room_label": room_label,
            "hardware_id": hardware_id,
            "fileID": file_id,
            "filename": filename,
            "dirname": dirname,
            "source_path": str(src.relative_to(ROOT)),
            "preview_path": str(preview_path.relative_to(ROOT)),
            "raw_unix_source_path": raw_unix_source_path,
            "raw_preview_path": raw_preview_path,
            "adjusted_time_start_ms": adjusted_start,
            "adjusted_time_end_ms": "",
            "adjusted_time_start_utc": format_ms_as_utc_text(adjusted_start),
            "adjusted_time_end_utc": "",
            "column_count": len(header),
            "columns": ",".join(header),
            "raw_column_count": raw_column_count,
            "raw_columns": raw_columns,
            "n_bytes": n_bytes,
        },
    )


def build_processed_eda_session_entries(source_summary_dir: Path, example_dir: Path) -> tuple[list[dict], list[dict]]:
    manifest: list[dict] = []
    session_index_rows: list[dict] = []
    for variant in ("filtered", "scr"):
        src_dir = PROCESSED_ROOT / "eda" / variant
        if not src_dir.exists():
            continue
        for src in sorted(src_dir.glob("*.csv"), key=lambda p: int(p.stem)):
            manifest_row, session_row = build_processed_eda_session_entry(src, variant, source_summary_dir, example_dir)
            manifest.append(manifest_row)
            session_index_rows.append(session_row)
    return manifest, session_index_rows


def process_combined_eda(season: str, src: Path, session_dir: Path, source_summary_dir: Path, example_dir: Path) -> tuple[list[dict], list[dict]]:
    if not src.exists():
        return [], []
    
    print(f"  [EDA] Extracting sessions from {season} combined file (optimized)...")
    manifest: list[dict] = []
    session_index_rows: list[dict] = []
    
    chunk_size = 1000000
    reader = pd.read_csv(src, chunksize=chunk_size)
    
    session_id_counter = 100
    session_map: dict[tuple[str, str], int] = {}
    handles: dict[int, any] = {}
    
    try:
        for chunk in reader:
            # Drop rows with missing identifiers early
            chunk = chunk.dropna(subset=["Filename", "Dirname"])
            if chunk.empty:
                continue
                
            for (filename, dirname), group in chunk.groupby(["Filename", "Dirname"]):
                key = (str(filename), str(dirname))
                if key not in session_map:
                    session_map[key] = session_id_counter
                    session_id_counter += 1
                
                sid = session_map[key]
                if sid not in handles:
                    dst_path = session_dir / f"{sid}.csv"
                    handles[sid] = dst_path.open("w", encoding="utf-8")
                    # Write header
                    group.to_csv(handles[sid], index=False)
                else:
                    group.to_csv(handles[sid], header=False, index=False)
    finally:
        for h in handles.values():
            h.close()

    # After extraction, build metadata
    for (filename, dirname), sid in sorted(session_map.items(), key=lambda x: x[1]):
        dst_path = session_dir / f"{sid}.csv"
        manifest_row, session_row = build_processed_eda_session_entry(dst_path, "filtered", source_summary_dir, example_dir)
        manifest.append(manifest_row)
        session_index_rows.append(session_row)
        
    return manifest, session_index_rows


def copy_eda_outputs() -> tuple[list[dict], list[dict]]:
    eda_out = PROCESSED_ROOT / "eda"
    source_summary_dir = eda_out / "source_summary"
    example_dir = eda_out / "examples"
    session_dir = eda_out / "sessions"
    
    source_summary_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = build_combined_eda_manifest_entries(source_summary_dir, example_dir)
    
    # Read all processed EDA sessions (Autumn 0-38 and Spring 39-105) from processed_data/eda/
    session_manifest, session_index_rows = build_processed_eda_session_entries(source_summary_dir, example_dir)
    manifest.extend(session_manifest)

    return manifest, session_index_rows
