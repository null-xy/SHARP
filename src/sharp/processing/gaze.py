from __future__ import annotations

import pandas as pd

from .config import GAZE_XLSX, PROCESSED_ROOT


def process_gaze_data(sync_timing_rows: list[dict], video_timing_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    gaze_out = PROCESSED_ROOT / "gaze"
    gaze_out.mkdir(parents=True, exist_ok=True)
    if not GAZE_XLSX.exists():
        return [], []
    df = pd.read_excel(GAZE_XLSX, sheet_name="Gaze")
    df["Group"] = df["Group"].astype(str).str.strip()
    df["group_project"] = df["Group"].str.upper()
    df["Participant"] = df["Participant"].astype(str).str.strip()
    if "Duration" in df.columns:
        df["Duration"] = pd.to_timedelta(df["Duration"].astype(str))
    sync_df = pd.DataFrame(sync_timing_rows)
    if not sync_df.empty:
        sync_df["group_project"] = sync_df["group_project"].astype(str).str.strip().str.upper()
        df = df.merge(
            sync_df[
                [
                    "group_project",
                    "season",
                    "date_iso",
                    "date_str",
                    "video",
                    "processed_video_start_realtime",
                    "processed_video_start_unix_ms",
                    "processed_video_end_realtime",
                    "processed_video_end_unix_ms",
                ]
            ].drop_duplicates(),
            on="group_project",
            how="left",
        )
    video_df = pd.DataFrame(video_timing_rows)
    if not video_df.empty:
        video_df["group_project"] = video_df["group_project"].astype(str).str.strip().str.upper()
        video_df["participant_identification"] = video_df["participant_identification"].astype(str).str.strip()
        df = df.merge(
            video_df[
                [
                    "group_project",
                    "participant_identification",
                    "room_label",
                    "name",
                    "shimmer_device_code",
                    "hardware_id",
                    "p_id",
                    "origin_id",
                ]
            ].drop_duplicates(),
            left_on=["group_project", "Participant"],
            right_on=["group_project", "participant_identification"],
            how="left",
        )
    if "Start" in df.columns:
        df["start_seconds"] = pd.to_timedelta(df["Start"].astype(str), errors="coerce").dt.total_seconds()
    if "End" in df.columns:
        df["end_seconds"] = pd.to_timedelta(df["End"].astype(str), errors="coerce").dt.total_seconds()
    if "processed_video_start_unix_ms" in df.columns:
        start_ms = pd.to_numeric(df["processed_video_start_unix_ms"], errors="coerce")
        if "start_seconds" in df.columns:
            df["event_start_unix_ms"] = (start_ms + df["start_seconds"] * 1000).round().astype("Int64")
        if "end_seconds" in df.columns:
            df["event_end_unix_ms"] = (start_ms + df["end_seconds"] * 1000).round().astype("Int64")
    df.to_csv(gaze_out / "gaze_events.csv", index=False)
    summary_rows: list[dict] = []
    for (group_id, participant), sub in df.groupby(["Group", "Participant"], dropna=False):
        summary_rows.append(
            {
                "group_id": group_id,
                "group_project": str(sub["group_project"].iloc[0]) if "group_project" in sub.columns else "",
                "participant_color": participant,
                "date_str": str(sub["date_str"].iloc[0]) if "date_str" in sub.columns else "",
                "room_label": str(sub["room_label"].iloc[0]) if "room_label" in sub.columns else "",
                "name": str(sub["name"].iloc[0]) if "name" in sub.columns else "",
                "hardware_id": str(sub["hardware_id"].iloc[0]) if "hardware_id" in sub.columns else "",
                "p_id": str(sub["p_id"].iloc[0]) if "p_id" in sub.columns else "",
                "event_count": len(sub),
                "total_duration": str(sub["Duration"].sum()) if "Duration" in sub.columns else "",
                "top_gazed_entities": "; ".join(sub["Gazed_entity"].astype(str).value_counts().head(5).index.tolist()),
            }
        )
    group_rows: list[dict] = []
    for group_id, sub in df.groupby("Group", dropna=False):
        participants = sorted(sub["Participant"].dropna().astype(str).unique().tolist())
        room_values = sorted({str(v) for v in sub.get("room_label", pd.Series(dtype=str)).dropna().tolist() if str(v) and str(v) != "nan"})
        group_rows.append(
            {
                "group_id": group_id,
                "group_project": str(sub["group_project"].iloc[0]) if "group_project" in sub.columns else "",
                "participants_in_gaze": ",".join(participants),
                "season": str(sub["season"].iloc[0]) if "season" in sub.columns else "",
                "date_iso": str(sub["date_iso"].iloc[0]) if "date_iso" in sub.columns else "",
                "date_str": str(sub["date_str"].iloc[0]) if "date_str" in sub.columns else "",
                "room_label": ",".join(room_values),
                "video": str(sub["video"].iloc[0]) if "video" in sub.columns else "",
                "processed_video_start_realtime": str(sub["processed_video_start_realtime"].iloc[0]) if "processed_video_start_realtime" in sub.columns else "",
                "processed_video_start_unix_ms": str(sub["processed_video_start_unix_ms"].iloc[0]) if "processed_video_start_unix_ms" in sub.columns else "",
                "processed_video_end_realtime": str(sub["processed_video_end_realtime"].iloc[0]) if "processed_video_end_realtime" in sub.columns else "",
                "processed_video_end_unix_ms": str(sub["processed_video_end_unix_ms"].iloc[0]) if "processed_video_end_unix_ms" in sub.columns else "",
                "mapping_confidence": "high" if "date_str" in sub.columns and sub["date_str"].notna().any() else "unmapped",
                "rule": "group_to_sync_exact_casefold",
            }
        )
    return summary_rows, group_rows
