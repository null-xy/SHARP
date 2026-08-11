from __future__ import annotations

from .common import write_csv
from .config import PROCESSED_ROOT


def write_eda_outputs(
    eda_manifest: list[dict],
    eda_session_rows: list[dict],
    host_rows: list[dict],
    device_id_rows: list[dict],
    emotion_origin_rows: list[dict],
    sync_timing_rows: list[dict],
    oct_timepoint_rows: list[dict],
    video_timing_rows: list[dict],
) -> None:
    write_csv(PROCESSED_ROOT / "eda" / "eda_index.csv", eda_manifest, ["dataset_type", "season", "session_id", "variant", "source_path", "processed_path", "row_count", "first_timestamp", "last_timestamp", "column_count", "columns"])
    write_csv(PROCESSED_ROOT / "eda" / "eda_sessions.csv", eda_session_rows, ["session_id", "variant", "date_iso", "date_str", "season", "room_label", "hardware_id", "fileID", "filename", "dirname", "source_path", "preview_path", "raw_unix_source_path", "raw_preview_path", "adjusted_time_start_ms", "adjusted_time_end_ms", "adjusted_time_start_utc", "adjusted_time_end_utc", "column_count", "columns", "raw_column_count", "raw_columns", "n_bytes"])
    write_csv(PROCESSED_ROOT / "eda" / "host_device_schedule.csv", host_rows, ["season", "date_iso", "date_str", "room_label", "host_sheet", "date_room_condition", "name", "pseudonym", "path", "shimmer_device_code", "participant_identification", "all_videos", "task_start_time", "start_video", "start_video_time", "start_video_time_secs", "time_ct", "ct_video", "ct_video_time", "ct_time_secs", "ct_secs_since_task_start_time", "time_et1", "et1_video", "et1_video_time", "et1_video_time_secs", "et1_time_since_task_start_time", "time_et2", "et2_video", "et2_video_time", "et2_video_time_secs", "et2_time_since_task_start_time", "time_et3", "et3_video", "et3_video_time", "et3_video_time_secs", "et3_time_since_task_start_time", "task_end_time", "end_video", "end_video_time"])
    write_csv(PROCESSED_ROOT / "eda" / "device_id_records.csv", device_id_rows, ["season", "date_iso", "date_str", "room_label", "participant_identification", "shimmer_device_code", "hardware_id", "source_path"])
    write_csv(PROCESSED_ROOT / "eda" / "emotion_origin_map.csv", emotion_origin_rows, ["date_str", "p_id", "origin_id", "origin_token", "example_source_path"])
    write_csv(PROCESSED_ROOT / "eda" / "sync_timing_map.csv", sync_timing_rows, ["season", "group_project", "date_label", "date_iso", "date_str", "video", "audio_start_realtime", "audio_start_unix_ms", "mic_audio_start_time", "mic_audio_start_seconds", "mic_audio_start_frame", "cut_task_video_start_time", "cut_task_video_start_seconds", "cut_task_video_start_frame", "cut_task_video_end_time", "cut_task_video_end_seconds", "cut_task_video_end_frame", "raw_video_start_realtime", "raw_video_start_unix_ms", "processed_video_start_realtime", "processed_video_start_unix_ms", "processed_video_end_realtime", "processed_video_end_unix_ms", "note", "source_path"])
    write_csv(PROCESSED_ROOT / "eda" / "trigger_timepoints_oct.csv", oct_timepoint_rows, ["season", "trigger_id", "date_str", "event_label", "frame", "time_seconds", "source_path"])
    write_csv(
        PROCESSED_ROOT / "eda" / "video_timing_map.csv",
        video_timing_rows,
        [
            "season",
            "date_iso",
            "date_str",
            "room_label",
            "group_project",
            "sync_match_status",
            "name",
            "participant_identification",
            "shimmer_device_code",
            "hardware_id",
            "p_id",
            "origin_id",
            "eda_raw_path",
            "eda_processed_path",
            "emotion_raw_path",
            "start_video",
            "start_video_time",
            "start_video_offset_seconds",
            "ct_video",
            "ct_video_time",
            "ct_video_offset_seconds",
            "et1_video",
            "et1_video_time",
            "et1_video_offset_seconds",
            "et2_video",
            "et2_video_time",
            "et2_video_offset_seconds",
            "et3_video",
            "et3_video_time",
            "et3_video_offset_seconds",
            "processed_video_start_realtime",
            "processed_video_start_unix_ms",
            "processed_video_end_realtime",
            "processed_video_end_unix_ms",
            "raw_video_start_realtime",
            "raw_video_start_unix_ms",
            "emotion_relative_zero_unix_ms",
            "ct_unix_ms",
            "et1_unix_ms",
            "et2_unix_ms",
            "et3_unix_ms",
            "start_video_clip_anchor_unix_ms",
            "mapping_notes",
            "source_path",
        ],
    )


def write_trigger_emotion_outputs(trigger_emotion_rows: list[dict]) -> None:
    write_csv(
        PROCESSED_ROOT / "eda" / "trigger_emotion_frames.csv",
        trigger_emotion_rows,
        [
            "season", "date_str", "p_id", "origin_id",
            "participant_identification", "hardware_id", "room_label",
            "identity_join_status",
            "total_frames", "total_seconds",
            "start_frame",
            "ct_frame", "ct_second",
            "et1_frame", "et1_second",
            "et2_frame", "et2_second",
            "et3_frame", "et3_second",
            "end_frame", "end_second",
            "source_path",
        ],
    )


def write_gaze_outputs(gaze_summary_rows: list[dict], gaze_group_rows: list[dict]) -> None:
    if gaze_summary_rows:
        write_csv(PROCESSED_ROOT / "gaze" / "gaze_group_summary.csv", gaze_summary_rows, ["group_id", "group_project", "participant_color", "date_str", "room_label", "name", "hardware_id", "p_id", "event_count", "total_duration", "top_gazed_entities"])
    if gaze_group_rows:
        write_csv(PROCESSED_ROOT / "gaze" / "gaze_group_map.csv", gaze_group_rows, ["group_id", "group_project", "participants_in_gaze", "season", "date_iso", "date_str", "room_label", "video", "processed_video_start_realtime", "processed_video_start_unix_ms", "processed_video_end_realtime", "processed_video_end_unix_ms", "mapping_confidence", "rule"])


def write_readme(
    eda_manifest: list[dict],
    transcript_manifest: list[dict],
    edu_manifest: list[dict],
    eda_session_rows: list[dict],
    host_rows: list[dict],
    device_id_rows: list[dict],
    sync_timing_rows: list[dict],
    oct_timepoint_rows: list[dict],
    video_timing_rows: list[dict],
    gaze_summary_rows: list[dict],
    gaze_group_rows: list[dict],
) -> None:
    readme = PROCESSED_ROOT / "README.md"
    readme.write_text(
        f"""# processed_data

This directory holds the analysis-ready outputs derived from
`raw_data/SHARP_EDA`, `raw_data/SHARP_Transcription`, and `raw_data/education`.

## Directory structure

- `eda/source_summary/`
  - Metadata summary of the raw EDA files: source path, file size, column names/count.
- `eda/examples/`
  - Lightweight previews of large files (column layout + first few rows).
- `eda/eda_index.csv`
  - Inventory of EDA files with their source paths.
- `eda/eda_sessions.csv`
  - Session-level EDA index, keyed by `date/room/hardware_id/fileID/time`.
- `eda/host_device_schedule.csv`
  - Host-sheet mapping of `room + shimmer device code + participant identification + CT time`.
- `eda/device_id_records.csv`
  - `device_code -> hardware_id` mapping read directly from `PhysiologicalData/Device ID Records`.
- `eda/sync_timing_map.csv`
  - Real-time start/end of each processed task video, read directly from the `Sync` sheet
    (`Realtime Formular` columns).
- `eda/trigger_timepoints_oct.csv`
  - `Start/CT/ET1/ET2/End` frame/time positions within the trigger video, read directly
    from `time point Oct.xlsx`.
- `eda/video_timing_map.csv`
  - Video-time-indexed join of Host Sheet + identity mapping + Sync: within-video seconds
    for `start/ct/et`, plus the task video's real-time start.
- `eda/emotion_origin_map.csv`
  - `date + P1/P2/P3 + origin_*` mapping extracted from the raw emotion-prediction txt files.
- `transcription/utterances/`
  - One per-utterance CSV per session.
- `transcription/plain_text/`
  - Concatenated plain-text transcript per session.
- `transcription/json/`
  - Structured JSON transcript per session.
- `transcription/transcripts_index.csv`
  - Transcript source priority and output-path index.
- `transcription/all_utterances.csv`
  - All sessions' per-utterance rows merged into one table.
- `education/emotion/`
  - Per-second facial-emotion recognition output (emotion, valence, arousal) extracted from Excel.
- `education/surveys/`
  - Cleaned survey data (pre-test, post-test); names removed, anonymous IDs and scores retained.
- `education/education_index.csv`
  - Index of education-related data (emotion, surveys).
- `gaze/gaze_events.csv`
  - Raw gaze event table.
- `gaze/gaze_group_summary.csv`
  - Per `Group + Participant(color)` event-count summary, joined with identity/device mapping.
- `gaze/gaze_group_map.csv`
  - High-confidence group-level mapping (`gaze Group -> Sync Group/project`) with real-time
    video start/end.

## Transcript processing rules

- Source priority: `uploaded` > `uploaded 111020` > `unsorted`.
- For duplicate sessions, only the higher-priority copy is kept.

## EDA processing rules

- The multi-GB combined master table is not copied, and large CSVs are not rewritten in full.
- `device code -> hardware_id` is read directly from the raw `Device ID Records`, not inferred.
- The time column in the emotion directory is `relative_second` (a relative second index,
  not an absolute timestamp).

## Output size summary

- EDA index records: {len(eda_manifest)}
- EDA session rows: {len(eda_session_rows)}
- Host device rows: {len(host_rows)}
- Device ID rows: {len(device_id_rows)}
- Sync timing rows: {len(sync_timing_rows)}
- Trigger timepoint rows: {len(oct_timepoint_rows)}
- Video timing rows: {len(video_timing_rows)}
- Gaze group summary rows: {len(gaze_summary_rows)}
- Gaze group mapping rows: {len(gaze_group_rows)}
- Transcribed sessions: {len(transcript_manifest)}
- Education data records: {len(edu_manifest)}
""",
        encoding="utf-8",
    )
