from __future__ import annotations

import json
import re

from .common import TranscriptSource, write_csv
from .config import LANG_TAG_RE, PROCESSED_ROOT, ROOT, SRT_BLOCK_RE, TRANSCRIPT_SOURCE_PRIORITY, TRANSCRIPTION_ROOT


def build_transcript_session_index(sync_timing_rows: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in sync_timing_rows:
        group_project = row.get("group_project", "")
        if not group_project or not re.match(r"[SD]*D\d+G\d+", group_project, re.IGNORECASE):
            continue
        session_id = group_project.lower()
        m = re.search(r"G(\d)", group_project, re.IGNORECASE)
        room_label = f"Room{m.group(1)}" if m else ""
        index[session_id] = {
            "season": row.get("season", ""),
            "date_iso": row.get("date_iso", ""),
            "date_str": row.get("date_str", ""),
            "room_label": room_label,
            "group_project": group_project,
        }
    return index


def choose_transcript_sources() -> list[TranscriptSource]:
    chosen: dict[str, TranscriptSource] = {}
    for bucket in TRANSCRIPT_SOURCE_PRIORITY:
        for path in sorted((TRANSCRIPTION_ROOT / bucket).glob("*.txt")):
            session_id = path.name.removesuffix(".wav.txt")
            chosen.setdefault(session_id, TranscriptSource(session_id=session_id, source_bucket=bucket, source_path=path))
    return [chosen[k] for k in sorted(chosen)]


def srt_time_to_seconds(value: str) -> float:
    hh, mm, rest = value.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_transcript(path) -> list[dict]:
    text = path.read_text(encoding="latin-1", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    utterances: list[dict] = []
    for idx, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        match = SRT_BLOCK_RE.match(lines[0])
        if not match:
            continue
        body = " ".join(lines[1:]).strip()
        lang = ""
        lang_match = LANG_TAG_RE.match(body)
        if lang_match:
            lang = lang_match.group("lang")
            body = LANG_TAG_RE.sub("", body, count=1).strip()
        utterances.append(
            {
                "utterance_index": idx,
                "start_time": match.group("start"),
                "end_time": match.group("end"),
                "start_seconds": f"{srt_time_to_seconds(match.group('start')):.3f}",
                "end_seconds": f"{srt_time_to_seconds(match.group('end')):.3f}",
                "language": lang,
                "text": body,
            }
        )
    return utterances


def process_transcripts(sync_timing_rows: list[dict] | None = None) -> list[dict]:
    session_index = build_transcript_session_index(sync_timing_rows or [])

    transcript_out = PROCESSED_ROOT / "transcription"
    utterance_dir = transcript_out / "utterances"
    text_dir = transcript_out / "plain_text"
    json_dir = transcript_out / "json"
    utterance_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    all_rows: list[dict] = []
    for source in choose_transcript_sources():
        utterances = parse_transcript(source.source_path)
        lookup_id = source.session_id.removesuffix("_final")
        sync_meta = session_index.get(lookup_id, {})
        session_join_status = "ok" if sync_meta else "no_sync_match"

        utterance_csv = utterance_dir / f"{source.session_id}.csv"
        write_csv(utterance_csv, utterances, ["utterance_index", "start_time", "end_time", "start_seconds", "end_seconds", "language", "text"])
        (text_dir / f"{source.session_id}.txt").write_text("\n".join(row["text"] for row in utterances if row["text"]).strip() + "\n", encoding="utf-8")
        (json_dir / f"{source.session_id}.json").write_text(
            json.dumps(
                {
                    "session_id": source.session_id,
                    "source_bucket": source.source_bucket,
                    "source_path": str(source.source_path.relative_to(ROOT)),
                    "season": sync_meta.get("season", ""),
                    "date_iso": sync_meta.get("date_iso", ""),
                    "date_str": sync_meta.get("date_str", ""),
                    "room_label": sync_meta.get("room_label", ""),
                    "group_project": sync_meta.get("group_project", ""),
                    "utterance_count": len(utterances),
                    "utterances": utterances,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        duration_seconds = float(utterances[-1]["end_seconds"]) if utterances else 0.0
        manifest.append(
            {
                "session_id": source.session_id,
                "source_bucket": source.source_bucket,
                "season": sync_meta.get("season", ""),
                "date_iso": sync_meta.get("date_iso", ""),
                "date_str": sync_meta.get("date_str", ""),
                "room_label": sync_meta.get("room_label", ""),
                "group_project": sync_meta.get("group_project", ""),
                "session_join_status": session_join_status,
                "source_path": str(source.source_path.relative_to(ROOT)),
                "utterance_csv": str(utterance_csv.relative_to(ROOT)),
                "plain_text": str((text_dir / f"{source.session_id}.txt").relative_to(ROOT)),
                "json_path": str((json_dir / f"{source.session_id}.json").relative_to(ROOT)),
                "utterance_count": len(utterances),
                "duration_seconds": f"{duration_seconds:.3f}",
            }
        )
        for row in utterances:
            all_rows.append({
                "session_id": source.session_id,
                "source_bucket": source.source_bucket,
                "season": sync_meta.get("season", ""),
                "date_iso": sync_meta.get("date_iso", ""),
                "date_str": sync_meta.get("date_str", ""),
                "room_label": sync_meta.get("room_label", ""),
                "group_project": sync_meta.get("group_project", ""),
                **row,
            })
    write_csv(
        transcript_out / "transcripts_index.csv",
        manifest,
        ["session_id", "source_bucket", "season", "date_iso", "date_str", "room_label", "group_project", "session_join_status", "source_path", "utterance_csv", "plain_text", "json_path", "utterance_count", "duration_seconds"],
    )
    write_csv(
        transcript_out / "all_utterances.csv",
        all_rows,
        ["session_id", "source_bucket", "season", "date_iso", "date_str", "room_label", "group_project", "utterance_index", "start_time", "end_time", "start_seconds", "end_seconds", "language", "text"],
    )
    return manifest
