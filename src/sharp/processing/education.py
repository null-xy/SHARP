from __future__ import annotations

import pandas as pd

from .common import ensure_clean_dir, write_csv
from .config import EDUCATION_ROOT, PROCESSED_ROOT, ROOT


def process_education_data() -> list[dict]:
    edu_out = PROCESSED_ROOT / "education"
    emotion_out = edu_out / "emotion"
    surveys_out = edu_out / "surveys"
    ensure_clean_dir(emotion_out)
    ensure_clean_dir(surveys_out)
    
    emotion_manifest = []
    emotion_configs = [
        ("autumn", EDUCATION_ROOT / "2022 02 combined emotion 1s" / "Oct_Combined" / "autumn 4 emotion 1s"),
        ("spring", EDUCATION_ROOT / "2022 02 combined emotion 1s" / "May_Combined" / "spring emotion4 1s"),
    ]
    
    for season, emotion_src_dir in emotion_configs:
        if emotion_src_dir.exists():
            for src in sorted(emotion_src_dir.glob("*.xlsx")):
                if src.name.startswith("~$"):
                    continue
                df = pd.read_excel(src, header=None)
                df.columns = ["emotion", "valence", "arousal"]
                df.insert(0, "second", range(len(df)))
                dst_csv = emotion_out / f"{src.stem}.csv"
                df.to_csv(dst_csv, index=False)
                emotion_manifest.append({
                    "type": "emotion",
                    "season": season,
                    "id": src.stem,
                    "processed_path": str(dst_csv.relative_to(ROOT))
                })

    survey_manifest = []
    survey_configs = [
        ("autumn", EDUCATION_ROOT / "0 important tables" / "Self-Reports" / "Self-Reports" / "Autumn", ""),
        ("spring", EDUCATION_ROOT / "0 important tables" / "Self-Reports" / "Self-Reports" / "Spring", "_Spring"),
    ]
    
    for season, survey_src_dir, suffix in survey_configs:
        if survey_src_dir.exists():
            for survey_base in ["Pre-test", "Post-test"]:
                src = survey_src_dir / f"{survey_base}{suffix}.xlsx"
                if not src.exists():
                    continue
                df = pd.read_excel(src, header=None)
                header_row_idx = -1
                for i, row in df.iterrows():
                    if any("Pseudonyms" in str(v) for v in row.values if pd.notna(v)):
                        header_row_idx = i
                        break
                if header_row_idx == -1:
                    continue
                headers = df.iloc[header_row_idx].fillna("Unnamed").astype(str).str.strip()
                data_df = df.iloc[header_row_idx + 1 :].copy()
                data_df.columns = headers
                if "Name" in data_df.columns:
                    data_df = data_df.drop(columns=["Name"])
                
                dst_name = f"{season}_{survey_base.lower()}_cleaned.csv"
                dst_csv = surveys_out / dst_name
                data_df.to_csv(dst_csv, index=False)
                survey_manifest.append({
                    "type": "survey",
                    "season": season,
                    "id": survey_base.lower(),
                    "processed_path": str(dst_csv.relative_to(ROOT))
                })
                
    edu_manifest = emotion_manifest + survey_manifest
    write_csv(edu_out / "education_index.csv", edu_manifest, ["type", "season", "id", "processed_path"])
    return edu_manifest
