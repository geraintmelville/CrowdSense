import argparse
import csv
import sqlite3
import zipfile
from pathlib import Path

from preprocessing.functions import (
    build_zip_index,
    create_schema,
    extract_clip_data,
    extract_match_data,
    insert_or_get_match,
    parse_raw_match_key
)
from constants import (
    CLIP_DATABASE_PATH,
    RAW_AUDIO_DIR,
    RAW_CLIPS_DIR,
    RAW_VIDEO_DIR,
)

RAW_DIR = RAW_VIDEO_DIR
CLIPS_DIR = RAW_CLIPS_DIR
AUDIO_DIR = RAW_AUDIO_DIR
DB_PATH = CLIP_DATABASE_PATH


def export_database_csvs(
    db_path: Path,
    matches_csv_path: Path | None = None,
    clips_csv_path: Path | None = None,
) -> tuple[Path, Path]:
    """Export a clip database's matches and clips tables to CSV files."""
    matches_csv_path = matches_csv_path or db_path.parent / "matches.csv"
    clips_csv_path = clips_csv_path or db_path.parent / "clips.csv"

    with sqlite3.connect(db_path) as conn:
        matches = conn.execute(
            """
            SELECT match_id, raw_filename, zip_filename, match_date,
                   length_sec AS audio_length_sec
            FROM matches
            ORDER BY match_id
            """
        ).fetchall()
        clips = conn.execute(
            """
            SELECT match_id, clip_number, timestamp_formatted, filename, length_sec
            FROM clips
            ORDER BY match_id, clip_number
            """
        ).fetchall()

    matches_csv_path.parent.mkdir(parents=True, exist_ok=True)
    clips_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with matches_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("match_id", "raw_filename", "zip_filename", "match_date", "audio_length_sec"))
        writer.writerows(matches)

    with clips_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow((
            "clip_id", "match_id", "clip_number", "timestamp_formatted",
            "filename", "length_sec"
        ))
        writer.writerows(
            (clip_id, *clip)
            for clip_id, clip in enumerate(clips, start=1)
        )

    return matches_csv_path, clips_csv_path


def process_all(
    raw_dir: Path = RAW_DIR,
    clips_dir: Path = CLIPS_DIR,
    audio_dir: Path = AUDIO_DIR,
    db_path: Path = DB_PATH,
    matches_csv_path: Path | None = None,
    clips_csv_path: Path | None = None,
) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory does not exist: {clips_dir}")

    raw_paths = sorted(raw_dir.glob("*.mp4"))
    if not raw_paths:
        print("No raw match files found.")

    zip_index, unparsable_zips = build_zip_index(clips_dir)
    for zip_path in unparsable_zips:
        print(f"[FLAG] {zip_path.name}: ZIP_FILENAME_UNPARSABLE")

    print(f"Found {len(raw_paths)} raw matches and {sum(len(v) for v in zip_index.values())} recognizable ZIPs.")
    print()

    used_zip_keys: set[tuple[str, str]] = set()

    with sqlite3.connect(db_path) as conn:
        create_schema(conn)
        cursor = conn.cursor()

        for raw_path in raw_paths:
            try:
                match_data = extract_match_data(raw_path, audio_dir)
                key = parse_raw_match_key(raw_path.name)
            except ValueError as error:
                print(f"[FLAG] {raw_path.name}: {error}")
                continue

            candidates = zip_index.get(key, [])
            zip_filename = None
            if len(candidates) == 1:
                zip_filename = candidates[0].name
                used_zip_keys.add(key)
            elif len(candidates) > 1:
                names = ", ".join(p.name for p in candidates)
                print(f"[FLAG] {raw_path.name}: AMBIGUOUS_ZIP_MATCH ({names})")
            else:
                print(f"[FLAG] {raw_path.name}: NO_ZIP_MATCH")

            match_id, inserted = insert_or_get_match(conn, match_data, zip_filename)
            conn.commit()

            details = f"match_id={match_id} | {zip_filename or 'NO ZIP'}"
            if match_data["match_date"]:
                details += f" | {match_data['match_date']}"
            status = "OK" if inserted else "EXISTS"
            print(f"[{status}] {raw_path.name} -> {details}")

            if zip_filename is None:
                continue

            zip_path = clips_dir / zip_filename
            try:
                clips_data = extract_clip_data(zip_path)
            except zipfile.BadZipFile:
                print(f"[FLAG] {zip_filename}: BAD_ZIP_FILE")
                continue

            cursor.execute("DELETE FROM clips WHERE match_id = ?", (match_id,))
            probe_failures = [
                clip for clip in clips_data.values() if clip["duration"] is None
            ]
            for clip in probe_failures:
                print(f"[FLAG] {zip_filename}/{clip['filename']}: DURATION_PROBE_FAILED")
            clips_data = {
                clip_num: clip
                for clip_num, clip in clips_data.items()
                if clip["duration"] is not None
            }
            cursor.executemany(
                """
                INSERT INTO clips (
                    match_id, clip_number, timestamp_formatted,
                    filename, length_sec
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        match_id,
                        clip_num,
                        clip["timestamp"],
                        clip["filename"],
                        clip["duration"],
                    )
                    for clip_num, clip in clips_data.items()
                ],
            )
            conn.commit()
            print(f"       loaded {len(clips_data)} clips")

        for key, paths in zip_index.items():
            if key not in used_zip_keys:
                names = ", ".join(p.name for p in paths)
                print(f"[FLAG] {names}: ZIP_NOT_MATCHED_TO_RAW")

    export_database_csvs(db_path, matches_csv_path, clips_csv_path)

    print()
    print("Processing complete; opposition + date match between raw and zip filenames was used to associate them.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan raw match footage and highlight ZIPs, populate the matches and clips tables."
    )
    parser.parse_args()
    process_all()


if __name__ == "__main__":
    main()
