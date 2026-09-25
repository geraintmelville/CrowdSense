import argparse
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


def process_all() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {RAW_DIR}")
    if not CLIPS_DIR.exists():
        raise FileNotFoundError(f"Clips directory does not exist: {CLIPS_DIR}")

    raw_paths = sorted(RAW_DIR.glob("*.mp4"))
    if not raw_paths:
        print("No raw match files found.")

    zip_index, unparsable_zips = build_zip_index(CLIPS_DIR)
    for zip_path in unparsable_zips:
        print(f"[FLAG] {zip_path.name}: ZIP_FILENAME_UNPARSABLE")

    print(f"Found {len(raw_paths)} raw matches and {sum(len(v) for v in zip_index.values())} recognizable ZIPs.")
    print()

    used_zip_keys: set[tuple[str, str]] = set()

    with sqlite3.connect(DB_PATH) as conn:
        create_schema(conn)
        cursor = conn.cursor()

        for raw_path in raw_paths:
            try:
                match_data = extract_match_data(raw_path, AUDIO_DIR)
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

            zip_path = CLIPS_DIR / zip_filename
            try:
                clips_data = extract_clip_data(zip_path)
            except zipfile.BadZipFile:
                print(f"[FLAG] {zip_filename}: BAD_ZIP_FILE")
                continue

            cursor.execute("DELETE FROM clips WHERE match_id = ?", (match_id,))
            cursor.executemany(
                """
                INSERT INTO clips (
                    match_id, clip_number, timestamp_formatted, description,
                    filename, length_sec
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        match_id,
                        clip_num,
                        clip["timestamp"],
                        clip["description"],
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