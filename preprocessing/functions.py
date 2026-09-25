import subprocess
import zipfile
import imageio_ffmpeg
import os
import hashlib
import re
import shutil
import wave

from collections import defaultdict
from pathlib import Path
from typing import Any
import sqlite3

from constants import SCORE_INDICES

CLIP_FILENAME_REGEX = re.compile(r"^(?P<clip_num>\d+)\s+(?P<timestamp>\d{6})_-_(?P<desc>.+?)\.[a-zA-Z0-9]+$")

MATCH_FILENAME_REGEX = re.compile(r"^(?P<teams>.+)-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.mp4$")

# Zip filenames carry the same "opposition + date" identity as raw filenames but with
# looser formatting (spaces/apostrophes instead of hyphens, an optional "_highlights_"
# infix). e.g. "Men's A v Baku United_highlights_2026-04-19.zip" vs.
# raw "mens-a-v-baku-united-2026-04-19.mp4". The date always sits at the end.
ZIP_FILENAME_REGEX = re.compile(
    r"^(?P<teams>.+?)(?:[\s_-]*highlights[\s_-]*)?(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.zip$",
    re.IGNORECASE,
)

'''Stores functions used in processing the full match footage + folders of highlight clips 
into a schema that labels each file.'''

def create_schema(conn: sqlite3.Connection) -> None:
    """Create the current clip-processing schema."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_filename TEXT UNIQUE,
            zip_filename TEXT UNIQUE,
            match_date TEXT,
            length_sec REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            match_id INTEGER NOT NULL,
            clip_number INTEGER NOT NULL,
            timestamp_formatted TEXT NOT NULL,
            description TEXT NOT NULL,
            filename TEXT NOT NULL,
            length_sec REAL NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
            PRIMARY KEY (match_id, clip_number)
        )
    """)
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS matches_zip_filename_unique "
        "ON matches(zip_filename) WHERE zip_filename IS NOT NULL"
    )
    conn.commit()
'''
------------------------------------------------------------------------------------------------------------------------------------
'''
def get_raw_duration(raw_path: Path, audio_dir: Path) -> float | None:
    """Returns the duration of a raw video file by analyzing its corresponding audio file, or None if the file does not exist or cannot be read."""

    audio_path = audio_dir / f"{raw_path.stem}.wav"

    if not audio_path.exists():
        return None

    try:
        with wave.open(str(audio_path), "rb") as audio:
            framerate = audio.getframerate()
            if framerate <= 0:
                return None
            return round(audio.getnframes() / float(framerate), 2)
    except wave.Error:
        # File is either non-PCM WAV, corrupted, or not actually a WAV
        return None
    except Exception:
        return None

def extract_match_data(raw_path: Path, audio_dir: Path) -> dict[str, Any]:
    '''Extracts data from a raw match file path.
    Returns a dictionary containing raw_filename, match_date, and length_sec.'''

    filename = raw_path.name
    match = MATCH_FILENAME_REGEX.match(filename)
    
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern (YYYY-MM-DD.mp4).")

    # Extract year, month, and day directly from regex named groups
    year = match.group("year")
    month = match.group("month")
    day = match.group("day")
    
    match_date = f"{year}-{month}-{day}" # Produces "2026-04-19"

    return {
        "raw_filename": filename,
        "match_date": match_date,
        "length_sec": get_raw_duration(raw_path, audio_dir),
    }
'''
------------------------------------------------------------------------------------------------------------------------------------
'''
def _ffmpeg_path() -> str | None:
    """Return the ffmpeg executable used by the local Python environment."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _probe_media(media: bytes) -> float | None:
    ''' Probes mp4 files in zipped folder to obtain its duration in seconds,
    using two methods. Returns None if probing fails.'''

    ffprobe = shutil.which("ffprobe")                                                      # Primary method: checks if ffprobe is installed
    if ffprobe:
        result = subprocess.run(                                                           # Runs media bytes through ffprobe to get duration
            [ffprobe, "-v", "error", "-f", "mp4", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", "pipe:0"],
            input=media, capture_output=True, timeout=30, check=False,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value:                                               # If successful, return duration as float rounded to 3 decimal places
            return round(float(value), 3)

    ffmpeg = _ffmpeg_path()                                                                # Secondary method: if primary fails, fetches ffmpeg path
    if ffmpeg:
        result = subprocess.run(                                                           # Feeds media bytes through ffmpeg
            [ffmpeg, "-hide_banner", "-i", "pipe:0", "-f", "null", "-"],
            input=media, capture_output=True, timeout=30, check=False,
        )
        match = re.search(rb"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)        # Searches log output for duration pattern
        if match:
            hours, minutes, seconds = match.groups()
            return round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 3)        # Returns total duration in seconds rounded to 3 decimal places
    return None

def get_zip_member_duration(zip_file: zipfile.ZipFile, member_name: str) -> float | None:
    """Probe a video member without extracting it to disk."""
    try:
        with zip_file.open(member_name, "r") as source:
            return _probe_media(source.read())
    except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile):
        return None

def extract_clip_data(zip_path: Path) -> dict[int, dict[str, Any]]:
    '''Extracts data from each clip in a ZIP folder.
    Returns a dictionary keyed by clip number containing clip metadata dicts.'''

    clips = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for file_info in archive.infolist():
            if file_info.is_dir():
                continue
                
            filename = Path(file_info.filename).name
            match = CLIP_FILENAME_REGEX.match(filename)
            if not match:
                continue
                
            timestamp = match.group("timestamp")
            clip_num = int(match.group("clip_num"))
            
            description = match.group("desc").replace("_", " ")
            if description.strip().casefold() != "goal":
                continue
            clips[clip_num] = {
                "timestamp": f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}",
                "description": description,
                "filename": filename,
                "duration": get_zip_member_duration(archive, file_info.filename),
            }

    # Python 3.7+ preserves insertion order; sorting dictionary keys keeps it ordered by clip_num
    return dict(sorted(clips.items()))
'''
------------------------------------------------------------------------------------------------------------------------------------
'''
def normalize_teams(text: str) -> str:
    """Normalize a teams string for matching raw filenames against zip filenames.

    Lowercases, strips apostrophes, and collapses any run of non-alphanumeric
    characters (hyphens, underscores, spaces) into a single space, so
    "mens-a-v-baku-united" and "Men's A v Baku United" both become
    "mens a v baku united".
    """
    text = text.lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_raw_match_key(filename: str) -> tuple[str, str]:
    """Return (normalized_teams, match_date) parsed from a raw match filename."""
    match = MATCH_FILENAME_REGEX.match(filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected raw pattern (teams-YYYY-MM-DD.mp4).")
    date = f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
    return normalize_teams(match.group("teams")), date


def parse_zip_match_key(filename: str) -> tuple[str, str] | None:
    """Return (normalized_teams, match_date) parsed from a zip filename, or None if unparsable."""
    match = ZIP_FILENAME_REGEX.match(filename)
    if not match:
        return None
    date = f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
    return normalize_teams(match.group("teams")), date


def build_zip_index(clips_dir: Path) -> tuple[dict[tuple[str, str], list[Path]], list[Path]]:
    """Scan clips_dir for ZIPs and index them by (normalized_teams, match_date).

    Returns (index, unparsable_paths). Multiple zips can share a key if
    filenames are ambiguous -- callers should treat len > 1 as a conflict.
    """
    index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    unparsable: list[Path] = []
    for zip_path in sorted(clips_dir.glob("*.zip")):
        key = parse_zip_match_key(zip_path.name)
        if key is None:
            unparsable.append(zip_path)
            continue
        index[key].append(zip_path)
    return index, unparsable


def insert_or_get_match(conn: sqlite3.Connection, match_data: dict[str, Any], zip_filename: str | None) -> tuple[int, bool]:
    """Insert a match row if it doesn't already exist (keyed on raw_filename).

    Returns (match_id, was_inserted). If the match already exists, its
    zip_filename is refreshed in case a zip was found on this run that
    wasn't found previously.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT match_id FROM matches WHERE raw_filename = ?", (match_data["raw_filename"],))
    existing = cursor.fetchone()
    if existing is not None:
        match_id = existing[0]
        if zip_filename is not None:
            cursor.execute("UPDATE matches SET zip_filename = ? WHERE match_id = ?", (zip_filename, match_id))
        return match_id, False

    cursor.execute(
        """
        INSERT INTO matches (raw_filename, zip_filename, match_date, length_sec)
        VALUES (?, ?, ?, ?)
        """,
        (match_data["raw_filename"], zip_filename, match_data["match_date"], match_data["length_sec"]),
    )
    return cursor.lastrowid, True
'''
------------------------------------------------------------------------------------------------------------------------------------
'''
def list_matches(db_path: Path) -> list[tuple[int, str]]:
    conn = sqlite3.connect(db_path)
    matches = conn.execute(
        "SELECT match_id, raw_filename FROM matches WHERE raw_filename IS NOT NULL ORDER BY match_id"
    ).fetchall()
    conn.close()
    return matches
