"""Extract narrow, acoustically-centered positive labels from highlight clips.

The clip database gives us WHICH span of each raw match corresponds to a
highlight clip (clip start timestamp + clip length), but that span is the
editor's cut, not the acoustic event: it typically includes lead-in, the
crowd-roar spike, a quiet gap, and sometimes the restart whistle. Training on
the full cut as "positive" dilutes the signal the model has to find.

This script narrows each clip down to a short window centered on its acoustic
peak, using the native-cadence YAMNet features already extracted by
extract_features_YAMNet.py (data/processed/features/096_048/<raw_filename>.parquet):

    1. For each clip whose description matches --target-labels, convert its
       timestamp_formatted + length_sec into a [clip_start, clip_end] span in
       match time (expanded slightly by --search-margin on both sides, since
       the roar can start just after or linger just past the editor's cut).
    2. Within that span, sum the YAMNet score columns named in --peak-labels
       (default: Cheering + Crowd + Applause) for every native frame and take
       the argmax -- the moment of peak crowd-roar energy.
    3. Emit a label interval of width --label-window centered on that peak
       frame's midpoint, clipped to the match's valid time range.

The whistle is deliberately NOT part of the peak search or the label window --
it stays a separate, later signal that can be added as an explicit feature at
train time (e.g. "does Whistling spike within N seconds after this frame"),
rather than smearing the positive-frame definition.

Output: a CSV with one row per extracted label, plus enough of the original
clip bounds and the raw peak score for auditing before it's trusted as ground
truth. This CSV is the new input wherever the old load_labels() fed
build_targets() -- (start_sec, end_sec, description) per match_id, just built
here instead of on the fly.

Usage:
    python preprocessing/extract_labels.py --features-dir data/processed/features/096_048 --output data/processed/labels/labels.csv
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    CLIP_DATABASE_PATH, FEATURES_DIR, LABELS_PATH, LABEL_WINDOW_SEC,
    MIN_PEAK_SCORE, PEAK_LABELS, SCORE_INDICES, SEARCH_MARGIN_SEC,
)


def timestamp_to_seconds(timestamp_formatted: str) -> float:
    """Convert an 'HH:MM:SS' clip timestamp (offset into the raw match) to seconds."""
    hours, minutes, seconds = (int(part) for part in timestamp_formatted.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def load_clips(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    clips = pd.read_sql_query(
        """
        SELECT clips.match_id, clips.clip_number, clips.timestamp_formatted,
               clips.description, clips.length_sec, matches.raw_filename
        FROM clips
        JOIN matches ON matches.match_id = clips.match_id
        WHERE matches.raw_filename IS NOT NULL
        ORDER BY clips.match_id, clips.clip_number
        """,
        conn,
    )
    conn.close()
    return clips.reset_index(drop=True)


def resolve_peak_columns(peak_labels: list[str]) -> list[str]:
    """Map human-readable score names (as they appear in SCORE_INDICES) to column names."""
    name_to_column = {name: f"yamnet_score_{index:03d}" for index, name in SCORE_INDICES.items()}
    resolved = []
    for label in peak_labels:
        if label not in name_to_column:
            available = ", ".join(sorted(name_to_column))
            raise ValueError(f"'{label}' is not a recognized SCORE_INDICES name. Available: {available}")
        resolved.append(name_to_column[label])
    return resolved


def load_match_features(features_dir: Path, raw_filename: str) -> pd.DataFrame | None:
    feature_path = features_dir / f"{Path(raw_filename).stem}.parquet"
    if not feature_path.exists():
        return None
    return pd.read_parquet(feature_path)


def find_peak(
    match_features: pd.DataFrame,
    peak_columns: list[str],
    window_sec: float,
    search_start: float,
    search_end: float,
) -> tuple[float, float] | None:
    """Return (peak_frame_midpoint_sec, peak_score) within [search_start, search_end], or None if empty."""
    in_range = match_features[
        (match_features["start_sec"] >= search_start) & (match_features["start_sec"] <= search_end)
    ]
    if in_range.empty:
        return None
    composite = in_range[peak_columns].sum(axis=1).to_numpy()
    peak_position = int(np.argmax(composite))
    peak_row = in_range.iloc[peak_position]
    peak_midpoint = float(peak_row["start_sec"]) + window_sec / 2.0
    return peak_midpoint, float(composite[peak_position])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=CLIP_DATABASE_PATH)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR,
                        help="Directory of per-match parquet files from extract_features_YAMNet.py.")
    parser.add_argument("--output", type=Path, default=LABELS_PATH)
    parser.add_argument("--peak-labels", type=str, default=",".join(PEAK_LABELS),
                        help="Comma-separated SCORE_INDICES names summed to find the acoustic peak. "
                             "Deliberately excludes Whistling -- that stays a downstream feature, not a label input.")
    parser.add_argument("--label-window", type=float, default=LABEL_WINDOW_SEC,
                        help="Width in seconds of the emitted positive label, centered on the peak (default: 3.0).")
    parser.add_argument("--search-margin", type=float, default=SEARCH_MARGIN_SEC,
                        help="Seconds to expand each clip's own [start, start+length] span by, on both sides, "
                             "before searching for the peak (default: 5.0).")
    parser.add_argument("--min-peak-score", type=float, default=MIN_PEAK_SCORE,
                        help="Peaks below this composite score are flagged as weak in stdout (default: 0, i.e. no flag).")
    parser.add_argument("--drop-weak-peaks", action="store_true",
                        help="Exclude weak-peak clips (see --min-peak-score) from the output instead of just flagging them.")
    args = parser.parse_args()

    peak_labels = [label.strip() for label in args.peak_labels.split(",")]
    peak_columns = resolve_peak_columns(peak_labels)
    print(f"Peak search columns: {dict(zip(peak_labels, peak_columns))}")

    meta_path = args.features_dir / "_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path} -- expected alongside the per-match parquet files")
    meta = pd.read_csv(meta_path).iloc[0]
    window_sec = float(meta["window_sec"])

    clips = load_clips(args.db)
    print(f"{len(clips)} goal clips loaded")

    feature_cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    n_no_features = n_empty_search = n_weak = 0

    for clip in clips.itertuples(index=False):
        if clip.raw_filename not in feature_cache:
            feature_cache[clip.raw_filename] = load_match_features(args.features_dir, clip.raw_filename)
        match_features = feature_cache[clip.raw_filename]
        if match_features is None:
            print(f"[FLAG] match {clip.match_id} ({clip.raw_filename}): NO_FEATURES_FOUND -- skipping clip {clip.clip_number}")
            n_no_features += 1
            continue

        clip_start = timestamp_to_seconds(clip.timestamp_formatted)
        clip_end = clip_start + (clip.length_sec or 0.0)
        match_max_time = float(match_features["start_sec"].max()) + window_sec

        search_start = max(0.0, clip_start - args.search_margin)
        search_end = min(match_max_time, clip_end + args.search_margin)

        peak = find_peak(match_features, peak_columns, window_sec, search_start, search_end)
        if peak is None:
            print(f"[FLAG] match {clip.match_id} clip {clip.clip_number}: EMPTY_SEARCH_WINDOW "
                  f"({search_start:.1f}-{search_end:.1f}s) -- skipping")
            n_empty_search += 1
            continue
        peak_midpoint, peak_score = peak

        if peak_score < args.min_peak_score:
            print(f"[FLAG] match {clip.match_id} clip {clip.clip_number}: WEAK_PEAK "
                  f"(score={peak_score:.3f} < {args.min_peak_score})")
            n_weak += 1
            if args.drop_weak_peaks:
                continue

        label_start = max(0.0, peak_midpoint - args.label_window / 2.0)
        label_end = min(match_max_time, peak_midpoint + args.label_window / 2.0)

        rows.append({
            "match_id": clip.match_id,
            "raw_filename": clip.raw_filename,
            "clip_number": clip.clip_number,
            "description": clip.description,
            "clip_start_sec": round(clip_start, 2),
            "clip_end_sec": round(clip_end, 2),
            "peak_time_sec": round(peak_midpoint, 2),
            "peak_score": round(peak_score, 4),
            "label_start_sec": round(label_start, 2),
            "label_end_sec": round(label_end, 2),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(rows)
    output_df.to_csv(args.output, index=False)

    print(f"\nWrote {len(output_df)} extracted labels -> {args.output}")
    print(f"  skipped (no features file): {n_no_features}")
    print(f"  skipped (empty search window): {n_empty_search}")
    print(f"  weak peaks below --min-peak-score={args.min_peak_score} "
          f"({'dropped' if args.drop_weak_peaks else 'kept, flagged only'}): {n_weak}")
    if not output_df.empty:
        print(f"  peak_score distribution: min={output_df['peak_score'].min():.3f} "
              f"median={output_df['peak_score'].median():.3f} max={output_df['peak_score'].max():.3f}")


if __name__ == "__main__":
    main()
