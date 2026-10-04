"""Prepare demo audio, clip metadata, and model-compatible feature files."""

import numpy as np
import pandas as pd

from pathlib import Path

from constants import (
    DEMO_DIR,
    DEMO_FEATURES_DIR,
    DEMO_RAW_AUDIO_DIR,
    DEMO_RAW_CLIPS_DIR,
    DEMO_RAW_DIR,
    DEMO_RAW_VIDEO_DIR,
    PCA_DIR,
    PCA_COMPONENTS,
    SCORE_INDICES,
    YAMNET_STRIDE_SEC,
    YAMNET_WINDOW_SEC,
)
from crowdsense.demo_functions import format_timestamp
from preprocessing.build_clip_database import process_all
from preprocessing.extract_audio import extract_audio
from preprocessing.functions import (
    build_feature_dataframe, extract_yamnet_match, list_matches, load_yamnet_model,
    timestamp_to_seconds,
)

DEMO_DATABASE_PATH = DEMO_DIR / "clips_data.db"
DEMO_MATCHES_PATH = DEMO_DIR / "matches.csv"
DEMO_CLIPS_PATH = DEMO_DIR / "clips.csv"
DEMO_PCA_PATH = PCA_DIR / "pca_transform.npz"
DEMO_START_OFFSET_SEC = 18 * 60 + 4


def _renormalize_demo_clip_timestamps(
    clips_csv_path: Path,
    matches_csv_path: Path,
    start_offset_sec: float = DEMO_START_OFFSET_SEC,
) -> None:
    """Rebase labels to the demo media and clip them to its actual duration."""
    if not clips_csv_path.exists():
        return

    clips = pd.read_csv(clips_csv_path)
    matches = pd.read_csv(matches_csv_path)
    durations = matches.set_index("match_id")["audio_length_sec"].astype(float)

    clips["_start_sec"] = clips["timestamp_formatted"].map(timestamp_to_seconds).astype(float)
    clips["_end_sec"] = clips["_start_sec"] + clips["length_sec"].astype(float)
    clips["_demo_start_sec"] = (clips["_start_sec"] - start_offset_sec).clip(lower=0)
    clips["_demo_end_sec"] = clips.apply(
        lambda row: min(row["_end_sec"] - start_offset_sec,
                        durations.get(row["match_id"], float("inf"))),
        axis=1,
    )
    # Keep only the part of each labelled span that overlaps the trimmed media.
    clips = clips[clips["_demo_end_sec"] > clips["_demo_start_sec"]].copy()
    clips["timestamp_formatted"] = clips["_demo_start_sec"].map(format_timestamp)
    clips["length_sec"] = clips["_demo_end_sec"] - clips["_demo_start_sec"]
    clips.drop(columns=["_start_sec", "_end_sec", "_demo_start_sec", "_demo_end_sec"], inplace=True)
    clips.to_csv(clips_csv_path, index=False)


def prepare_demo() -> None:
    for directory in (DEMO_RAW_DIR, DEMO_RAW_VIDEO_DIR, DEMO_RAW_AUDIO_DIR,
                      DEMO_RAW_CLIPS_DIR, DEMO_FEATURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    extract_audio(DEMO_RAW_VIDEO_DIR, DEMO_RAW_AUDIO_DIR)
    process_all(
        raw_dir=DEMO_RAW_VIDEO_DIR,
        clips_dir=DEMO_RAW_CLIPS_DIR,
        audio_dir=DEMO_RAW_AUDIO_DIR,
        db_path=DEMO_DATABASE_PATH,
        matches_csv_path=DEMO_MATCHES_PATH,
        clips_csv_path=DEMO_CLIPS_PATH,
    )
    _renormalize_demo_clip_timestamps(DEMO_CLIPS_PATH, DEMO_MATCHES_PATH)

    pca_data = np.load(DEMO_PCA_PATH)
    model = load_yamnet_model()
    for match_id, raw_filename in list_matches(DEMO_DATABASE_PATH):
        result = extract_yamnet_match(raw_filename, DEMO_RAW_AUDIO_DIR, model=model)
        if result is None:
            continue
        starts, score_rows, embeddings = result
        features = build_feature_dataframe(
            match_id,
            raw_filename,
            starts,
            score_rows,
            embeddings,
            pca_data["components"],
            pca_data["mean"],
        )
        output_path = DEMO_FEATURES_DIR / f"{Path(raw_filename).stem}.parquet"
        features.to_parquet(output_path, index=False)
        print(f"Wrote {len(features)} feature rows -> {output_path}")

    pd.DataFrame([{
        "window_sec": YAMNET_WINDOW_SEC,
        "stride_sec": YAMNET_STRIDE_SEC,
        "yamnet_score_indices": ",".join(map(str, SCORE_INDICES)),
        "yamnet_embedding_dimensions": int(pca_data["components"].shape[1]),
        "pca_components": int(pca_data["components"].shape[0]),
        "pca_transform_path": str(DEMO_PCA_PATH),
    }]).to_csv(DEMO_FEATURES_DIR / "_meta.csv", index=False)


if __name__ == "__main__":
    prepare_demo()
