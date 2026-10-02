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
from preprocessing.build_clip_database import process_all
from preprocessing.extract_audio import extract_audio
from preprocessing.functions import build_feature_dataframe, extract_yamnet_match, list_matches
from modelling.inference import load_yamnet_model

DEMO_DATABASE_PATH = DEMO_DIR / "clips_data.db"
DEMO_MATCHES_PATH = DEMO_DIR / "matches.csv"
DEMO_CLIPS_PATH = DEMO_DIR / "clips.csv"
DEMO_PCA_PATH = PCA_DIR / "pca_transform.npz"


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