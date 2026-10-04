"""Inference helpers used by the Streamlit demo."""

from pathlib import Path

import numpy as np
import pandas as pd
from constants.constants import DEMO_DIR
from modelling.functions import (
    build_candidate_clips, score_feature_matrix,
)
from preprocessing.functions import build_feature_dataframe, extract_yamnet_features_streaming


def youtube_timestamp_url(video_url: str, start_seconds: float) -> str:
    """Return a YouTube URL pointed at the candidate start time."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    parts = urlsplit(video_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["t"] = f"{int(start_seconds)}s"
    return urlunsplit(parts._replace(query=urlencode(query)))


def format_mmss(seconds: float) -> str:
    minutes, secs = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def goal_intervals(result_key: str) -> list[tuple[float, float]]:
    """Read editor-labelled goal spans for the match shown in the demo."""
    clips_path = DEMO_DIR / "clips.csv"
    matches_path = DEMO_DIR / "matches.csv"
    if not clips_path.exists() or not matches_path.exists():
        return []
    match_name = Path(result_key.split(":", 1)[1]).stem
    matches = pd.read_csv(matches_path)
    matched = matches[matches["raw_filename"].map(lambda name: Path(name).stem) == match_name]
    if matched.empty:
        return []
    match_id = matched.iloc[0]["match_id"]
    labels = pd.read_csv(clips_path)
    labels = labels[labels["match_id"] == match_id]
    intervals = []
    for label in labels.itertuples(index=False):
        start = sum(
            int(part) * factor
            for part, factor in zip(label.timestamp_formatted.split(":"), (3600, 60, 1))
        )
        intervals.append((float(start), float(start) + float(label.length_sec)))
    return intervals


def score_precomputed_features(features: pd.DataFrame, bundle: dict, duration: float):
    """Score model-ready YAMNet feature rows without running YAMNet."""
    feature_columns = bundle.get("feature_columns", [])
    missing_columns = set(feature_columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"Precomputed features are missing model columns: {sorted(missing_columns)}")
    if "start_sec" not in features.columns:
        raise ValueError("Precomputed features must include start_sec")
    if features.empty:
        raise ValueError("Precomputed feature file contains no windows")

    features = features.sort_values("start_sec")
    starts = features["start_sec"].to_numpy(dtype=np.float64)
    probabilities = score_feature_matrix(
        bundle["model"], features[feature_columns].to_numpy(dtype=np.float32)
    )
    return build_candidate_clips(starts, probabilities, duration, bundle)


def score_match(audio_path: Path, bundle: dict, yamnet_model=None):
    """Extract YAMNet features and score one full-match audio file."""
    feature_columns = bundle.get("feature_columns", [])
    uses_pca = any(column.startswith("yamnet_embedding_pca_") for column in feature_columns)
    if uses_pca and not {"pca_components", "pca_mean"}.issubset(bundle):
        raise ValueError("This PCA model bundle is missing its PCA projection")

    scores, embeddings, duration = extract_yamnet_features_streaming(
        audio_path, include_embeddings=uses_pca, model=yamnet_model,
    )
    starts = np.arange(len(scores)) * bundle["stride_sec"]
    score_indices = list(bundle["yamnet_score_indices"])
    score_features = scores[:, score_indices].astype(np.float32)
    if uses_pca:
        feature_df = build_feature_dataframe(
            0, Path(audio_path).name, starts, score_features, embeddings,
            np.asarray(bundle["pca_components"]), np.asarray(bundle["pca_mean"]),
        )
    else:
        score_columns = [f"yamnet_score_{index:03d}" for index in score_indices]
        feature_df = pd.DataFrame(score_features, columns=score_columns)
        feature_df["start_sec"] = starts
    feature_matrix = feature_df[bundle["feature_columns"]].to_numpy(dtype=np.float32)
    probabilities = score_feature_matrix(bundle["model"], feature_matrix)
    return build_candidate_clips(starts, probabilities, duration, bundle)
