"""Inference helpers used by the Streamlit demo."""

from pathlib import Path

import numpy as np
import pandas as pd
from constants.constants import DEMO_CLIPS, DEMO_MATCH
from constants import DEMO_CANDIDATE_BUDGET
from modelling.functions import merge_interval_arrays, score_feature_matrix
from preprocessing.functions import extract_yamnet_features_streaming, timestamp_to_seconds


def format_timestamp(seconds: float) -> str:
    """Format a time in seconds as HH:MM:SS for the demo UI."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_candidate_clips(
    starts: np.ndarray,
    probabilities: np.ndarray,
    duration: float,
    bundle: dict,
    target_budget: float = DEMO_CANDIDATE_BUDGET,
) -> tuple[pd.DataFrame, float, int]:
    """Turn scored windows into padded, merged candidate clips for the demo."""
    def make_clips(threshold: float) -> pd.DataFrame:
        selected_starts = starts[probabilities >= threshold]
        interval_starts = np.maximum(0.0, selected_starts - bundle["lookback"])
        interval_ends = np.minimum(
            duration,
            selected_starts + bundle["window_sec"] + bundle["postroll"],
        )
        merged_starts, merged_ends = merge_interval_arrays(
            interval_starts, interval_ends, bundle["merge_gap"]
        )
        return pd.DataFrame({"start_sec": merged_starts, "end_sec": merged_ends})

    breakpoints = np.unique(probabilities)
    if len(breakpoints):
        low, high = 0, len(breakpoints)
        while low < high:
            middle = (low + high) // 2
            candidate = make_clips(float(breakpoints[middle]))
            seconds = candidate["end_sec"].sub(candidate["start_sec"]).sum()
            if (seconds / duration if duration else 0.0) <= target_budget:
                high = middle
            else:
                low = middle + 1
        choices = []
        for index in {max(0, low - 1), min(low, len(breakpoints) - 1)}:
            threshold = float(breakpoints[index])
            candidate = make_clips(threshold)
            seconds = candidate["end_sec"].sub(candidate["start_sec"]).sum()
            budget = seconds / duration if duration else 0.0
            choices.append((abs(budget - target_budget), threshold, candidate))
        empty_threshold = float(np.nextafter(breakpoints[-1], np.inf))
        choices.append((target_budget, empty_threshold, make_clips(empty_threshold)))
        _, threshold, clips = min(choices, key=lambda item: (item[0], item[1]))
    else:
        threshold = float(bundle["threshold"])
        clips = make_clips(threshold)

    clips["length_sec"] = clips["end_sec"] - clips["start_sec"]
    clips["start"] = clips["start_sec"].apply(format_timestamp)
    clips["end"] = clips["end_sec"].apply(format_timestamp)
    clips.attrs["threshold"] = threshold
    clips.attrs["target_budget"] = target_budget
    return clips, duration, len(starts)


def _score_and_build_candidates(
    starts: np.ndarray,
    feature_matrix: np.ndarray,
    bundle: dict,
    duration: float,
) -> tuple[pd.DataFrame, float, int]:
    probabilities = score_feature_matrix(bundle["model"], feature_matrix)
    return build_candidate_clips(starts, probabilities, duration, bundle)


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
    clips_path = DEMO_CLIPS
    matches_path = DEMO_MATCH
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
        start = timestamp_to_seconds(label.timestamp_formatted)
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
    feature_matrix = features[feature_columns].to_numpy(dtype=np.float32)
    return _score_and_build_candidates(starts, feature_matrix, bundle, duration)


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
    feature_columns = bundle["feature_columns"]
    feature_matrix = np.empty((len(scores), len(feature_columns)), dtype=np.float32)
    score_indices = list(bundle["yamnet_score_indices"])
    score_column_positions = {
        f"yamnet_score_{index:03d}": index
        for index in score_indices
    }
    for feature_position, column in enumerate(feature_columns):
        if column in score_column_positions:
            feature_matrix[:, feature_position] = scores[:, score_column_positions[column]]
    if uses_pca:
        pca_components = np.asarray(bundle["pca_components"], dtype=np.float32)
        pca_mean = np.asarray(bundle["pca_mean"], dtype=np.float32)
        np.subtract(embeddings, pca_mean, out=embeddings)
        pca_features = np.empty((len(embeddings), pca_components.shape[0]), dtype=np.float32)
        np.matmul(embeddings, pca_components.T, out=pca_features)
        for component in range(pca_components.shape[0]):
            column = f"yamnet_embedding_pca_{component:02d}"
            feature_matrix[:, feature_columns.index(column)] = pca_features[:, component]
        del embeddings, pca_features
    del scores
    return _score_and_build_candidates(starts, feature_matrix, bundle, duration)
