"""YAMNet feature extraction and match scoring for inference."""

from functools import lru_cache
import os
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from constants import YAMNET_CHUNK_SEC, YAMNET_LOOKAHEAD_SEC, YAMNET_SAMPLE_RATE, YAMNET_STRIDE_SEC
from modelling.functions import merge_intervals


@lru_cache(maxsize=1)
def load_yamnet_model():
    """Load YAMNet once for callers that do not provide a model."""
    cache_dir = Path(os.environ.setdefault(
        "TFHUB_CACHE_DIR", str(Path.home() / ".cache" / "tensorflow_hub")
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)
    import tensorflow_hub as hub

    return hub.load("https://tfhub.dev/google/yamnet/1")


def extract_yamnet_features_streaming(audio_path, include_embeddings=True, model=None):
    """Extract YAMNet features from bounded audio chunks."""
    model = model if model is not None else load_yamnet_model()
    target_rate = YAMNET_SAMPLE_RATE
    chunk_seconds = YAMNET_CHUNK_SEC
    lookahead_seconds = YAMNET_LOOKAHEAD_SEC
    scores_chunks = []
    embedding_chunks = [] if include_embeddings else None

    with sf.SoundFile(audio_path) as audio:
        source_rate = audio.samplerate
        chunk_frames = max(1, int(chunk_seconds * source_rate))
        lookahead_frames = int(lookahead_seconds * source_rate)
        up = target_rate // gcd(target_rate, source_rate)
        down = source_rate // gcd(target_rate, source_rate)
        total_duration = audio.frames / source_rate

        for source_start in range(0, audio.frames, chunk_frames):
            audio.seek(source_start)
            wav_data = audio.read(
                min(audio.frames - source_start, chunk_frames + lookahead_frames),
                dtype="float32",
                always_2d=False,
            )
            if wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)
            if source_rate != target_rate:
                wav_data = resample_poly(wav_data, up, down).astype(np.float32)

            scores, embeddings, _ = model(wav_data)
            chunk_duration = min(chunk_seconds, total_duration - source_start / source_rate)
            frame_starts = np.arange(len(scores), dtype=np.float32) * YAMNET_STRIDE_SEC
            keep = frame_starts < chunk_duration
            scores_chunks.append(scores.numpy()[keep])
            if include_embeddings:
                embedding_chunks.append(embeddings.numpy()[keep])

    return (
        np.concatenate(scores_chunks),
        np.concatenate(embedding_chunks) if include_embeddings else None,
        total_duration,
    )


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _build_candidate_clips(
    starts: np.ndarray, probabilities: np.ndarray, duration: float, bundle: dict
) -> tuple[pd.DataFrame, float, int]:
    selected_starts = starts[probabilities >= bundle["threshold"]]
    intervals = merge_intervals(
        [(max(0, start - bundle["lookback"]), min(duration, start + bundle["window_sec"] + bundle["postroll"]))
         for start in selected_starts],
        bundle["merge_gap"],
    )
    clips = pd.DataFrame(intervals, columns=["start_sec", "end_sec"])
    clips["length_sec"] = clips["end_sec"] - clips["start_sec"]
    clips["start"] = clips["start_sec"].apply(format_timestamp)
    clips["end"] = clips["end_sec"].apply(format_timestamp)
    return clips, duration, len(starts)


def score_precomputed_features(
    features: pd.DataFrame, bundle: dict, duration: float
) -> tuple[pd.DataFrame, float, int]:
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
    probabilities = bundle["model"].predict_proba(
        features[feature_columns].to_numpy(dtype=np.float32)
    )[:, 1]
    return _build_candidate_clips(starts, probabilities, duration, bundle)


def score_match(audio_path: Path, bundle: dict, yamnet_model=None) -> tuple[pd.DataFrame, float, int]:
    feature_columns = bundle.get("feature_columns", [])
    uses_pca = any(column.startswith("yamnet_embedding_pca_") for column in feature_columns)
    if uses_pca and not {"pca_components", "pca_mean"}.issubset(bundle):
        raise ValueError("This PCA model bundle is missing its PCA projection")
    scores, embeddings, duration = extract_yamnet_features_streaming(
        audio_path,
        include_embeddings=uses_pca,
        model=yamnet_model if yamnet_model is not None else load_yamnet_model(),
    )
    window_sec = bundle["window_sec"]
    stride_sec = bundle["stride_sec"]
    starts = np.arange(len(scores)) * stride_sec

    score_features = scores[:, list(bundle["yamnet_score_indices"])]
    feature_matrix = score_features.astype(np.float32)
    if uses_pca:
        if embeddings is None:
            raise ValueError("Embeddings are required when the model uses PCA features")
        pca_components = np.asarray(bundle["pca_components"])
        pca_mean = np.asarray(bundle["pca_mean"])
        pca_features = (embeddings - pca_mean) @ pca_components.T
        feature_matrix = np.concatenate((feature_matrix, pca_features), axis=1).astype(np.float32)
    probabilities = bundle["model"].predict_proba(feature_matrix)[:, 1]
    return _build_candidate_clips(starts, probabilities, duration, bundle)