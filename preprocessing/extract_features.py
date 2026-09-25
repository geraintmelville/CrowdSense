"""Extract YAMNet scores and PCA-compressed embeddings, one parquet per match.

Windowing is YAMNet's native frame cadence: a 0.96s window every 0.48s
stride. Since this is exactly the frame YAMNet already emits internally,
there is no window aggregation step -- each output row is simply one
untouched YAMNet frame (sliced down to the selected score classes, and with
its embedding PCA-reduced).

PCA is fit ONLY on the training matches (everything except TEST_MATCH_IDS)
and then used to transform every match's embeddings, including the 8 held-out
test matches. This avoids any leakage of test-set embedding structure into
the PCA basis used at train/tune time.

Output: one parquet file per match, written to --output-dir with the same
stem as the match's audio file (e.g. data/raw/audio/foo.wav ->
data/processed/features/foo.parquet), each containing columns:
    match_id, raw_filename, start_sec, <score columns>, <pca columns>
plus a single sidecar data/processed/features/_meta.csv with the run configuration.

Usage:
    python preprocessing/extract_features.py --output-dir data/processed/features/096_048
"""

import argparse
from functools import lru_cache
import time
from pathlib import Path
import tempfile

import tensorflow_hub as hub
import numpy as np
import pandas as pd
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly
from sklearn.decomposition import IncrementalPCA

from constants import (
    CLIP_DATABASE_PATH, FEATURES_DIR, PCA_BATCH_SIZE, PCA_COMPONENTS,
    PCA_DIR, RAW_AUDIO_DIR, SCORE_INDICES, TEST_MATCH_IDS,
    YAMNET_CHUNK_SEC, YAMNET_LOOKAHEAD_SEC, YAMNET_SAMPLE_RATE,
    YAMNET_STRIDE_SEC, YAMNET_WINDOW_SEC,
)

from preprocessing.functions import list_matches

# --- Configuration ----------------------------------------------------------

# YAMNet's native frame cadence: a 0.96s frame emitted every 0.48s. Using
# these exact values means every output row is one raw YAMNet frame, with no
# aggregation across frames.
WINDOW_SEC = YAMNET_WINDOW_SEC
STRIDE_SEC = YAMNET_STRIDE_SEC


@lru_cache(maxsize=1)
def load_yamnet_model():
    """Load YAMNet once for callers that do not provide a model."""
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


def extract_match(raw_filename, audio_dir, score_indices=SCORE_INDICES, model=None):
    """Return (starts, score_features, embeddings) for one match's audio file.

    No window aggregation: WINDOW_SEC/STRIDE_SEC match YAMNet's native frame
    cadence exactly, so each raw frame IS a window. We just slice the score
    columns we care about and pass the embeddings straight through to PCA.
    """
    audio_path = audio_dir / f"{Path(raw_filename).stem}.wav"
    if not audio_path.exists():
        print(f"[SKIP] Missing audio: {audio_path}")
        return None

    scores, embeddings, duration = extract_yamnet_features_streaming(
        audio_path, include_embeddings=True, model=model
    )
    starts = (np.arange(len(scores), dtype=np.float32) * STRIDE_SEC)
    score_rows = scores[:, list(score_indices)].astype(np.float32)
    embedding_rows = embeddings.astype(np.float32)
    return starts, score_rows, embedding_rows


def fit_pca(embedding_paths, n_components, batch_size):
    """Fit IncrementalPCA over saved TRAIN-only window embeddings, in bounded batches."""
    total_rows = sum(np.load(path, mmap_mode="r").shape[0] for path in embedding_paths)
    if total_rows < n_components:
        raise ValueError(f"Need at least {n_components} training windows to fit PCA; found {total_rows}")
    batch_count = max(1, int(np.ceil(total_rows / batch_size)))
    target_batch_size = total_rows // batch_count
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pending = []
    pending_rows = 0
    for embedding_path in embedding_paths:
        embeddings = np.load(embedding_path)
        pending.append(embeddings)
        pending_rows += len(embeddings)
        while pending_rows >= target_batch_size:
            combined = np.concatenate(pending)
            pca.partial_fit(combined[:target_batch_size])
            combined = combined[target_batch_size:]
            pending = [combined] if len(combined) else []
            pending_rows = len(combined)
    if pending_rows:
        pca.partial_fit(np.concatenate(pending))
    return pca


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=CLIP_DATABASE_PATH)
    parser.add_argument("--audio-dir", type=Path, default=RAW_AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--pca-dir", type=Path, default=PCA_DIR)
    parser.add_argument("--limit-matches", type=int, default=None)
    args = parser.parse_args()

    if not TEST_MATCH_IDS:
        raise ValueError(
            "TEST_MATCH_IDS is empty -- hardcode the 8 most recent match_ids at the top "
            "of this script before running (see the comment above TEST_MATCH_IDS)."
        )
    test_match_ids = set(TEST_MATCH_IDS)

    print("Loading YAMNet model...", flush=True)
    model = load_yamnet_model()

    matches = list_matches(args.db)
    if args.limit_matches:
        matches = matches[:args.limit_matches]

    missing_test_ids = test_match_ids - {match_id for match_id, _ in matches}
    if missing_test_ids:
        raise RuntimeError(f"TEST_MATCH_IDS not found among matches: {sorted(missing_test_ids)}")

    score_names = [f"yamnet_score_{index:03d}" for index in SCORE_INDICES]
    pca_columns = [f"yamnet_embedding_pca_{index:02d}" for index in range(PCA_COMPONENTS)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="yamnet_extract_") as temp_dir:
        temp_dir = Path(temp_dir)
        train_embedding_paths = []
        # Per-match cache of what we've extracted, so we only run YAMNet once
        # per match before PCA is fit, then reload for the transform pass.
        match_cache = []  # list of dicts: match_id, raw_filename, starts, score_rows, embedding_path

        for index, (match_id, raw_filename) in enumerate(matches):
            started = time.time()
            print(f"[{index + 1}/{len(matches)}] extracting YAMNet features: match {match_id}", flush=True)
            result = extract_match(raw_filename, args.audio_dir, model=model)
            if result is None:
                continue
            starts, score_rows, embedding_rows = result

            embedding_path = temp_dir / f"{index:06d}.npy"
            np.save(embedding_path, embedding_rows)
            match_cache.append({
                "match_id": match_id,
                "raw_filename": raw_filename,
                "starts": starts,
                "score_rows": score_rows,
                "embedding_path": embedding_path,
            })
            if match_id not in test_match_ids:
                train_embedding_paths.append(embedding_path)

            print(f"    -> {len(starts)} windows in {time.time() - started:.1f}s", flush=True)

        if not match_cache:
            raise RuntimeError("No matches produced any features -- check --audio-dir and filenames.")
        if not train_embedding_paths:
            raise RuntimeError("No training matches (all matches fell in TEST_MATCH_IDS) -- cannot fit PCA.")

        print(f"\nFitting PCA on {len(train_embedding_paths)} training matches "
              f"(test matches {sorted(test_match_ids)} excluded from the fit)...", flush=True)
        pca = fit_pca(train_embedding_paths, PCA_COMPONENTS, PCA_BATCH_SIZE)

        # Persist the fitted PCA itself (not just a summary stat) so it can be
        # reapplied later to embeddings extracted from brand-new footage --
        # e.g. by save_final_model.py / dashboard.py -- without ever refitting
        # on data the deployed model wasn't trained with.
        pca_path = args.pca_dir / "pca_transform.npz"
        np.savez(pca_path, components=pca.components_, mean=pca.mean_)
        print(f"Saved PCA transform -> {pca_path}")

        print("Transforming and writing per-match parquet files...", flush=True)
        total_rows = 0
        for entry in match_cache:
            embeddings = np.load(entry["embedding_path"])
            pca_features = pca.transform(embeddings).astype(np.float32)

            df = pd.DataFrame({
                "match_id": entry["match_id"],
                "raw_filename": entry["raw_filename"],
                "start_sec": entry["starts"],
            })
            df[score_names] = entry["score_rows"]
            df[pca_columns] = pca_features

            output_path = args.output_dir / f"{Path(entry['raw_filename']).stem}.parquet"
            df.to_parquet(output_path, index=False)
            total_rows += len(df)
            split = "test" if entry["match_id"] in test_match_ids else "train"
            print(f"    match {entry['match_id']} ({split}): {len(df)} rows -> {output_path}")

    meta_path = args.output_dir / "_meta.csv"
    pd.DataFrame([{
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "yamnet_score_indices": ",".join(map(str, SCORE_INDICES)),
        "yamnet_embedding_dimensions": int(pca.n_features_in_),
        "pca_components": PCA_COMPONENTS,
        "pca_fit_match_count": len(train_embedding_paths),
        "test_match_ids": ",".join(map(str, sorted(test_match_ids))),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "pca_transform_path": str(pca_path),
    }]).to_csv(meta_path, index=False)

    print(f"\nWrote {total_rows} rows across {len(match_cache)} matches -> {args.output_dir}")
    print(f"Wrote metadata -> {meta_path}")


if __name__ == "__main__":
    main()