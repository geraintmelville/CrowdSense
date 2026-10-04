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
import time
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from constants import (
    CLIP_DATABASE_PATH, FEATURES_DIR, PCA_BATCH_SIZE, PCA_COMPONENTS,
    PCA_DIR, RAW_AUDIO_DIR, SCORE_INDICES, TEST_MATCH_IDS,
    YAMNET_STRIDE_SEC, YAMNET_WINDOW_SEC,
)

from preprocessing.functions import (
    build_feature_dataframe,
    extract_yamnet_match,
    fit_pca,
    load_yamnet_model,
    list_matches,
)

# --- Configuration ----------------------------------------------------------

# YAMNet's native frame cadence: a 0.96s frame emitted every 0.48s. Using
# these exact values means every output row is one raw YAMNet frame, with no
# aggregation across frames.
WINDOW_SEC = YAMNET_WINDOW_SEC
STRIDE_SEC = YAMNET_STRIDE_SEC


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
            result = extract_yamnet_match(raw_filename, args.audio_dir, model=model)
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
            df = build_feature_dataframe(
                entry["match_id"],
                entry["raw_filename"],
                entry["starts"],
                entry["score_rows"],
                embeddings,
                pca.components_,
                pca.mean_,
            )

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
