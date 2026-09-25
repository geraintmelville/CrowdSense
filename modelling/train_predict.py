"""Train the highlight classifier from tuned parameters and score the test matches.

The model is fit on every match except TEST_MATCH_IDS (modelling/functions.py).
Per-window probabilities for those test matches are written out so
eval_model.py can sweep thresholds and report a recall-vs-budget curve -- no
threshold is chosen here (that decision is made later, by eye, from the
curve eval_model.py reports).
"""

import argparse
from pathlib import Path

import numpy as np

from constants import (
    FEATURES_DIR, LABELS_PATH, MODEL_RESULTS_PATH, RANDOM_STATE,
    TEST_PROBABILITIES_PATH,
)

from modelling.functions import (
    TEST_MATCH_IDS, build_model, build_targets, load_best_params,
    load_features, load_labels, required_feature_columns,
    train_test_split_by_match_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_DIR,
                        help="Directory of per-match parquet feature files.")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="Refined labels CSV written by extract_labels.py.")
    parser.add_argument("--results", type=Path, default=MODEL_RESULTS_PATH,
                        help="Tuning results written by tune_model.py.")
    parser.add_argument("--output", type=Path, default=TEST_PROBABILITIES_PATH)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    df = load_features(args.features)
    model_params = load_best_params(args.results)
    print(f"Loaded tuned parameters from {args.results}: {model_params}")

    labels = load_labels(args.labels)
    targets = build_targets(df, labels)
    selected_features = required_feature_columns(df)
    feature_matrix = df[selected_features].to_numpy()
    groups = df["match_id"].to_numpy()

    train_idx, test_idx = train_test_split_by_match_id(groups, TEST_MATCH_IDS)
    if np.unique(targets[train_idx]).size < 2:
        raise RuntimeError("Training split needs both positive and negative windows")

    random_state = model_params.pop("random_state", args.random_state)
    model = build_model(model_params, random_state)
    model.fit(feature_matrix[train_idx], targets[train_idx])
    print(f"Trained on {len(train_idx)} windows; test matches reserved for eval_model: {sorted(TEST_MATCH_IDS)}")

    probabilities = model.predict_proba(feature_matrix[test_idx])[:, 1]
    output_df = df.iloc[test_idx][["match_id", "raw_filename", "start_sec"]].copy()
    output_df["probability"] = probabilities

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Wrote {len(output_df)} test-match window probabilities -> {args.output}")


if __name__ == "__main__":
    main()
