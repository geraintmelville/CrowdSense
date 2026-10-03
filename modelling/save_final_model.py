"""Train and persist a single final model + decision threshold for the dashboard.

Run this ONCE after tune_model.py has produced tuned hyperparameters, or supply
model hyperparameters and candidate-window settings explicitly. It
produces data/modelling/final_model/model.ubj and model.json, containing everything
dashboard.py needs to score new footage without retraining on every upload.

Usage:
    python -m modelling.save_final_model

Candidate-window settings (lookback/postroll/merge-gap): now loaded by
default from whichever combination tune_model.py's grid search selected
alongside the winning model hyperparameters (same --results file). Pass
--lookback/--postroll/--merge-gap explicitly to override any of them.
Pass --model-params with a JSON object to use explicit model hyperparameters
instead of loading them from tune_model.py's results.

PCA: this now loads the PCA transform extract_features.py already fit and
saved to --pca-transform (pca_transform.npz).

Threshold selection: this selects the operating point whose measured
candidate budget is closest to --budget, breaking ties on recall. Both OOF
threshold selection and final fitting exclude TEST_MATCH_IDS.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from constants import (
    FEATURES_DIR, FINAL_CURVE_N_THRESHOLDS, LABELS_PATH, MATCHES_PATH, TARGET_BUDGET,
    MODEL_PATH, MODEL_RESULTS_PATH, PCA_DIR, RANDOM_STATE,
)

from modelling.functions import (
    STRIDE_SEC, TEST_MATCH_IDS, WINDOW_SEC, YAMNET_SCORE_INDICES, build_model,
    build_targets, load_best_candidate_config, load_best_params, load_features,
    load_labels, load_raw_durations, pooled_oof_predict, recall_budget_curve,
    required_feature_columns, train_test_split_by_match_id,
)
from modelling.model_artifact import save_model_artifact


def select_threshold(
    groups: np.ndarray,
    starts: np.ndarray,
    probabilities: np.ndarray,
    labels: dict,
    raw_durations: dict,
    merge_gap: float,
    lookback: float,
    postroll: float,
    budget_limit: float,
) -> tuple[float, float, float]:
    """Select the operating point whose budget is closest to the requested budget."""
    if not 0.0 <= budget_limit <= 1.0:
        raise ValueError(f"budget must be between 0 and 1, got {budget_limit}")
    budgets, recalls, thresholds = recall_budget_curve(
        groups, starts, probabilities, labels, merge_gap,
        raw_durations=raw_durations, lookback=lookback, postroll=postroll,
        n_thresholds=FINAL_CURVE_N_THRESHOLDS,
    )
    # Exclude the budget=0 anchor: it must never become a deployable threshold.
    eligible = np.flatnonzero(budgets > 1e-12)
    if not eligible.size:
        raise RuntimeError("No non-empty threshold is available; adjust the candidate-window settings.")
    distance = np.abs(budgets[eligible] - budget_limit)
    closest = eligible[np.isclose(distance, distance.min())]
    position = int(closest[np.argmax(recalls[closest])])  # tie-break on recall
    return float(thresholds[position]), float(recalls[position]), float(budgets[position])

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=FEATURES_DIR,
                        help="Directory of per-match parquet feature files.")
    parser.add_argument("--pca-transform", type=Path, default=PCA_DIR / "pca_transform.npz",
                        help="Fitted PCA transform saved by extract_features.py.")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="Labels CSV written by extract_labels.py; refined bounds train the model and original clip bounds select recall.")
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH,
                        help="matches.csv (match_id, raw_filename, ..., audio_length_sec).")
    parser.add_argument("--results", type=Path, default=MODEL_RESULTS_PATH,
                        help="Tuning results written by tune_model.py; only needed for settings not supplied explicitly.")
    parser.add_argument("--model-params", type=json.loads, default=None,
                        help='Explicit XGBoost hyperparameters as a JSON object, e.g. \'{"n_estimators":900,"max_depth":3}\'; '
                             "default: winning parameters from --results.")
    parser.add_argument("--budget", type=float, default=TARGET_BUDGET,
                        help="Target candidate-footage budget as a fraction of raw footage (default: 0.33).")
    parser.add_argument("--lookback", type=float, default=None,
                        help="Override the tuned lookback (seconds); default: whatever tune_model.py's "
                             "grid search selected alongside the winning model hyperparameters.")
    parser.add_argument("--postroll", type=float, default=None,
                        help="Override the tuned postroll (seconds); default: tuned value.")
    parser.add_argument("--merge-gap", type=float, default=None,
                        help="Override the tuned merge gap (seconds); default: tuned value.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    df = load_features(args.features)
    if args.model_params is not None:
        if not isinstance(args.model_params, dict) or not args.model_params:
            parser.error("--model-params must be a non-empty JSON object.")
        model_params = dict(args.model_params)
        print(f"Using explicit model parameters: {model_params}")
    else:
        model_params = dict(load_best_params(args.results))
        print(f"Loaded tuned parameters from {args.results}: {model_params}")

    window_overrides = {
        name: value for name, value in
        [("lookback", args.lookback), ("postroll", args.postroll), ("merge_gap", args.merge_gap)]
        if value is not None
    }
    if len(window_overrides) == 3:
        tuned_window = None
        lookback, postroll, merge_gap = args.lookback, args.postroll, args.merge_gap
    else:
        tuned_window = load_best_candidate_config(args.results)
        lookback = args.lookback if args.lookback is not None else tuned_window["lookback"]
        postroll = args.postroll if args.postroll is not None else tuned_window["postroll"]
        merge_gap = args.merge_gap if args.merge_gap is not None else tuned_window["merge_gap"]
    overrides = {name: value for name, value in
                 [("lookback", args.lookback), ("postroll", args.postroll), ("merge_gap", args.merge_gap)]
                 if value is not None}
    source = f"tuned={tuned_window}" if tuned_window is not None else "all settings explicit"
    print(f"Candidate window: lookback={lookback:.0f}s postroll={postroll:.0f}s merge_gap={merge_gap:.0f}s "
          f"({source}{', overridden: ' + str(overrides) if overrides else ''})")

    refined_labels = load_labels(args.labels)
    clip_labels = load_labels(args.labels, start_column="clip_start_sec", end_column="clip_end_sec")
    raw_durations = load_raw_durations(args.matches)
    targets = build_targets(df, refined_labels)
    selected_features = required_feature_columns(df)

    pca_columns = sorted(column for column in selected_features if column.startswith("yamnet_embedding_pca_"))
    pca_bundle = {}
    if pca_columns:
        pca_data = np.load(args.pca_transform)
        pca_bundle = {"pca_components": pca_data["components"], "pca_mean": pca_data["mean"]}
        if pca_data["components"].shape[0] != len(pca_columns):
            raise ValueError(
                f"{args.pca_transform} has {pca_data['components'].shape[0]} PCA components "
                f"but the features have {len(pca_columns)} PCA columns -- mismatched run."
            )

    groups = df["match_id"].to_numpy()
    starts = df["start_sec"].to_numpy()
    feature_matrix = df[selected_features].to_numpy()
    train_idx, _ = train_test_split_by_match_id(groups, TEST_MATCH_IDS)

    if np.unique(targets[train_idx]).size < 2:
        raise RuntimeError("Training split needs both positive and negative windows")

    random_state = model_params.pop("random_state", args.random_state)
    model = build_model(model_params, random_state)
    probabilities, _ = pooled_oof_predict(
        feature_matrix[train_idx], targets[train_idx], groups[train_idx],
        model_params, random_state, n_folds=4,
    )
    threshold, recall, budget = select_threshold(
        groups[train_idx], starts[train_idx], probabilities,
        clip_labels, raw_durations, merge_gap, lookback, postroll, args.budget,
    )
    print(f"Decision threshold: {threshold:.4f} (recall={recall:.1%}, budget={budget:.1%}; "
          f"target={args.budget:.1%} on training data; "
          f"test matches excluded: {sorted(TEST_MATCH_IDS)})")

    model.fit(feature_matrix[train_idx], targets[train_idx])

    metadata = {
        "threshold": threshold,
        "target_budget": args.budget,
        "feature_columns": selected_features,
        "yamnet_score_indices": list(YAMNET_SCORE_INDICES),
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "lookback": lookback,
        "postroll": postroll,
        "merge_gap": merge_gap,
    }
    if pca_bundle:
        metadata["pca_components"] = pca_bundle["pca_components"].tolist()
        metadata["pca_mean"] = pca_bundle["pca_mean"].tolist()
    save_model_artifact(model, metadata, args.output)
    print(f"Saved model -> {args.output}")
    print(f"Saved model config -> {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
