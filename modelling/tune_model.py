"""Tune the audio classifier with grouped cross-validation.

TEST_MATCH_IDS (modelling/functions.py) are reserved for eval_model.py and
are never used here. Hyperparameters are ranked by partial recall-budget
AUC (band 25-40%) for each candidate's pooled out-of-fold curve (see
modelling/functions.py: recall_budget_curve / curve_partial_auc).

Candidate-window generation (lookback/postroll/merge_gap) is now part of the
search instead of being fixed CLI values. These three don't require
retraining to evaluate -- they only reshape how already-pooled OOF
probabilities get turned into candidate windows -- so for every sampled set
of model hyperparameters we fit ONCE (expensive) and then sweep the full
CANDIDATE_PARAM_GRID (constants.py) against that same pooled set of
probabilities (cheap, no refitting), keeping whichever window combination
gives the best partial recall-budget AUC (band 25-40%) for that model. The winning combo is
written to the results CSV alongside the winning model params, and
save_final_model.py / eval_model.py load it from there by default.

Features are read from the fixed per-match parquet cache (native YAMNet
window/stride -- see modelling/functions.py); there's nothing left to sweep
there, so --features now points at the parquet directory, not a CSV.
"""

import argparse
import json
from pathlib import Path

from constants import (
    CANDIDATE_PARAM_GRID, CV_FOLDS, FEATURES_DIR, LABELS_PATH, MATCHES_PATH,
    MODEL_RESULTS_PATH, N_ITER, RANDOM_STATE,
)

from modelling.functions import candidate_window_combos, load_training_data, run_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=FEATURES_DIR,
                        help="Directory of per-match parquet feature files.")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="Labels CSV written by extract_labels.py; refined bounds train the model and original clip bounds score recall.")
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH,
                        help="matches.csv (match_id, raw_filename, ..., audio_length_sec).")
    parser.add_argument("--n-iter", type=int, default=N_ITER,
                        help="Number of sampled MODEL-hyperparameter combinations (default: 60). Each one "
                             "is evaluated against every candidate-window combo in CANDIDATE_PARAM_GRID.")
    parser.add_argument("--cv", type=int, default=CV_FOLDS,
                        help="Number of GroupKFold splits per candidate (default: 4).")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--results", type=Path, default=MODEL_RESULTS_PATH)
    args = parser.parse_args()

    candidate_combos = candidate_window_combos(CANDIDATE_PARAM_GRID)
    print(f"Candidate-window grid: {CANDIDATE_PARAM_GRID} -> {len(candidate_combos)} combos evaluated per model candidate")

    data = load_training_data(args)

    results = run_search(data, args, candidate_combos)

    best_row = results.iloc[0]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    results[["rank_test_score", "mean_test_score", "std_test_score",
             "lookback", "postroll", "merge_gap", "params", "full_auc"]].to_csv(args.results, index=False)
    print(f"Best pooled partial recall-budget AUC (band 25-40%): {best_row['mean_test_score']:.3f}")
    print(f"Best candidate window: lookback={best_row['lookback']:.0f}s postroll={best_row['postroll']:.0f}s "
          f"merge_gap={best_row['merge_gap']:.0f}s")
    print(f"Best parameters: {json.dumps(best_row['params'], sort_keys=True)}")
    print(f"Wrote search results: {args.results}")


if __name__ == "__main__":
    main()
