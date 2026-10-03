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
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from constants import (
    BUDGET_BAND, CANDIDATE_PARAM_GRID, CV_FOLDS, FEATURES_DIR, LABELS_PATH, MATCHES_PATH,
    MODEL_RESULTS_PATH, MODEL_PARAM_DISTRIBUTIONS, N_ITER, RANDOM_STATE,
)

from modelling.functions import (
    TEST_MATCH_IDS, build_targets, count_points_in_band, curve_auc,
    curve_partial_auc, load_features, load_labels, load_raw_durations,
    pooled_oof_predict, recall_budget_curve,
    required_feature_columns,
)


@dataclass
class TrainingData:
    feature_matrix: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    starts: np.ndarray
    labels: dict
    raw_durations: dict


def load_training_data(args: argparse.Namespace) -> TrainingData:
    df = load_features(args.features)
    refined_labels = load_labels(args.labels)
    clip_labels = load_labels(args.labels, start_column="clip_start_sec", end_column="clip_end_sec")
    raw_durations = load_raw_durations(args.matches)
    targets = build_targets(df, refined_labels)
    selected_features = required_feature_columns(df)
    print(f"Using {len(selected_features)} features")

    groups = df["match_id"].to_numpy()
    holdout_match_ids = set(TEST_MATCH_IDS)
    missing_match_ids = holdout_match_ids - set(groups.astype(int))
    if missing_match_ids:
        raise RuntimeError(f"Test matches are missing from features: {sorted(missing_match_ids)}")
    train_idx = np.flatnonzero(~np.isin(groups, list(holdout_match_ids)))
    if not train_idx.size:
        raise RuntimeError("Need training matches outside the test set")
    print(f"Training windows: {len(train_idx)} across {len(np.unique(groups[train_idx]))} matches")
    print(f"Test matches (excluded from tuning): {sorted(holdout_match_ids)}")

    return TrainingData(
        feature_matrix=df[selected_features].iloc[train_idx].to_numpy(),
        targets=targets[train_idx],
        groups=groups[train_idx],
        starts=df["start_sec"].to_numpy()[train_idx],
        labels=clip_labels,
        raw_durations=raw_durations,
    )


def candidate_window_combos(candidate_grid: dict[str, list[float]]) -> list[dict[str, float]]:
    """Every (lookback, postroll, merge_gap) combination in the grid, as dicts."""
    keys = list(candidate_grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(candidate_grid[key] for key in keys))]


def best_candidate_window(
    groups: np.ndarray,
    starts: np.ndarray,
    probabilities: np.ndarray,
    labels: dict,
    raw_durations: dict,
    combos: list[dict[str, float]],
) -> tuple[dict[str, float], float]:
    """Sweep every candidate-window combo against one set of already-pooled
    probabilities -- no refitting -- and return the best (combo, partial recall-budget AUC (band 25-40%))."""
    best_combo, best_auc = combos[0], -np.inf
    for combo in combos:
        budgets, recalls, _ = recall_budget_curve(
            groups, starts, probabilities, labels, combo["merge_gap"],
            raw_durations=raw_durations, lookback=combo["lookback"], postroll=combo["postroll"],
        )
        auc = curve_partial_auc(budgets, recalls, *BUDGET_BAND)
        if auc > best_auc:
            best_combo, best_auc = combo, auc
    return best_combo, best_auc


def score_candidate(
    sampled_params: dict,
    data: TrainingData,
    args: argparse.Namespace,
    candidate_combos: list[dict[str, float]],
) -> dict:
    """Fit this candidate's params across GroupKFold folds ONCE (expensive),
    pool the OOF probabilities, then pick whichever lookback/postroll/
    merge_gap combination gives the best pooled partial recall-budget AUC
    (band 25-40%; cheap --
    reuses the same pooled probabilities, no refitting). std_test_score is
    then computed per-fold using that winning combo, for comparability with
    the pooled partial recall-budget AUC (band 25-40%)."""
    pooled_probabilities, held_positions = pooled_oof_predict(
        data.feature_matrix, data.targets, data.groups, sampled_params, args.random_state, args.cv,
    )

    best_combo, pooled_auc = best_candidate_window(
        data.groups, data.starts, pooled_probabilities, data.labels, data.raw_durations, candidate_combos,
    )

    best_budgets, best_recalls, _ = recall_budget_curve(
        data.groups, data.starts, pooled_probabilities, data.labels, best_combo["merge_gap"],
        raw_durations=data.raw_durations, lookback=best_combo["lookback"], postroll=best_combo["postroll"],
    )
    full_auc = curve_auc(best_budgets, best_recalls)
    band_points = count_points_in_band(best_budgets, *BUDGET_BAND)

    fold_aucs = []
    for held_pos in held_positions:
        budgets, recalls, _ = recall_budget_curve(
            data.groups[held_pos], data.starts[held_pos], pooled_probabilities[held_pos],
            data.labels, best_combo["merge_gap"], raw_durations=data.raw_durations,
            lookback=best_combo["lookback"], postroll=best_combo["postroll"],
        )
        fold_aucs.append(curve_partial_auc(budgets, recalls, *BUDGET_BAND))

    return {
        "params": sampled_params,
        "lookback": best_combo["lookback"],
        "postroll": best_combo["postroll"],
        "merge_gap": best_combo["merge_gap"],
        "mean_test_score": pooled_auc,
        "std_test_score": float(np.std(fold_aucs)),
        "full_auc": full_auc,
        "_band_points": band_points,
    }


def run_search(
    data: TrainingData,
    args: argparse.Namespace,
    candidate_combos: list[dict[str, float]],
) -> pd.DataFrame:
    results_rows = []
    sampler = ParameterSampler(MODEL_PARAM_DISTRIBUTIONS, n_iter=args.n_iter, random_state=args.random_state)
    for candidate_number, sampled_params in enumerate(sampler, start=1):
        result = score_candidate(sampled_params, data, args, candidate_combos)
        result["candidate_number"] = candidate_number
        results_rows.append(result)
        print(f"Candidate {candidate_number}/{args.n_iter}: partial recall-budget AUC (band 25-40%)="
              f"{result['mean_test_score']:.3f} "
              f"(lookback={result['lookback']:.0f}s postroll={result['postroll']:.0f}s "
              f"merge_gap={result['merge_gap']:.0f}s)")
        if candidate_number == 1:
            band_points = result["_band_points"]
            print(f"Best combo curve points in budget band 25-40%: {band_points}")
            if band_points < 15:
                print("[WARNING] Fewer than 15 curve points fall in the budget band; increase CURVE_N_THRESHOLDS.")

    results = pd.DataFrame(results_rows).sort_values(
        ["mean_test_score", "candidate_number"], ascending=[False, True]
    ).reset_index(drop=True)
    results["rank_test_score"] = np.arange(1, len(results) + 1)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=FEATURES_DIR,
                        help="Directory of per-match parquet feature files.")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="Labels CSV written by extract_labels.py; refined bounds train the model and original clip bounds score recall.")
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH,
                        help="matches.csv (match_id, raw_filename, ..., audio_length_sec).")
    parser.add_argument("--n-iter", type=int, default=N_ITER,
                        help="Number of sampled MODEL-hyperparameter combinations (default: 80). Each one "
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
