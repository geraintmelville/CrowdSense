"""Plot the pooled training-only OOF recall-budget curve for visual review.

The plot includes the 25-40% tuning band. Review its shape to choose a target
budget manually, then set TARGET_BUDGET in constants/constants.py.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import (
    BUDGET_BAND, CV_FOLDS, FEATURES_DIR, FINAL_CURVE_N_THRESHOLDS,
    LABELS_PATH, MATCHES_PATH, MODEL_PERFORMANCE_DIR, MODEL_RESULTS_PATH,
    RANDOM_STATE,
)
from modelling.functions import (
    count_points_in_band, curve_partial_auc, load_best_candidate_config,
    load_best_params, pooled_oof_predict, recall_budget_curve,
)
from modelling.tune_model import load_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=FEATURES_DIR,
                        help="Directory of per-match parquet feature files.")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="Labels CSV; clip bounds are used to score recall.")
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH,
                        help="matches.csv with raw audio durations.")
    parser.add_argument("--results", type=Path, default=MODEL_RESULTS_PATH,
                        help="Tuning results CSV with best parameters and candidate window.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--cv", type=int, default=CV_FOLDS,
                        help="Number of GroupKFold splits for pooled OOF predictions.")
    parser.add_argument("--output-plot", type=Path,
                        default=MODEL_PERFORMANCE_DIR / "oof_recall_budget.png",
                        help="Path for the pooled training OOF recall-budget plot.")
    args = parser.parse_args()

    data = load_training_data(args)
    model_params = load_best_params(args.results)
    candidate = load_best_candidate_config(args.results)
    print(f"Loaded best parameters from {args.results}: {model_params}")
    print(f"Using tuned candidate window: {candidate}")

    probabilities, _ = pooled_oof_predict(
        data.feature_matrix,
        data.targets,
        data.groups,
        model_params,
        args.random_state,
        args.cv,
    )
    budgets, recalls, _ = recall_budget_curve(
        data.groups,
        data.starts,
        probabilities,
        data.labels,
        candidate["merge_gap"],
        raw_durations=data.raw_durations,
        lookback=candidate["lookback"],
        postroll=candidate["postroll"],
        n_thresholds=FINAL_CURVE_N_THRESHOLDS,
    )
    partial_auc = curve_partial_auc(budgets, recalls, *BUDGET_BAND)
    band_points = count_points_in_band(budgets, *BUDGET_BAND)

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(budgets, recalls, color="#1f77b4", linewidth=2.5,
              label="Pooled training OOF curve")
    axis.fill_between(budgets, recalls, alpha=0.10, color="#1f77b4")
    axis.axvspan(*BUDGET_BAND, color="#ffbf00", alpha=0.18,
                 label="Tuning band (25-40%)")
    axis.set(
        title="Training-only pooled OOF recall-budget curve",
        xlabel="Candidate footage budget",
        ylabel="Recall",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_plot, dpi=160)
    plt.close(figure)

    print(f"Partial recall-budget AUC (band 25-40%): {partial_auc:.3f}")
    print(f"Curve points in budget band 25-40%: {band_points}")
    print(f"Saved training-only OOF plot: {args.output_plot}")
    print("Review the plot, then set TARGET_BUDGET manually in constants/constants.py.")


if __name__ == "__main__":
    main()
