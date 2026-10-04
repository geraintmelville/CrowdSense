"""Evaluate the saved final model on the held-out test matches.

Run save_final_model.py after tuning, then run this script to score the test
features, report the recall-budget curve, and save its plot. The operating
threshold is selected from training-only out-of-fold predictions and loaded
from the model artifact; test labels are used only for evaluation.

Usage:
    python -m modelling.save_final_model
    python -m modelling.eval_model
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import (
    BUDGET_BAND, BUDGET_CHECKPOINTS, FEATURES_DIR, LABELS_PATH, MATCHES_PATH,
    MODEL_PATH, RECALL_BUDGET_PLOT_PATH, TARGET_BUDGET,
)
from modelling.functions import (
    TEST_MATCH_IDS, curve_partial_auc, load_features, load_labels,
    load_raw_durations, recall_at_budget, recall_budget_curve,
    train_test_split_by_match_id,
)
from modelling.model_artifact import load_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=FEATURES_DIR,
                        help="Directory of per-match parquet feature files.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="Saved model artifact created by save_final_model.py.")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="Labels CSV; original clip bounds are used to evaluate recall.")
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH,
                        help="matches.csv containing raw audio durations.")
    parser.add_argument("--plot-output", type=Path, default=RECALL_BUDGET_PLOT_PATH,
                        help="Path for the held-out test recall-budget plot.")
    args = parser.parse_args()

    bundle = load_model_artifact(args.model)
    required_metadata = {"threshold", "feature_columns", "lookback", "postroll", "merge_gap"}
    missing_metadata = required_metadata - bundle.keys()
    if missing_metadata:
        raise ValueError(f"Model artifact is missing required metadata: {sorted(missing_metadata)}")

    df = load_features(args.features)
    _, test_idx = train_test_split_by_match_id(df["match_id"].to_numpy(), TEST_MATCH_IDS)
    test_df = df.iloc[test_idx]
    missing_features = set(bundle["feature_columns"]) - set(test_df.columns)
    if missing_features:
        raise ValueError(f"Test features are missing model columns: {sorted(missing_features)}")

    model = bundle["model"]
    probabilities = model.predict_proba(
        test_df[bundle["feature_columns"]].to_numpy(dtype=np.float32)
    )[:, 1]
    groups = test_df["match_id"].to_numpy()
    starts = test_df["start_sec"].to_numpy()
    labels = load_labels(args.labels, start_column="clip_start_sec", end_column="clip_end_sec")
    raw_durations = load_raw_durations(args.matches)

    threshold = float(bundle["threshold"])
    budgets, recalls, thresholds = recall_budget_curve(
        groups, starts, probabilities, labels, float(bundle["merge_gap"]),
        raw_durations=raw_durations, lookback=float(bundle["lookback"]),
        postroll=float(bundle["postroll"]), extra_thresholds=np.asarray([threshold]),
    )
    operating_idx = int(np.flatnonzero(np.isclose(thresholds, threshold, rtol=1e-12, atol=1e-15))[0])
    operating_budget = float(budgets[operating_idx])
    operating_recall = float(recalls[operating_idx])
    partial_auc = curve_partial_auc(budgets, recalls, *BUDGET_BAND)
    checkpoints = ", ".join(
        f"{budget:.0%}→{recall_at_budget(budgets, recalls, budget):.1%}"
        for budget in BUDGET_CHECKPOINTS
    )

    print(f"Evaluating saved model: {args.model}")
    print(f"Test matches: {sorted(TEST_MATCH_IDS)}")
    print(f"Candidate window: lookback={bundle['lookback']:.0f}s postroll={bundle['postroll']:.0f}s "
          f"merge_gap={bundle['merge_gap']:.0f}s")
    print(f"Training-selected threshold: {threshold:.6f} "
          f"(target budget={bundle.get('target_budget', TARGET_BUDGET):.1%}; "
          f"test budget={operating_budget:.1%}, test recall={operating_recall:.1%})")
    print(f"Test partial recall-budget AUC (band {BUDGET_BAND[0]:.0%}-{BUDGET_BAND[1]:.0%}): {partial_auc:.3f}")
    print(f"Test recall @ budget: {checkpoints}")

    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(budgets, recalls, color="#1f77b4", linewidth=2.5, label="Held-out test curve")
    axis.fill_between(budgets, recalls, alpha=0.12, color="#1f77b4")
    axis.axvspan(*BUDGET_BAND, color="#ffbf00", alpha=0.18, label="Tuning band (25-40%)")
    axis.scatter([operating_budget], [operating_recall], color="#d62728", zorder=3,
                 label="Headline metric threshold")
    axis.set(
        title="Recall-budget curve (held-out test set)",
        xlabel="Budget",
        ylabel="Recall",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.plot_output, dpi=160)
    plt.close(figure)
    print(f"Saved test recall-budget plot: {args.plot_output}")


if __name__ == "__main__":
    main()
