"""Evaluate test-set probabilities written by train_predict.py.

Reads the per-window probability CSV for TEST_MATCH_IDS (modelling/functions.py),
sweeps thresholds, and reports a recall-vs-budget curve (recall at a handful
of budget checkpoints, plus the full-range AUC).

There is no tiering in this project -- the old --tier-map breakdown is
removed; this reports the pooled curve over the full test set only.

Candidate-window settings (lookback/postroll/merge-gap) default to whichever
combination tune_model.py's grid search selected (loaded from --results, the
same file train_predict.py's model hyperparameters come from), so this
reports the curve for the settings actually shipped in the tuned model.
Override any of them individually with --lookback/--postroll/--merge-gap.

Usage:
    python -m modelling.eval_model --probabilities data/modelling/predictions/yamnet_audio_test_probabilities.csv \
        --labels data/processed/labels/labels.csv --matches data/metadata/matches.csv
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import (
    BUDGET_CHECKPOINTS, LABELS_PATH, MATCHES_PATH, MODEL_RESULTS_PATH,
    RECALL_BUDGET_PLOT_PATH, TEST_PROBABILITIES_PATH,
)

from modelling.functions import (
    TEST_MATCH_IDS, curve_auc, load_best_candidate_config, load_labels,
    load_raw_durations, recall_at_budget, recall_budget_curve,
)

def report_curve(
    name: str,
    probabilities_df: pd.DataFrame,
    labels: dict[int, list[tuple[float, float, str]]],
    raw_durations: dict[int, float],
    merge_gap: float,
    lookback: float,
    postroll: float,
    plot_output: Path | None = None,
) -> None:
    """Print the recall-vs-budget curve (at a few checkpoints) and its AUC."""
    groups = probabilities_df["match_id"].to_numpy()
    starts = probabilities_df["start_sec"].to_numpy()
    probabilities = probabilities_df["probability"].to_numpy()
    if len(groups) == 0:
        print(f"  [{name}] no rows, skipping")
        return

    budgets, recalls, _ = recall_budget_curve(
        groups, starts, probabilities, labels, merge_gap,
        raw_durations=raw_durations, lookback=lookback, postroll=postroll,
    )
    auc = curve_auc(budgets, recalls)
    checkpoints = ", ".join(
        f"{budget:.0%}\u2192{recall_at_budget(budgets, recalls, budget):.1%}" for budget in BUDGET_CHECKPOINTS
    )
    n_matches = probabilities_df["match_id"].nunique()
    print(f"  [{name}] n_matches={n_matches}  AUC={auc:.3f}")
    print(f"      recall @ budget: {checkpoints}")

    if plot_output is not None:
        plot_output.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(budgets, recalls, color="#1f77b4", linewidth=2.5)
        axis.fill_between(budgets, recalls, alpha=0.12, color="#1f77b4")
        axis.set(
            title="Recall-budget curve (held-out test set)",
            xlabel="Candidate footage budget",
            ylabel="Recall",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plot_output, dpi=160)
        plt.close(figure)
        print(f"      wrote plot -> {plot_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probabilities", type=Path, default=TEST_PROBABILITIES_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="Labels CSV written by extract_labels.py; original clip bounds are used for recall.")
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH,
                        help="matches.csv (match_id, raw_filename, ..., audio_length_sec).")
    parser.add_argument("--results", type=Path, default=MODEL_RESULTS_PATH,
                        help="Tuning results written by tune_model.py; supplies the default candidate window.")
    parser.add_argument("--plot-output", type=Path, default=RECALL_BUDGET_PLOT_PATH,
                        help="Path for the pooled held-out test-set recall-budget plot.")
    parser.add_argument("--lookback", type=float, default=None,
                        help="Override the tuned lookback (seconds); default: tuned value.")
    parser.add_argument("--postroll", type=float, default=None,
                        help="Override the tuned postroll (seconds); default: tuned value.")
    parser.add_argument("--merge-gap", type=float, default=None,
                        help="Override the tuned merge gap (seconds); default: tuned value.")
    args = parser.parse_args()

    probabilities_df = pd.read_csv(args.probabilities)
    required_columns = {"match_id", "start_sec", "probability"}
    missing_columns = required_columns - set(probabilities_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required probability columns: {sorted(missing_columns)}")

    expected_match_ids = set(TEST_MATCH_IDS)
    actual_match_ids = set(probabilities_df["match_id"].astype(int))
    unexpected_match_ids = actual_match_ids - expected_match_ids
    if unexpected_match_ids:
        raise ValueError(f"Probability file contains matches outside the test set: {sorted(unexpected_match_ids)}")

    tuned_window = load_best_candidate_config(args.results)
    lookback = args.lookback if args.lookback is not None else tuned_window["lookback"]
    postroll = args.postroll if args.postroll is not None else tuned_window["postroll"]
    merge_gap = args.merge_gap if args.merge_gap is not None else tuned_window["merge_gap"]
    print(f"Candidate window: lookback={lookback:.0f}s postroll={postroll:.0f}s merge_gap={merge_gap:.0f}s")

    labels = load_labels(args.labels, start_column="clip_start_sec", end_column="clip_end_sec")
    raw_durations = load_raw_durations(args.matches)

    print(f"Evaluating test-set probabilities: {args.probabilities}")
    print(f"Test matches: {sorted(expected_match_ids)}")
    report_curve("ALL TEST MATCHES", probabilities_df, labels, raw_durations,
                 merge_gap, lookback, postroll, args.plot_output)


if __name__ == "__main__":
    main()
