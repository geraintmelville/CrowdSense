import numpy as np

from modelling.functions import (
    WINDOW_SEC,
    coverage_metrics,
    merge_intervals,
    recall_budget_curve,
)


def test_vectorized_recall_budget_curve_matches_interval_reference():
    groups = np.asarray([1, 1, 1, 1, 1, 2, 2])
    starts = np.asarray([0.0, 0.48, 1.44, 3.36, 5.28, 0.0, 1.44])
    probabilities = np.asarray([0.9, 0.4, 0.8, 0.2, 0.6, 0.7, 0.3])
    labels = {
        1: [(0.1, 1.2, "a"), (2.0, 2.5, "b"), (6.0, 7.0, "c")],
        2: [(0.2, 1.0, "d"), (3.0, 3.5, "e")],
    }
    raw_durations = {1: 10.0, 2: 8.0}
    lookback, postroll, merge_gap = 1.0, 0.5, 0.25

    expected_thresholds = np.unique(
        np.concatenate(
            ([probabilities.max() + 1e-6], np.quantile(probabilities, np.linspace(0, 1, 100)))
        )
    )[::-1]
    expected_budgets, expected_recalls = [], []
    for threshold in expected_thresholds:
        found_total = label_total = 0
        seconds_total = 0.0
        for match_id in np.unique(groups):
            match_mask = groups == match_id
            intervals = merge_intervals(
                [
                    (max(0, start - lookback), start + WINDOW_SEC + postroll)
                    for start, probability in zip(starts[match_mask], probabilities[match_mask])
                    if probability >= threshold
                ],
                merge_gap,
            )
            found, count, seconds = coverage_metrics(labels[int(match_id)], intervals)
            found_total += found
            label_total += count
            seconds_total += seconds
        expected_budgets.append(seconds_total / sum(raw_durations.values()))
        expected_recalls.append(found_total / label_total if label_total else 0.0)

    expected_budgets = np.asarray(expected_budgets)
    expected_recalls = np.asarray(expected_recalls)
    if expected_budgets[0] > 0:
        expected_budgets = np.concatenate(([0.0], expected_budgets))
        expected_recalls = np.concatenate(([0.0], expected_recalls))
        expected_thresholds = np.concatenate(([np.inf], expected_thresholds))
    if expected_budgets[-1] < 1:
        expected_budgets = np.concatenate((expected_budgets, [1.0]))
        expected_recalls = np.concatenate((expected_recalls, [expected_recalls[-1]]))
        expected_thresholds = np.concatenate((expected_thresholds, [0.0]))
    order = np.argsort(expected_budgets)
    expected = expected_budgets[order], expected_recalls[order], expected_thresholds[order]

    actual = recall_budget_curve(
        groups,
        starts,
        probabilities,
        labels,
        merge_gap,
        raw_durations,
        lookback=lookback,
        postroll=postroll,
    )

    for actual_values, expected_values in zip(actual, expected):
        np.testing.assert_allclose(actual_values, expected_values, rtol=0, atol=1e-12)