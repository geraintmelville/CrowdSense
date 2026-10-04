"""Shared functions for the modelling pipeline (tune_model.py,
eval_model.py, save_final_model.py).

Feature extraction is fixed to a single configuration (see
preprocessing/extract_features.py): YAMNet's native frame cadence (0.96s
window / 0.48s stride), 11 raw YAMNet score columns, and a 16-dim
PCA-reduced embedding
"""

from pathlib import Path
import ast
from dataclasses import dataclass
import itertools
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterSampler
from xgboost import XGBClassifier

from constants import (
    BUDGET_BAND, CURVE_N_THRESHOLDS, DEMO_CANDIDATE_BUDGET,
    FINAL_CURVE_N_THRESHOLDS, MODEL_N_JOBS, MODEL_PARAM_DISTRIBUTIONS,
    SCORE_INDICES, TEST_MATCH_IDS, YAMNET_STRIDE_SEC, YAMNET_WINDOW_SEC,
)


# --- Fixed feature-extraction config --------------------------------------------
# These values are taken from the preprocessing pipeline so the modelling code
# stays aligned with the exact YAMNet feature layout the extractor writes.
WINDOW_SEC = YAMNET_WINDOW_SEC
STRIDE_SEC = YAMNET_STRIDE_SEC
YAMNET_SCORE_INDICES = tuple(SCORE_INDICES)


# --- Label loading -------------------------------------------------------------

def load_labels(
    labels_csv: Path,
    start_column: str = "label_start_sec",
    end_column: str = "label_end_sec",
) -> dict[int, list[tuple[float, float]]]:
    """Load label intervals from the labels CSV produced by extract_labels.py.

    The refined interval columns are the default because they define training
    targets. Evaluation can request the original editor clip bounds explicitly.
    """
    rows = pd.read_csv(labels_csv)
    required = {"match_id", start_column, end_column}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing required label columns in {labels_csv}: {sorted(missing)}")
    labels: dict[int, list[tuple[float, float]]] = {}
    for row in rows.itertuples(index=False):
        labels.setdefault(int(row.match_id), []).append(
            (float(getattr(row, start_column)), float(getattr(row, end_column)))
        )
    return labels


def load_raw_durations(matches_csv: Path) -> dict[int, float]:
    """Read match_id -> audio_length_sec directly from matches.csv (no DB)."""
    matches = pd.read_csv(matches_csv).dropna(subset=["audio_length_sec"])
    return {
        int(match_id): float(duration)
        for match_id, duration in zip(matches["match_id"], matches["audio_length_sec"])
    }


# --- Intervals & coverage -------------------------------------------------------

def interval_overlap(start: float, end: float, labels: list[tuple[float, float]]) -> bool:
    return any(start < label_end and end > label_start for label_start, label_end in labels)


def merge_intervals(intervals: list[tuple[float, float]], merge_gap: float) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def format_timestamp(seconds: float) -> str:
    """Format a time in seconds as HH:MM:SS."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def score_feature_matrix(model: XGBClassifier, feature_matrix: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for a model-ready feature matrix."""
    return model.predict_proba(np.asarray(feature_matrix, dtype=np.float32))[:, 1]


def build_candidate_clips(
    starts: np.ndarray,
    probabilities: np.ndarray,
    duration: float,
    bundle: dict,
    target_budget: float = DEMO_CANDIDATE_BUDGET,
) -> tuple[pd.DataFrame, float, int]:
    """Turn scored windows into padded, merged candidate clips."""
    def make_clips(threshold: float) -> pd.DataFrame:
        selected_starts = starts[probabilities >= threshold]
        intervals = merge_intervals(
            [(max(0, start - bundle["lookback"]),
              min(duration, start + bundle["window_sec"] + bundle["postroll"]))
             for start in selected_starts],
            bundle["merge_gap"],
        )
        return pd.DataFrame(intervals, columns=["start_sec", "end_sec"])

    breakpoints = np.unique(probabilities)
    if len(breakpoints):
        low, high = 0, len(breakpoints)
        while low < high:
            middle = (low + high) // 2
            candidate = make_clips(float(breakpoints[middle]))
            seconds = candidate["end_sec"].sub(candidate["start_sec"]).sum()
            if (seconds / duration if duration else 0.0) <= target_budget:
                high = middle
            else:
                low = middle + 1
        choices = []
        for index in {max(0, low - 1), min(low, len(breakpoints) - 1)}:
            threshold = float(breakpoints[index])
            candidate = make_clips(threshold)
            seconds = candidate["end_sec"].sub(candidate["start_sec"]).sum()
            budget = seconds / duration if duration else 0.0
            choices.append((abs(budget - target_budget), threshold, candidate))
        empty_threshold = float(np.nextafter(breakpoints[-1], np.inf))
        choices.append((target_budget, empty_threshold, make_clips(empty_threshold)))
        _, threshold, clips = min(choices, key=lambda item: (item[0], item[1]))
    else:
        threshold = float(bundle["threshold"])
        clips = make_clips(threshold)

    clips["length_sec"] = clips["end_sec"] - clips["start_sec"]
    clips["start"] = clips["start_sec"].apply(format_timestamp)
    clips["end"] = clips["end_sec"].apply(format_timestamp)
    clips.attrs["threshold"] = threshold
    clips.attrs["target_budget"] = target_budget
    return clips, duration, len(starts)


def _fully_contained(interval: tuple[float, float], candidates: list[tuple[float, float]]) -> bool:
    """A label only counts as 'found' if some candidate window entirely
    brackets it."""
    start, end = interval
    return any(candidate_start <= start and candidate_end >= end for candidate_start, candidate_end in candidates)


def coverage_metrics(
    labels: list[tuple[float, float]],
    candidates: list[tuple[float, float]],
) -> tuple[int, int, float]:
    found = sum(_fully_contained((start, end), candidates) for start, end in labels)
    candidate_seconds = sum(end - start for start, end in candidates)
    return found, len(labels), candidate_seconds


# --- Recall-vs-budget curve -----------------------------------------------------

def recall_budget_curve(
    groups: np.ndarray,
    starts: np.ndarray,
    probabilities: np.ndarray,
    labels: dict[int, list[tuple[float, float]]],
    merge_gap: float,
    raw_durations: dict[int, float],
    lookback: float = 0.0,
    postroll: float = 0.0,
    n_thresholds: int = CURVE_N_THRESHOLDS,
    extra_thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep decision thresholds and return (budgets, recalls, thresholds),
    sorted by ascending budget, covering the full [0, 1] budget range.

    budget = total merged candidate seconds / total raw match seconds.
    recall = fraction of labels fully contained by the merged candidate
    windows at that threshold. Labels are already scoped to the single target
    signal by load_labels()/extract_labels.py.
    """
    if not raw_durations:
        raise ValueError("raw_durations is required to compute footage budget")

    by_match: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for match_id in np.unique(groups):
        mask = groups == match_id
        match_labels = labels.get(int(match_id), [])
        label_starts = np.fromiter((start for start, _ in match_labels), dtype=float)
        label_ends = np.fromiter((end for _, end in match_labels), dtype=float)
        by_match[int(match_id)] = (starts[mask], probabilities[mask], label_starts, label_ends)

    raw_seconds = sum(raw_durations.get(match_id, 0.0) for match_id in by_match)
    if not raw_seconds:
        raise ValueError("raw_durations has no positive footage seconds for these matches")

    levels = 1.0 - np.geomspace(1e-4, 0.5, n_thresholds)
    quantile_grid = np.quantile(probabilities, levels)
    threshold_values = [probabilities.max() + 1e-6, *quantile_grid]
    if extra_thresholds is not None:
        threshold_values.extend(np.asarray(extra_thresholds, dtype=float).reshape(-1))
    thresholds = np.unique(threshold_values)[::-1]

    budgets, recalls = [], []
    for threshold in thresholds:
        total_found = total_labels = 0
        total_seconds = 0.0
        for match_starts, match_probs, label_starts, label_ends in by_match.values():
            selected_starts = np.sort(match_starts[match_probs >= threshold].astype(float))
            total_labels += len(label_starts)
            if not selected_starts.size:
                continue

            interval_starts = np.maximum(0.0, selected_starts - lookback)
            interval_ends = selected_starts + WINDOW_SEC + postroll
            breaks = np.flatnonzero(interval_starts[1:] > interval_ends[:-1] + merge_gap) + 1
            group_starts = np.concatenate(([0], breaks))
            group_ends = np.concatenate((breaks - 1, [len(selected_starts) - 1]))
            merged_starts = interval_starts[group_starts]
            merged_ends = interval_ends[group_ends]

            total_seconds += np.sum(merged_ends - merged_starts)
            containing = np.searchsorted(merged_starts, label_starts, side="right") - 1
            valid = containing >= 0
            total_found += np.count_nonzero(
                valid & (merged_ends[np.maximum(containing, 0)] >= label_ends)
            )
        budgets.append(total_seconds / raw_seconds)
        recalls.append(total_found / total_labels if total_labels else 0.0)

    budgets = np.array(budgets)
    recalls = np.array(recalls)
    thresholds = np.array(thresholds)

    # Anchor the curve across the full [0, 1] budget range: below the lowest
    # swept threshold, recall is flat at whatever was achieved by selecting
    # every window -- there's no operating point beyond that, but for the
    # AUC/plot it's neither a gain nor a loss to extend it out to budget=1.
    if budgets[0] > 0.0:
        budgets = np.concatenate(([0.0], budgets))
        recalls = np.concatenate(([0.0], recalls))
        thresholds = np.concatenate(([np.inf], thresholds))
    if budgets[-1] < 1.0:
        budgets = np.concatenate((budgets, [1.0]))
        recalls = np.concatenate((recalls, [recalls[-1]]))
        thresholds = np.concatenate((thresholds, [0.0]))

    order = np.argsort(budgets)
    return budgets[order], recalls[order], thresholds[order]


def curve_auc(budgets: np.ndarray, recalls: np.ndarray) -> float:
    """Trapezoidal area under a recall-vs-budget curve."""
    order = np.argsort(budgets)
    return float(np.trapezoid(np.asarray(recalls)[order], np.asarray(budgets)[order]))


def curve_partial_auc(
    budgets: np.ndarray,
    recalls: np.ndarray,
    lo: float,
    hi: float,
    n: int = 101,
) -> float:
    """Return mean interpolated recall over the requested budget interval."""
    order = np.argsort(budgets)
    sorted_budgets = np.asarray(budgets)[order]
    sorted_recalls = np.asarray(recalls)[order]
    if sorted_budgets.size == 0:
        raise ValueError("At least one curve point is required")
    if hi <= lo:
        raise ValueError(f"Expected hi > lo, got lo={lo} and hi={hi}")
    if sorted_budgets.max() < hi:
        warnings.warn(
            f"Recall-budget curve ends at budget {sorted_budgets.max():.3f}, below the requested upper bound {hi:.3f}; "
            "np.interp will extend the final recall value across the remainder of the band.",
            RuntimeWarning,
            stacklevel=2,
        )
    grid = np.linspace(lo, hi, n)
    interpolated = np.interp(grid, sorted_budgets, sorted_recalls)
    return float(np.trapezoid(interpolated, grid) / (hi - lo))


def count_points_in_band(budgets: np.ndarray, lo: float, hi: float) -> int:
    """Count curve points whose budget lies in the inclusive [lo, hi] band."""
    values = np.asarray(budgets)
    return int(np.count_nonzero((values >= lo) & (values <= hi)))


def recall_at_budget(budgets: np.ndarray, recalls: np.ndarray, budget: float) -> float:
    """Linearly interpolate recall at a specific budget level, for reporting
    (e.g. 'recall at 20% budget') alongside the headline AUC."""
    order = np.argsort(budgets)
    return float(np.interp(budget, np.asarray(budgets)[order], np.asarray(recalls)[order]))


# --- Model construction & OOF pooling -------------------------------------------

def build_model(params: dict, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=MODEL_N_JOBS,
        random_state=random_state,
        **params,
    )


def required_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the canonical modelling feature list, validating it is present."""
    selected = feature_columns(df)
    if not selected:
        raise ValueError("No recognized feature columns found in the feature dataframe")
    return selected


def train_test_split_by_match_id(groups: np.ndarray, test_match_ids: set[int] | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for a given holdout match set."""
    test_match_ids = set(int(match_id) for match_id in test_match_ids)
    available = set(int(match_id) for match_id in np.unique(groups))
    missing = test_match_ids - available
    if missing:
        raise RuntimeError(f"Test matches are missing from features: {sorted(missing)}")

    train_idx = np.flatnonzero(~np.isin(groups, list(test_match_ids)))
    test_idx = np.flatnonzero(np.isin(groups, list(test_match_ids)))
    if not train_idx.size or not test_idx.size:
        raise RuntimeError("Need both training matches and test matches")
    return train_idx, test_idx


def pooled_oof_predict(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    model_params: dict,
    random_state: int,
    n_folds: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """GroupKFold out-of-fold pooling: fit on each fold's fit split, predict
    on its held-out split, pool so every row gets a score from a model that
    never saw it. Returns (pooled_probabilities, held_out_positions_per_fold)
    so callers can also compute a per-fold score (e.g. for std_test_score)."""
    n_available_groups = np.unique(groups).size
    folds = max(2, min(n_folds, n_available_groups))
    if folds < n_folds:
        print(f"[WARNING] Only {n_available_groups} groups available; reducing folds from {n_folds} to {folds}.")
    splitter = GroupKFold(n_splits=folds)
    pooled = np.full(len(targets), np.nan, dtype=float)
    held_positions = []
    for fit_pos, held_pos in splitter.split(feature_matrix, targets, groups=groups):
        if np.unique(targets[fit_pos]).size < 2:
            raise RuntimeError("A fold's fit split has only one class -- cannot train")
        model = build_model(model_params, random_state)
        model.fit(feature_matrix[fit_pos], targets[fit_pos])
        pooled[held_pos] = model.predict_proba(feature_matrix[held_pos])[:, 1]
        held_positions.append(held_pos)
    if np.isnan(pooled).any():
        raise RuntimeError("Some rows never fell into a fold's held-out split")
    return pooled, held_positions


# --- Feature loading & selection -------------------------------------------------

def load_features(features_dir: Path) -> pd.DataFrame:
    """Load and concatenate the per-match parquet files written by
    extract_features.py. Window/stride are fixed (WINDOW_SEC/STRIDE_SEC
    above), so unlike the old single-CSV-plus-.meta.csv setup there's nothing
    to read back about the run configuration -- just the rows."""
    features_dir = Path(features_dir)
    parquet_paths = sorted(features_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No .parquet feature files found in {features_dir}")
    return pd.concat((pd.read_parquet(path) for path in parquet_paths), ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Select the feature columns: the 11 canonical YAMNet score columns plus
    the 16 PCA embedding columns. No legacy hand-crafted-feature fallback --
    every feature cache post-YAMNet has this shape."""
    score_columns = [f"yamnet_score_{index:03d}" for index in YAMNET_SCORE_INDICES]
    selected_score_columns = [column for column in score_columns if column in df.columns]
    pca_columns = sorted(column for column in df.columns if column.startswith("yamnet_embedding_pca_"))
    return selected_score_columns + pca_columns


def build_targets(df: pd.DataFrame, labels: dict[int, list[tuple[float, float]]]) -> np.ndarray:
    targets = np.zeros(len(df), dtype=int)
    for match_id, group in df.groupby("match_id"):
        match_labels = labels.get(int(match_id), [])
        if not match_labels:
            continue
        idx = group.index.to_numpy()
        starts = group["start_sec"].to_numpy()
        overlaps = np.array(
            [interval_overlap(float(start), float(start) + WINDOW_SEC, match_labels) for start in starts]
        )
        targets[idx] = overlaps.astype(int)
    return targets


def load_best_params(results_path: Path) -> dict[str, object]:
    """Load the winning parameter dictionary written by tune_model.py."""
    results = pd.read_csv(results_path)
    if results.empty or "params" not in results.columns:
        raise ValueError(f"No tuned parameters found in {results_path}")
    try:
        params = ast.literal_eval(results.iloc[0]["params"])
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"Invalid params value in {results_path}") from error
    if not isinstance(params, dict):
        raise ValueError(f"Expected a parameter dictionary in {results_path}")
    return params

def load_best_candidate_config(results_path: Path) -> dict[str, float]:
    """Load the winning lookback/postroll/merge_gap combo tune_model.py found
    alongside the winning model hyperparameters."""
    results = pd.read_csv(results_path)
    required = {"lookback", "postroll", "merge_gap"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(
            f"{results_path} has no {sorted(missing)} column(s) -- re-run tune_model.py "
            "with the candidate-window grid search to populate them."
        )
    row = results.iloc[0]
    return {"lookback": float(row["lookback"]), "postroll": float(row["postroll"]), "merge_gap": float(row["merge_gap"])}


def save_model_artifact(model: XGBClassifier, metadata: dict, model_path: Path) -> None:
    """Save the trained model and inference metadata sidecar."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    with model_path.with_suffix(".json").open("w", encoding="utf-8") as config_file:
        json.dump(metadata, config_file, indent=2)


def load_model_artifact(model_path: str | Path) -> dict:
    """Load a saved model and its inference metadata sidecar."""
    model_path = Path(model_path)
    model = XGBClassifier()
    model.load_model(model_path)
    with model_path.with_suffix(".json").open(encoding="utf-8") as config_file:
        metadata = json.load(config_file)
    return {"model": model, **metadata}


@dataclass
class TrainingData:
    feature_matrix: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    starts: np.ndarray
    labels: dict
    raw_durations: dict


def load_training_data(args) -> TrainingData:
    """Load and split the feature/label inputs used during parameter tuning."""
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
        targets=targets[train_idx], groups=groups[train_idx],
        starts=df["start_sec"].to_numpy()[train_idx], labels=clip_labels,
        raw_durations=raw_durations,
    )


def candidate_window_combos(candidate_grid: dict[str, list[float]]) -> list[dict[str, float]]:
    """Expand a candidate-window grid into its parameter combinations."""
    keys = list(candidate_grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(candidate_grid[key] for key in keys))]


def best_candidate_window(groups, starts, probabilities, labels, raw_durations, combos):
    """Choose the candidate window with the best partial recall-budget AUC."""
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


def score_candidate(sampled_params, data, args, candidate_combos):
    """Fit grouped OOF models, then score this candidate's best window settings."""
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
    fold_aucs = []
    for held_pos in held_positions:
        budgets, recalls, _ = recall_budget_curve(
            data.groups[held_pos], data.starts[held_pos], pooled_probabilities[held_pos],
            data.labels, best_combo["merge_gap"], raw_durations=data.raw_durations,
            lookback=best_combo["lookback"], postroll=best_combo["postroll"],
        )
        fold_aucs.append(curve_partial_auc(budgets, recalls, *BUDGET_BAND))
    return {
        "params": sampled_params, "lookback": best_combo["lookback"],
        "postroll": best_combo["postroll"], "merge_gap": best_combo["merge_gap"],
        "mean_test_score": pooled_auc, "std_test_score": float(np.std(fold_aucs)),
        "full_auc": curve_auc(best_budgets, best_recalls),
        "_band_points": count_points_in_band(best_budgets, *BUDGET_BAND),
    }


def run_search(data, args, candidate_combos) -> pd.DataFrame:
    """Run randomized model search and return rows sorted by partial AUC."""
    results_rows = []
    sampler = ParameterSampler(MODEL_PARAM_DISTRIBUTIONS, n_iter=args.n_iter, random_state=args.random_state)
    for candidate_number, sampled_params in enumerate(sampler, start=1):
        result = score_candidate(sampled_params, data, args, candidate_combos)
        result["candidate_number"] = candidate_number
        results_rows.append(result)
        print(f"Candidate {candidate_number}/{args.n_iter}: partial recall-budget AUC (band 25-40%)="
              f"{result['mean_test_score']:.3f} (lookback={result['lookback']:.0f}s "
              f"postroll={result['postroll']:.0f}s merge_gap={result['merge_gap']:.0f}s)")
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


def select_threshold(groups, starts, probabilities, labels, raw_durations,
                     merge_gap, lookback, postroll, budget_limit):
    """Choose the OOF threshold whose candidate budget is closest to target."""
    if not 0.0 <= budget_limit <= 1.0:
        raise ValueError(f"budget must be between 0 and 1, got {budget_limit}")
    budgets, recalls, thresholds = recall_budget_curve(
        groups, starts, probabilities, labels, merge_gap, raw_durations=raw_durations,
        lookback=lookback, postroll=postroll, n_thresholds=FINAL_CURVE_N_THRESHOLDS,
    )
    eligible = np.flatnonzero(budgets > 1e-12)
    if not eligible.size:
        raise RuntimeError("No non-empty threshold is available; adjust the candidate-window settings.")
    distance = np.abs(budgets[eligible] - budget_limit)
    closest = eligible[np.isclose(distance, distance.min())]
    position = int(closest[np.argmax(recalls[closest])])
    return float(thresholds[position]), float(recalls[position]), float(budgets[position])
