"""Shared functions for the modelling pipeline (tune_model.py, train_predict.py,
eval_model.py, save_final_model.py).

Feature extraction is fixed to a single configuration (see
preprocessing/extract_features.py): YAMNet's native frame cadence (0.96s
window / 0.48s stride), 11 raw YAMNet score columns, and a 16-dim
PCA-reduced embedding
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from constants import (
    CV_FOLDS, LOOKBACK_SEC, MERGE_GAP_SEC, MINIMUM_RECALL, N_ITER,
    MODEL_N_JOBS, POSTROLL_SEC, RANDOM_STATE, SCORE_INDICES, TEST_MATCH_IDS,
    YAMNET_STRIDE_SEC, YAMNET_WINDOW_SEC,
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
) -> dict[int, list[tuple[float, float, str]]]:
    """Load label intervals from the labels CSV produced by extract_labels.py.

    The refined interval columns are the default because they define training
    targets. Evaluation can request the original editor clip bounds explicitly.
    """
    rows = pd.read_csv(labels_csv)
    required = {"match_id", "description", start_column, end_column}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing required label columns in {labels_csv}: {sorted(missing)}")
    labels: dict[int, list[tuple[float, float, str]]] = {}
    for row in rows.itertuples(index=False):
        labels.setdefault(int(row.match_id), []).append(
            (float(getattr(row, start_column)), float(getattr(row, end_column)), str(row.description))
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

def interval_overlap(start: float, end: float, labels: list[tuple[float, float, str]]) -> bool:
    return any(start < label_end and end > label_start for label_start, label_end, _ in labels)


def merge_intervals(intervals: list[tuple[float, float]], merge_gap: float) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _fully_contained(interval: tuple[float, float], candidates: list[tuple[float, float]]) -> bool:
    """A label only counts as 'found' if some candidate window entirely
    brackets it."""
    start, end = interval
    return any(candidate_start <= start and candidate_end >= end for candidate_start, candidate_end in candidates)


def coverage_metrics(
    labels: list[tuple[float, float, str]],
    candidates: list[tuple[float, float]],
) -> tuple[int, int, float]:
    found = sum(_fully_contained((start, end), candidates) for start, end, _ in labels)
    candidate_seconds = sum(end - start for start, end in candidates)
    return found, len(labels), candidate_seconds


# --- Recall-vs-budget curve -----------------------------------------------------

def recall_budget_curve(
    groups: np.ndarray,
    starts: np.ndarray,
    probabilities: np.ndarray,
    labels: dict[int, list[tuple[float, float, str]]],
    merge_gap: float,
    raw_durations: dict[int, float],
    lookback: float = 0.0,
    postroll: float = 0.0,
    n_thresholds: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep decision thresholds and return (budgets, recalls, thresholds),
    sorted by ascending budget, covering the full [0, 1] budget range.

    budget = total merged candidate seconds / total raw match seconds.
    recall = fraction of labels fully contained by the merged candidate
    windows at that threshold. No label-description filtering happens here
    any more -- labels is already scoped to the single target label by
    load_labels()/extract_labels.py.
    """
    if not raw_durations:
        raise ValueError("raw_durations is required to compute footage budget")

    by_match: dict[int, tuple[np.ndarray, np.ndarray, list[tuple[float, float, str]]]] = {}
    for match_id in np.unique(groups):
        mask = groups == match_id
        by_match[int(match_id)] = (starts[mask], probabilities[mask], labels.get(int(match_id), []))

    raw_seconds = sum(raw_durations.get(match_id, 0.0) for match_id in by_match)
    if not raw_seconds:
        raise ValueError("raw_durations has no positive footage seconds for these matches")

    quantile_grid = np.quantile(probabilities, np.linspace(0.0, 1.0, n_thresholds))
    thresholds = np.unique(np.concatenate(([probabilities.max() + 1e-6], quantile_grid)))[::-1]

    budgets, recalls = [], []
    for threshold in thresholds:
        total_found = total_labels = 0
        total_seconds = 0.0
        for match_starts, match_probs, match_labels in by_match.values():
            intervals = merge_intervals(
                [(max(0, s - lookback), s + WINDOW_SEC + postroll)
                 for s, p in zip(match_starts, match_probs) if p >= threshold],
                merge_gap,
            )
            found, count, seconds = coverage_metrics(match_labels, intervals)
            total_found += found
            total_labels += count
            total_seconds += seconds
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


def build_targets(df: pd.DataFrame, labels: dict[int, list[tuple[float, float, str]]]) -> np.ndarray:
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
    import ast

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

@dataclass
class CandidateConfig:
    """Bundles the candidate-window-generation settings threaded through
    recall_budget_curve() calls, instead of passing lookback/postroll/
    merge_gap as separate arguments at every call site."""
    lookback: float
    postroll: float
    merge_gap: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CandidateConfig":
        return cls(lookback=args.lookback, postroll=args.postroll, merge_gap=args.merge_gap)
