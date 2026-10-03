"""Streamlit entry point for local demo scoring and highlight review."""

import argparse
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import streamlit as st

from constants import DEMO_DIR, DEMO_FEATURES_DIR, DEMO_RAW_AUDIO_DIR, MODEL_PATH
from modelling.inference import format_timestamp, score_match, score_precomputed_features
from modelling.model_artifact import load_model_artifact

DEFAULT_DEMO_YOUTUBE_URL = "https://youtu.be/W8DhX4CIdKM"


@st.cache_resource
def load_bundle(bundle_path: str) -> dict:
    return load_model_artifact(bundle_path)


def _youtube_timestamp_url(video_url: str, start_seconds: float) -> str:
    parts = urlsplit(video_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["t"] = f"{int(start_seconds)}s"
    return urlunsplit(parts._replace(query=urlencode(query)))


def _format_mmss(seconds: float) -> str:
    minutes, secs = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def _goal_intervals(result_key: str) -> list[tuple[float, float]]:
    """Load editor-labelled goal spans for the match currently being shown."""
    clips_path = DEMO_DIR / "clips.csv"
    matches_path = DEMO_DIR / "matches.csv"
    if not clips_path.exists() or not matches_path.exists():
        return []

    match_name = Path(result_key.split(":", 1)[1]).stem
    matches = pd.read_csv(matches_path)
    matched = matches[matches["raw_filename"].map(lambda name: Path(name).stem) == match_name]
    if matched.empty:
        return []
    match_id = matched.iloc[0]["match_id"]
    labels = pd.read_csv(clips_path)
    labels = labels[labels["match_id"] == match_id]
    intervals = []
    for label in labels.itertuples(index=False):
        start = sum(int(part) * factor for part, factor in zip(label.timestamp_formatted.split(":"), (3600, 60, 1)))
        intervals.append((float(start), float(start) + float(label.length_sec)))
    return intervals


def _render_results(result: dict, youtube_url: str) -> None:
    clips = result["clips"]
    goals = _goal_intervals(result["result_key"])
    hits_by_candidate = []
    hit_goals = set()
    for _, candidate in clips.iterrows():
        hits = [
            goal_index for goal_index, (goal_start, goal_end) in enumerate(goals)
            if candidate["start_sec"] <= goal_start and goal_end <= candidate["end_sec"]
        ]
        hits_by_candidate.append(hits)
        hit_goals.update(hits)

    candidate_seconds = float(clips["length_sec"].sum()) if not clips.empty else 0.0
    budget_percent = candidate_seconds / result["duration"] if result["duration"] else 0.0
    cols = st.columns(3)
    cols[0].metric("Confirmed budget", f"{_format_mmss(candidate_seconds)} ({budget_percent:.1%})")
    cols[1].metric("Recall", f"{len(hit_goals)} / {len(goals)} goals" if goals else "Unavailable")
    threshold = clips.attrs.get("threshold")
    cols[2].metric("Selected threshold", f"{threshold:.3f}" if threshold is not None else "Unavailable")

    st.success(
        f"Scored {result['n_windows']} windows across "
        f"{format_timestamp(result['duration'])} of footage."
    )
    if clips.empty:
        st.warning("No candidate windows cleared the threshold.")
        return

    table = pd.DataFrame({
        "Clip": range(1, len(clips) + 1),
        "Start–End": [
            f"{_format_mmss(row.start_sec)}-{_format_mmss(row.end_sec)}"
            for row in clips.itertuples()
        ],
        "Duration": clips["length_sec"].map(_format_mmss),
        "Goal/No Goal": ["Goal" if hits else "No Goal" for hits in hits_by_candidate],
    })
    if youtube_url:
        table["YouTube"] = clips["start_sec"].map(
            lambda start: _youtube_timestamp_url(youtube_url, start)
        )
        st.dataframe(
            table,
            column_config={"YouTube": st.column_config.LinkColumn(display_text="Open")},
            hide_index=True,
            width="stretch",
        )
    else:
        st.dataframe(table, hide_index=True, width="stretch")
        st.info("Set CROWDSENSE_DEMO_YOUTUBE_URL to enable timestamped YouTube playback.")

    selected_index = st.selectbox(
        "Select a candidate to review",
        options=clips.index,
        format_func=lambda index: (
            f"{clips.loc[index, 'start']} to {clips.loc[index, 'end']} "
            f"({clips.loc[index, 'length_sec']:.1f}s)"
        ),
    )
    if selected_index is not None and youtube_url:
        st.video(youtube_url, start_time=int(clips.loc[selected_index, "start_sec"]))


def render_quick_demo(bundle: dict, youtube_url: str) -> None:
    feature_files = sorted(path for path in DEMO_FEATURES_DIR.glob("*.parquet") if path.is_file())
    if not feature_files:
        st.info(
            f"No pre-extracted feature files found in {DEMO_FEATURES_DIR}. "
            "Prepare them with `python -m demo.prepare_demo` before using Quick demo."
        )
        return

    feature_file = st.selectbox(
        "Pre-extracted match features", feature_files, format_func=lambda path: path.stem
    )
    result_key = f"quick:{feature_file}"
    if st.button("Score pre-extracted features", type="primary"):
        try:
            features = pd.read_parquet(feature_file)
            if features.empty or "start_sec" not in features.columns:
                raise ValueError("Precomputed feature file contains no valid windows")
            duration = float(features["start_sec"].max() + bundle["window_sec"])
            with st.spinner("Scoring cached features..."):
                clips, duration, n_windows = score_precomputed_features(features, bundle, duration)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Quick demo failed: {error}")
            return
        st.session_state["demo_result"] = {
            "result_key": result_key,
            "clips": clips,
            "duration": duration,
            "n_windows": n_windows,
        }

    result = st.session_state.get("demo_result")
    if result and result.get("result_key") == result_key:
        _render_results(result, youtube_url)


def render_full_demo(bundle: dict, youtube_url: str) -> None:
    audio_files = sorted(path for path in DEMO_RAW_AUDIO_DIR.glob("*.wav") if path.is_file())
    if not audio_files:
        st.info(
            f"Add the full-match WAV to {DEMO_RAW_AUDIO_DIR} to run the full demo. "
            "Prepare it from a match video with `python -m demo.prepare_demo`."
        )
        return

    audio_path = st.selectbox("Full match audio", audio_files, format_func=lambda path: path.stem)

    if st.button("Extract features and run analysis", type="primary"):
        try:
            with st.spinner("Extracting YAMNet features from the full-match audio and scoring..."):
                clips, duration, n_windows = score_match(audio_path, bundle)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Full demo failed: {error}")
            return

        st.session_state["demo_result"] = {
            "result_key": f"full:{audio_path}",
            "clips": clips,
            "duration": duration,
            "n_windows": n_windows,
        }

    result = st.session_state.get("demo_result")
    result_key = f"full:{audio_path}"
    if result and result.get("result_key") == result_key:
        _render_results(result, youtube_url)


def render_demo(bundle: dict) -> None:
    st.title("Audio Highlight Candidate Finder")
    mode = st.radio("Demo mode", ["Quick demo", "Full demo"], horizontal=True)
    youtube_url = os.environ.get("CROWDSENSE_DEMO_YOUTUBE_URL", DEFAULT_DEMO_YOUTUBE_URL).strip()
    if mode == "Quick demo":
        render_quick_demo(bundle, youtube_url)
    else:
        render_full_demo(bundle, youtube_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=MODEL_PATH)
    known_args, _ = parser.parse_known_args()
    return known_args


def render_dashboard() -> None:
    args = parse_args()
    if not args.bundle.exists():
        st.error(f"Model artifact not found: {args.bundle}. Run save_final_model.py first.")
        return
    config_path = args.bundle.with_suffix(".json")
    if not config_path.exists():
        st.error(f"Model configuration not found: {config_path}. Run save_final_model.py first.")
        return
    bundle = load_bundle(str(args.bundle))

    render_demo(bundle)


render_dashboard()
