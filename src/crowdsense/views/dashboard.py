"""Streamlit entry point for local demo scoring and highlight review."""

import argparse
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from constants.constants import (
    DEFAULT_DEMO_YOUTUBE_URL, DEMO_RAW_AUDIO_DIR, MODEL_PATH,
)
from crowdsense.demo_functions import (
    format_mmss, format_timestamp, goal_intervals, score_match,
    youtube_timestamp_url,
)
from modelling.functions import load_model_artifact
from preprocessing.functions import load_yamnet_model


@st.cache_resource
def load_bundle(bundle_path: str) -> dict:
    return load_model_artifact(bundle_path)


@st.cache_resource
def load_demo_yamnet_model():
    """Keep one YAMNet instance shared across Streamlit reruns and sessions."""
    return load_yamnet_model()


def _render_results(result: dict, youtube_url: str) -> None:
    clips = result["clips"]
    goals = goal_intervals(result["result_key"])
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
    cols[0].metric("Confirmed budget", f"{format_mmss(candidate_seconds)} ({budget_percent:.1%})")
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
            f"{format_mmss(row.start_sec)}-{format_mmss(row.end_sec)}"
            for row in clips.itertuples()
        ],
        "Duration": clips["length_sec"].map(format_mmss),
        "Goal/No Goal": ["Goal" if hits else "No Goal" for hits in hits_by_candidate],
    })
    if youtube_url:
        table["YouTube"] = clips["start_sec"].map(
            lambda start: youtube_timestamp_url(youtube_url, start)
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

    selected_clip_number = st.selectbox(
        "Select a candidate to review",
        options=range(1, len(clips) + 1),
        format_func=lambda clip_number: f"Clip {clip_number}",
    )
    if selected_clip_number is not None and youtube_url:
        selected_clip = clips.iloc[selected_clip_number - 1]
        st.video(youtube_url, start_time=int(selected_clip["start_sec"]))


def render_audio_demo(bundle: dict, youtube_url: str) -> None:
    audio_files = sorted(path for path in DEMO_RAW_AUDIO_DIR.glob("*.wav") if path.is_file())
    if not audio_files:
        st.info(
            f"Add the match WAV to {DEMO_RAW_AUDIO_DIR} to run the demo. "
            "Prepare it from a match video with `python -m demo.prepare_demo`."
        )
        return

    audio_path = st.selectbox("Full match audio", audio_files, format_func=lambda path: path.stem)

    if st.button("Extract features and run analysis", type="primary"):
        try:
            with st.spinner("Extracting YAMNet features from the full-match audio and scoring..."):
                clips, duration, n_windows = score_match(
                    audio_path, bundle, yamnet_model=load_demo_yamnet_model()
                )
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Demo failed: {error}")
            return

        st.session_state["demo_result"] = {
            "result_key": f"demo:{audio_path}",
            "clips": clips,
            "duration": duration,
            "n_windows": n_windows,
        }

    result = st.session_state.get("demo_result")
    result_key = f"demo:{audio_path}"
    if result and result.get("result_key") == result_key:
        _render_results(result, youtube_url)


def render_demo(bundle: dict) -> None:
    st.title("Audio Highlight Candidate Finder")
    st.markdown(
        """
        CrowdSense live demo predicts on a ~25min audio file from a match not seen in training or testing,
        and lets users review the resulting candidate clips via YouTube.

        The demo loads YAMNet and extracts features from the WAV file, loads the saved
        PCA components to compress them. It then loads the pre-trained `XGBoost` model from `model.ubj`
        and `model.json`, scores the features and constructs candidate clips, choosing a threshold
        that corresponds with a budget as close to 33% as possible, see Modelling `eval_model.py` for justification of this.
        """
    )
    youtube_url = os.environ.get("CROWDSENSE_DEMO_YOUTUBE_URL", DEFAULT_DEMO_YOUTUBE_URL).strip()
    render_audio_demo(bundle, youtube_url)


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
st.markdown(
    """
    This project is a proof of concept, since the app provides a demo rather
    than the ability to upload matches and export candidates. An earlier version did
     include cloud upload to an s3 bucket, a Lambda function to extract the audio and
    send it back to the app for feature extraction and inference, however this was extremely
     slow and was again hampered by the lack of budget.
"""
)
