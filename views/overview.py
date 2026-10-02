"""Holistic project summary page."""

import streamlit as st


st.title("CrowdSense — Futsal Highlight Detection")
st.caption("Audio-based highlight candidate detection for full futsal match footage.")

st.header("Project Summary")
st.markdown(
    """
    Manually scrubbing through 1.5–2 hours of raw match footage to find goal moments is slow.
    CrowdSense uses the **audio track alone** — crowd noise, cheering, applause — to flag
    candidate windows likely to contain a goal, cutting down how much footage a human needs
    to review to build a highlight reel.

    **Pipeline in one line:** raw match video → extracted audio → YAMNet audio embeddings →
    XGBoost classifier → merged candidate clips → reviewed in this dashboard.
    """
)

st.header("At A Glance")
summary = st.columns(3)
summary[0].metric("Input", "Full match audio (1.5–2h)")
summary[1].metric("Core model", "YAMNet → XGBoost")
summary[2].metric("Output", "Candidate goal clips")

st.header("Current Performance")
perf = st.columns(2)
perf[0].metric("Goal recall", "79%")
perf[1].metric("Footage budget", "28%")
st.caption(
    "Recall = share of true goal clips fully covered by a candidate window. "
    "Budget = candidate footage as a fraction of total raw match length — "
    "i.e. reviewers watch ~28% of the match instead of 100% and still catch ~79% of goals."
)

st.header("How It Works, Briefly")
st.markdown(
    """
    1. **Extract audio** from raw match video (ffmpeg).
    2. **Window the audio** at YAMNet's native cadence (0.96s window, 0.48s stride) and run
       each window through YAMNet to get crowd/cheer/whistle-type scores plus a general audio embedding.
    3. **Reduce the embedding** to 16 dimensions with PCA (fit on training matches only).
    4. **Classify each window** with a tuned XGBoost model — probability that a goal is happening nearby.
    5. **Merge nearby high-probability windows** into candidate clips (with lookback/postroll padding),
       so a reviewer gets a handful of short clips per match instead of a probability per 0.48s.

    See the **Model & YAMNet Pipeline** page for the full stage-by-stage detail, and
    **System Architecture** for how the local demo segment is created and scored.
    """
)

st.header("Project Areas")
st.markdown(
    """
    - **Dataset & Ingestion** — how raw footage, highlight clips, and labels become training data.
    - **System Architecture** — local video segment creation, audio extraction, and inference.
    - **Model & YAMNet Pipeline** — feature extraction, PCA, XGBoost, threshold selection.
    - **Dashboard** (this app's main entry point) — choose a full-match time range and review candidates.
    """
)

st.header("What's Next")
st.markdown(
    """
    - Temporal pattern features (crowd spike → quiet gap → restart whistle) to push recall further.
    - Per-match recall breakdown in evaluation, not just the pooled curve.
    """
)