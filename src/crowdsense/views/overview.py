"""Welcome page: project summary, headline numbers, and the clickable pipeline map."""

import streamlit as st

from constants import RESULTS
from src.crowdsense.nav import render_map

st.session_state.pop("focus", None)

st.title("CrowdSense — Futsal Highlight Detection")
st.caption("Audio-based highlight candidate detection for full futsal match footage.")

st.header("Project Summary")
st.markdown(
    """
    Manually scrubbing through 1.5–2 hours of raw match footage to find goal moments is slow.
    CrowdSense uses the **audio track alone** — crowd noise, cheering, applause — to flag
    candidate windows likely to contain a goal, so a human reviews far less footage when
    building a highlight reel.
    """
)

cols = st.columns(4)
cols[0].metric("Input", "Full match audio (1.5–2h)")
cols[1].metric("Core model", "YAMNet → XGBoost")
cols[2].metric("Goal recall", f"{RESULTS['recall']:.0%}")
cols[3].metric("Footage budget", f"{RESULTS['budget']:.0%}")
st.caption(
    "Recall = share of true goal clips fully covered by a candidate window. Budget = candidate "
    "footage as a fraction of raw match length. Measured on the 8 most recent held-out matches."
)

st.header("Pipeline Map")
render_map()
