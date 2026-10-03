"""Held-out results, limitations, and planned improvements."""

import streamlit as st
from constants import RESULTS

from src.crowdsense.nav import page_header, section

P = "results"
page_header(P, "Results", "How well the model performs and what remains to improve.")

with section(P, "results", "1 · Results"):
    cols = st.columns(3)
    cols[0].metric("Goal recall", f"{RESULTS['recall']:.0%}")
    cols[1].metric("Footage budget", f"{RESULTS['budget']:.0%}")
    cols[2].metric("Test matches", "8 (most recent)")
    st.caption(
        f"Up from an earlier {RESULTS['baseline_recall']:.0%} recall / "
        f"{RESULTS['baseline_budget']:.0%} budget baseline, after replacing full editor clip "
        "bounds with peak-centred narrow labels."
    )

with section(P, "limits", "2 · Limitations"):
    st.info(
        "The dataset is small (~31 matches), and test results come from one fixed split of eight "
        "recent matches. Label quality depends on the peak-finding heuristic; editor clips are "
        "the reference labels, and only 'Goal' clips are positives. The pooled result can hide "
        "match-to-match variation; performance on other clubs, venues or recording conditions "
        "has not been established."
    )

with section(P, "next", "3 · Planned improvements"):
    st.markdown(
        """
        - Temporal features for the two-phase goal signature (crowd spike → quiet period →
          restart whistle), from rolling filters up to a sequence model as a stretch goal.
        - Sweep a frame-merge factor against cached probabilities.
        - Per-match recall breakdown in `eval_model.py`, not just the pooled curve.
        """
    )
