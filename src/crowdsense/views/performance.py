"""Held-out model performance."""

from contextlib import contextmanager
from pathlib import Path

import streamlit as st
from constants import RECALL_BUDGET_PLOT_PATH, RESULTS


@contextmanager
def section(title):
    with st.container(border=True):
        st.subheader(title)
        yield


st.title("Performance")
st.caption("Evaluated on the 8 held-out test matches.")

if RECALL_BUDGET_PLOT_PATH.is_file():
    st.image(
        str(RECALL_BUDGET_PLOT_PATH),
        width="stretch",
    )
else:
    try:
        relative_plot_path = RECALL_BUDGET_PLOT_PATH.relative_to(
            RECALL_BUDGET_PLOT_PATH.parents[2]
        )
    except ValueError:
        relative_plot_path = Path(RECALL_BUDGET_PLOT_PATH.name)
    st.info(
        "Recall-budget curve not found. Run `python -m modelling.eval_model` "
        f"to generate `{relative_plot_path}`."
    )

with section("Deployment threshold"):
    cols = st.columns(3)
    cols[0].metric("Goal recall", f"{RESULTS['recall']:.0%}")
    cols[1].metric("Footage budget", f"{RESULTS['budget']:.0%}")
    cols[2].metric("Partial AUC (25-40% )", "0.92")
    st.caption(
        "Results are measured on the test set and use the deployment" \
        " threshold that corresponds with budget closest to 33%"
    )

with section("Limitations & next steps"):
    st.markdown(
        """
        The dataset is small (32 matches), and test results come from one fixed split of
        eight recent matches. Label quality depends on the peak-finding heuristic; editor clips
        are the reference labels, and only Goal clips are positives. Pooled results can hide
        match-to-match variation, and performance on other clubs, venues, or recording
        conditions has not been established. A possible improvement could be temporal features for the two-phase goal signature
        (crowd spike → quiet period → restart whistle).
        """
    )
