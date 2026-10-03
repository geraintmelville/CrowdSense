"""Evaluation metrics used to measure candidate coverage."""

import streamlit as st
from constants import RECALL_BUDGET_PLOT_PATH

from src.crowdsense.nav import page_header, section

P = "metrics"
page_header(P, "Metrics", "How candidate coverage is measured on held-out matches.")

with section(P, "metrics", "1 · Recall vs budget"):
    st.markdown(
        """
        - **Recall** — share of editor-labelled "Goal" clips *fully contained* in a merged
          candidate window (scored against original editor clip bounds). This measures clip
          coverage, not whether a detected event is a goal or classification accuracy.
        - **Budget** — total merged candidate seconds ÷ total raw match seconds.
        - **Partial recall-budget AUC (25-40%)** — mean recall over the deployment budget
          band, swept over thresholds. This is the metric used to rank tuning candidates and
          reported by `eval_model.py` on held-out matches.
        """
    )
    st.caption(
        "The evaluation pools goal clips and footage seconds across eight held-out matches. "
        "The displayed values are rounded summary metrics; performance can vary by match."
    )
    if RECALL_BUDGET_PLOT_PATH.is_file():
        st.image(
            str(RECALL_BUDGET_PLOT_PATH),
            caption="Held-out test-set recall as the candidate footage budget increases.",
            use_container_width=True,
        )
    else:
        st.info(
            "Recall-budget curve not found. Run `python -m modelling.eval_model` "
            f"to generate `{RECALL_BUDGET_PLOT_PATH.relative_to(RECALL_BUDGET_PLOT_PATH.parents[2])}`."
        )
