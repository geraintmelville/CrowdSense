"""Performance: metrics, results, limitations, next steps. Single home for headline numbers."""

import streamlit as st
from constants import RECALL_BUDGET_PLOT_PATH, RESULTS

from src.crowdsense.nav import page_header, section

P = "performance"
page_header(P, "Performance", "How the model is measured and how well it does on held-out matches.")

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

with section(P, "results", "2 · Results"):
    cols = st.columns(3)
    cols[0].metric("Goal recall", f"{RESULTS['recall']:.0%}")
    cols[1].metric("Footage budget", f"{RESULTS['budget']:.0%}")
    cols[2].metric("Test matches", "8 (most recent)")
    st.caption(
        f"Up from an earlier {RESULTS['baseline_recall']:.0%} recall / "
        f"{RESULTS['baseline_budget']:.0%} budget baseline, after replacing full editor clip "
        "bounds with peak-centred narrow labels."
    )

with section(P, "limits", "3 · Limitations"):
    st.info(
        "The dataset is small (~31 matches), and test results come from one fixed split of eight "
        "recent matches. Label quality depends on the peak-finding heuristic; editor clips are "
        "the reference labels, and only 'Goal' clips are positives. The pooled result can hide "
        "match-to-match variation; performance on other clubs, venues or recording conditions "
        "has not been established."
    )

with section(P, "next", "4 · Planned improvements"):
    st.markdown(
        """
        - Temporal features for the two-phase goal signature (crowd spike → quiet period →
          restart whistle), from rolling filters up to a sequence model as a stretch goal.
        - Sweep a frame-merge factor against cached probabilities.
        - Per-match recall breakdown in `eval_model.py`, not just the pooled curve.
        """
    )
