"""Performance: metrics, results, limitations, next steps. Single home for headline numbers."""

import streamlit as st

from constants import RESULTS
from src.crowdsense.nav import page_header, section

P = "performance"
page_header(P, "Performance", "How the model is measured and how well it does on held-out matches.")

with section(P, "metrics", "1 · Recall vs budget"):
    st.markdown(
        """
        - **Recall** — share of true goal clips *fully contained* in a merged candidate window
          (scored against the original editor clip bounds).
        - **Budget** — total merged candidate seconds ÷ total raw match seconds.
        - **Curve AUC** — area under recall vs budget, swept over thresholds. Used to rank
          tuning candidates; `eval_model.py` also reports recall at fixed budget checkpoints.
        """
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
        "~31 matches limit model complexity. Label quality depends on the peak-finding "
        "heuristic. Only 'Goal' clips are used as positives."
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
