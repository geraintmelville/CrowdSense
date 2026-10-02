"""Shared navigation: pipeline map on the Welcome page, section focus on doc pages.

Clicking a node stores (page, section_key) in session_state and switches page.
The target page opens only that section's expander; everything else collapses.
"""

import streamlit as st

PAGES = {
    "welcome": "src/crowdsense/views/overview.py",
    "preprocessing": "src/crowdsense/views/preprocessing.py",
    "modelling": "src/crowdsense/views/modelling.py",
    "performance": "src/crowdsense/views/performance.py",
    "inference": "src/crowdsense/views/inference.py",
}

# (section_key, label, train_only)
LANES = [
    ("preprocessing", "1 · Preprocessing", [
        ("ingestion", "Ingestion & clip DB", False),
        ("audio", "Audio extraction", False),
        ("features", "YAMNet features", False),
        ("pca", "PCA (16-dim)", True),
        ("labels", "Label refinement", False),
    ]),
    ("modelling", "2 · Modelling", [
        ("targets", "Targets & grouped CV", True),
        ("tuning", "Hyperparameter tuning", True),
        ("window_grid", "Candidate-window grid", True),
        ("train_score", "Train & test scoring", False),
        ("final", "Final model & threshold", True),
    ]),
    ("performance", "3 · Performance", [
        ("metrics", "Recall vs budget", False),
        ("results", "Results", False),
        ("limits", "Limitations", False),
        ("next", "Planned improvements", False),
    ]),
    ("inference", "4 · Inference & demo", [
        ("segment", "Segment creation", False),
        ("scoring", "Scoring", False),
        ("merging", "Candidate merging", False),
        ("review", "Dashboard review", False),
    ]),
]


def render_map() -> None:
    st.caption("Click any stage to jump to its explanation. 🔒 = uses training matches only "
               "(the 8 test matches never touch it).")
    columns = st.columns(len(LANES))
    for column, (page, title, nodes) in zip(columns, LANES):
        with column:
            st.markdown(f"**{title}**")
            for index, (key, label, train_only) in enumerate(nodes):
                text = f"🔒 {label}" if train_only else label
                if st.button(text, key=f"map_{page}_{key}", width="stretch"):
                    st.session_state["focus"] = (page, key)
                    st.switch_page(PAGES[page])
                if index < len(nodes) - 1:
                    st.markdown("<div style='text-align:center;line-height:1'>↓</div>",
                                unsafe_allow_html=True)


def page_header(page: str, title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)
    left, right = st.columns([1, 1])
    if left.button("← Back to pipeline map", key=f"back_{page}"):
        st.session_state.pop("focus", None)
        st.switch_page(PAGES["welcome"])
    focus = st.session_state.get("focus")
    if focus and focus[0] == page:
        if right.button("Show all sections", key=f"all_{page}"):
            st.session_state.pop("focus", None)
            st.rerun()


def section(page: str, key: str, title: str):
    """Expander that is open if nothing is focused on this page, or if it is the focused one."""
    focus = st.session_state.get("focus")
    active = focus[1] if focus and focus[0] == page else None
    return st.expander(title, expanded=active is None or active == key)
