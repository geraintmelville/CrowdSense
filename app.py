"""Root entry point for the Futsal Analytics Streamlit application."""

import streamlit as st


st.set_page_config(
    page_title="CrowdSense: Audio Based Futsal Highlight Finder",
    page_icon="⚽",
    layout="wide",
)

pages = {
    "Welcome": [
        st.Page(
            "src/crowdsense/views/overview.py",
            title="Welcome",
            icon=":material/home:",
        ),
    ],
    "Demo": [
        st.Page(
            "src/crowdsense/views/dashboard.py",
            title="Highlight Finder Demo",
            icon=":material/live_tv:",
        ),
    ],
    "Deep Dive Docs": [
        st.Page(
            "src/crowdsense/views/architecture.py",
            title="System Architecture / Demo Workflow",
            icon=":material/account_tree:",
        ),
        st.Page(
            "src/crowdsense/views/model_pipeline.py",
            title="Model & YAMNet Pipeline",
            icon=":material/model_training:",
        ),
        st.Page(
            "src/crowdsense/views/dataset_info.py",
            title="Dataset & Ingestion",
            icon=":material/database:",
        ),
    ],
}

selected_page = st.navigation(pages)
selected_page.run()