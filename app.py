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
            "views/overview.py",
            title="Overview / Holistic Project Summary",
            icon=":material/home:",
        ),
        st.Page(
            "views/dashboard.py",
            title="Live Highlight Finder Dashboard",
            icon=":material/live_tv:",
        ),
    ],
    "Deep Dive Docs": [
        st.Page(
            "views/architecture.py",
            title="System Architecture / Cloud Infrastructure",
            icon=":material/account_tree:",
        ),
        st.Page(
            "views/model_pipeline.py",
            title="Model & YAMNet Pipeline",
            icon=":material/model_training:",
        ),
        st.Page(
            "views/dataset_info.py",
            title="Dataset & Ingestion",
            icon=":material/database:",
        ),
    ],
}

selected_page = st.navigation(pages)
selected_page.run()