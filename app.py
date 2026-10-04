"""Root entry point for the Futsal Analytics Streamlit application."""

import streamlit as st


st.set_page_config(
    page_title="CrowdSense: Audio Based Futsal Highlight Finder",
    page_icon="⚽",
    layout="wide",
)

V = "src/crowdsense/views"

pages = {
    "Welcome": [
        st.Page(f"{V}/overview.py", title="Welcome", icon=":material/home:"),
    ],
    "Demo": [
        st.Page(f"{V}/dashboard.py", title="Highlight Finder Demo", icon=":material/live_tv:"),
    ],
    "Deep Dive Docs": [
        st.Page(f"{V}/preprocessing.py", title="Preprocessing", icon=":material/database:"),
        st.Page(f"{V}/metrics.py", title="Metrics", icon=":material/monitoring:"),
        st.Page(f"{V}/modelling.py", title="Modelling", icon=":material/model_training:"),
        st.Page(f"{V}/performance.py", title="Performance", icon=":material/bar_chart:"),
    ],
}

selected_page = st.navigation(pages)
selected_page.run()
