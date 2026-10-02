"""Inference & demo flow."""

import streamlit as st

from src.crowdsense.nav import page_header, section

P = "inference"
page_header(P, "Inference & Demo Flow", "How a selected match segment becomes timestamped highlight candidates.")

with section(P, "segment", "1 · Segment creation"):
    st.markdown(
        """
        Pick a full-match MP4 from `demo/raw/video/` and a start/end timecode (default: first
        25 minutes). ffmpeg writes a 720p H.264 segment (CRF 28, AAC) to `demo/generated/`, then
        a mono 22.05 kHz WAV beside it. Generated files are reproducible and Git-ignored.
        """
    )

with section(P, "scoring", "2 · Scoring"):
    st.markdown(
        """
        `modelling/inference.py` runs YAMNet, applies the saved PCA projection (never refit),
        and scores each window with the XGBoost model from `model.ubj` + `model.json`. A Quick
        demo mode skips YAMNet and scores cached feature files from `demo/features/`.
        """
    )

with section(P, "merging", "3 · Candidate merging"):
    st.markdown(
        """
        Windows with probability above the saved threshold are padded (lookback before,
        postroll after) and merged when within `merge_gap` of each other, giving a short list
        of candidate clips instead of a probability per 0.48s.
        """
    )

with section(P, "review", "4 · Dashboard review"):
    st.markdown(
        """
        The dashboard lists merged start/end timestamps. Set `CROWDSENSE_DEMO_YOUTUBE_URL` to the
        uploaded segment's URL to enable timestamp links and in-page playback. Timestamps are
        relative to the selected segment, not the full match.

        This is a low-traffic prototype, not a production service.
        """
    )
