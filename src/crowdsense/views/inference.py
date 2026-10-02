"""Inference & demo flow."""

import streamlit as st

from src.crowdsense.nav import page_header, section

P = "inference"
page_header(P, "Inference & Demo Flow", "How selected match audio becomes timestamped highlight candidates.")

with section(P, "segment", "1 · Choose match audio"):
    st.markdown(
        """
        Both demo modes use the same prepared match. Quick demo selects its cached feature file
        from `demo/features/`; Full demo selects the corresponding WAV from `demo/raw/audio/`
        and extracts the features during analysis. Prepare the WAV from a source MP4 in
        `demo/raw/video/` with `python -m demo.prepare_demo`.
        Quick mode estimates duration from the final cached frame start plus the model window;
        this can differ slightly from the source audio duration.
        """
    )

with section(P, "scoring", "2 · Scoring"):
    st.markdown(
        """
        `modelling/inference.py` runs YAMNet, applies the saved PCA projection (never refit),
        and scores each window with the XGBoost model from `model.ubj` + `model.json`. Quick demo
        skips YAMNet and scores cached features; Full demo computes the same score and PCA
        features from the selected match audio.
        """
    )

with section(P, "merging", "3 · Candidate merging"):
    st.markdown(
        """
        Windows whose model score meets or exceeds the saved threshold are padded (lookback before,
        postroll after) and merged when within `merge_gap` of each other, giving a short list
        of candidate clips instead of a probability per 0.48s.
        """
    )

with section(P, "review", "4 · Dashboard review"):
    st.markdown(
        """
        The dashboard lists merged start/end timestamps. Set `CROWDSENSE_DEMO_YOUTUBE_URL` to the
        selected match's URL to enable timestamp links and in-page playback. Timestamps are
        relative to the start of the selected match audio, so the YouTube video must start at
        the same point as the audio for timestamps to align.

        This is a low-traffic prototype, not a production service.
        """
    )
