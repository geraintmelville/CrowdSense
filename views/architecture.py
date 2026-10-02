"""System architecture and local demo workflow documentation page."""

import streamlit as st


st.title("System Architecture")
st.caption("How a selected match segment becomes timestamped highlight candidates.")

st.header("Architecture Overview")
st.markdown(
    """
    CrowdSense selects a time range from a full match stored in `demo/raw/video/`, creates a compact
    local video segment, extracts its audio, and scores it with YAMNet + XGBoost.
    """
)

st.header("Live Demo Flow")
st.markdown(
    """
    1. **Choose source and range.** Select a full-match MP4 from `demo/raw/video/` and set the start
       and end timecodes. The controls default to the first 25 minutes.
    2. **Create demo segment.** ffmpeg writes a 720p H.264 segment at CRF 28 with AAC audio
       under `demo/generated/`.
    3. **Extract and score.** The app saves a mono 22.05 kHz WAV beside the segment, then
       runs YAMNet, the saved PCA projection, and the trained XGBoost classifier.
    4. **Review candidates.** The dashboard displays merged start/end timestamps. Set
       `CROWDSENSE_DEMO_YOUTUBE_URL` to the uploaded segment's YouTube URL to enable direct
       timestamp links and in-page playback.

    Generated segments and WAV files are reproducible local artifacts and are ignored by
    Git. YouTube timestamps are relative to the selected segment, not the full match.
    """
)

st.header("Component Responsibilities")
st.markdown(
    """
    - **Streamlit app (`dashboard.py`)** — chooses a full match and time range, creates the
      demo segment, scores it, and renders timestamped candidates.
    - **`preprocessing/extract_audio.py`** — creates the 720p demo segment and extracts its
      audio using ffmpeg.
    - **Inference (`modelling/inference.py` + `data/modelling/final_model/model.ubj` + `model.json`)** —
      YAMNet feature extraction and XGBoost scoring for the selected segment.
    """
)

st.header("Model Artifacts")
st.markdown(
    """
    `save_final_model.py` trains once on the complete feature set and persists the XGBoost
    booster in native UBJ format plus inference configuration in JSON. Together they contain
    everything the dashboard needs to score new audio without retraining: the decision
    threshold, feature column list, YAMNet score indices, window/stride, PCA components + mean,
    and candidate-window settings (lookback/postroll/merge-gap).
    """
)

st.header("Operational Notes")
st.markdown(
    """
    - This is a **low-traffic prototype**, not a production service — no autoscaling or
      multi-user concerns have been designed for yet.
    - Set `CROWDSENSE_DEMO_YOUTUBE_URL` in the environment to the uploaded segment URL.
    - Place source full-match videos under `demo/raw/video/`; generated clips and audio stay in the
      ignored `demo/generated/` directory.
    """
)"""System architecture and local demo workflow documentation page."""

import streamlit as st


st.title("System Architecture")
st.caption("How a selected match segment becomes timestamped highlight candidates.")

st.header("Architecture Overview")
st.markdown(
    """
    CrowdSense selects a time range from a full match stored in `demo/raw/video/`, creates a compact
    local video segment, extracts its audio, and scores it with YAMNet + XGBoost.
    """
)

st.header("Live Demo Flow")
st.markdown(
    """
    1. **Choose source and range.** Select a full-match MP4 from `demo/raw/video/` and set the start
       and end timecodes. The controls default to the first 25 minutes.
    2. **Create demo segment.** ffmpeg writes a 720p H.264 segment at CRF 28 with AAC audio
       under `demo/generated/`.
    3. **Extract and score.** The app saves a mono 22.05 kHz WAV beside the segment, then
       runs YAMNet, the saved PCA projection, and the trained XGBoost classifier.
    4. **Review candidates.** The dashboard displays merged start/end timestamps. Set
       `CROWDSENSE_DEMO_YOUTUBE_URL` to the uploaded segment's YouTube URL to enable direct
       timestamp links and in-page playback.

    Generated segments and WAV files are reproducible local artifacts and are ignored by
    Git. YouTube timestamps are relative to the selected segment, not the full match.
    """
)

st.header("Component Responsibilities")
st.markdown(
    """
    - **Streamlit app (`dashboard.py`)** — chooses a full match and time range, creates the
      demo segment, scores it, and renders timestamped candidates.
    - **`preprocessing/extract_audio.py`** — creates the 720p demo segment and extracts its
      audio using ffmpeg.
    - **Inference (`modelling/inference.py` + `data/modelling/final_model/model.ubj` + `model.json`)** —
      YAMNet feature extraction and XGBoost scoring for the selected segment.
    """
)

st.header("Model Artifacts")
st.markdown(
    """
    `save_final_model.py` trains once on the complete feature set and persists the XGBoost
    booster in native UBJ format plus inference configuration in JSON. Together they contain
    everything the dashboard needs to score new audio without retraining: the decision
    threshold, feature column list, YAMNet score indices, window/stride, PCA components + mean,
    and candidate-window settings (lookback/postroll/merge-gap).
    """
)

st.header("Operational Notes")
st.markdown(
    """
    - This is a **low-traffic prototype**, not a production service — no autoscaling or
      multi-user concerns have been designed for yet.
    - Set `CROWDSENSE_DEMO_YOUTUBE_URL` in the environment to the uploaded segment URL.
    - Place source full-match videos under `demo/raw/video/`; generated clips and audio stay in the
      ignored `demo/generated/` directory.
    """
)"""System architecture and local demo workflow documentation page."""

import streamlit as st


st.title("System Architecture")
st.caption("How a selected match segment becomes timestamped highlight candidates.")

st.header("Architecture Overview")
st.markdown(
    """
    CrowdSense selects a time range from a full match stored in `demo/raw/video/`, creates a compact
    local video segment, extracts its audio, and scores it with YAMNet + XGBoost.
    """
)

st.header("Live Demo Flow")
st.markdown(
    """
    1. **Choose source and range.** Select a full-match MP4 from `demo/raw/video/` and set the start
       and end timecodes. The controls default to the first 25 minutes.
    2. **Create demo segment.** ffmpeg writes a 720p H.264 segment at CRF 28 with AAC audio
       under `demo/generated/`.
    3. **Extract and score.** The app saves a mono 22.05 kHz WAV beside the segment, then
       runs YAMNet, the saved PCA projection, and the trained XGBoost classifier.
    4. **Review candidates.** The dashboard displays merged start/end timestamps. Set
       `CROWDSENSE_DEMO_YOUTUBE_URL` to the uploaded segment's YouTube URL to enable direct
       timestamp links and in-page playback.

    Generated segments and WAV files are reproducible local artifacts and are ignored by
    Git. YouTube timestamps are relative to the selected segment, not the full match.
    """
)

st.header("Component Responsibilities")
st.markdown(
    """
    - **Streamlit app (`dashboard.py`)** — chooses a full match and time range, creates the
      demo segment, scores it, and renders timestamped candidates.
    - **`preprocessing/extract_audio.py`** — creates the 720p demo segment and extracts its
      audio using ffmpeg.
    - **Inference (`modelling/inference.py` + `data/modelling/final_model/model.ubj` + `model.json`)** —
      YAMNet feature extraction and XGBoost scoring for the selected segment.
    """
)

st.header("Model Artifacts")
st.markdown(
    """
    `save_final_model.py` trains once on the complete feature set and persists the XGBoost
    booster in native UBJ format plus inference configuration in JSON. Together they contain
    everything the dashboard needs to score new audio without retraining: the decision
    threshold, feature column list, YAMNet score indices, window/stride, PCA components + mean,
    and candidate-window settings (lookback/postroll/merge-gap).
    """
)

st.header("Operational Notes")
st.markdown(
    """
    - This is a **low-traffic prototype**, not a production service — no autoscaling or
      multi-user concerns have been designed for yet.
    - Set `CROWDSENSE_DEMO_YOUTUBE_URL` in the environment to the uploaded segment URL.
    - Place source full-match videos under `demo/raw/video/`; generated clips and audio stay in the
      ignored `demo/generated/` directory.
    """
