"""System architecture and cloud infrastructure documentation page."""

import streamlit as st


st.title("System Architecture")
st.caption("How upload, storage, extraction, and inference fit together.")

st.header("Architecture Overview")
st.markdown(
    """
    CrowdSense supports two ways to get footage in: uploading already-extracted audio
    directly to this app, or uploading full raw match video to the cloud for automatic
    audio extraction. Both paths converge on the same YAMNet + XGBoost scoring step.
    """
)

st.header("Cloud Upload Flow")
st.markdown(
    """
    1. **Browser → S3, directly.** The dashboard requests a presigned POST URL; the raw
       video is uploaded straight from the browser to S3 (`raw/<key>.mp4`) and never passes
       through the Streamlit server's memory.
    2. **S3 → Lambda, automatically.** An `ObjectCreated` event on `raw/` triggers a Lambda
       function that runs ffmpeg (via a Lambda layer, since there's no apt/yum in the Lambda
       runtime) to extract the audio track.
    3. **Lambda → S3.** The extracted audio is written to `audio/<key>.wav`, and the raw
       video is deleted from the extraction step's working area (the original raw video
       itself is retained in S3 for in-app playback until lifecycle rules clean it up).
    4. **App polls, then downloads.** The dashboard polls for the extracted audio object,
       downloads just that (small) file, and runs it through the same scoring path as a
       directly-uploaded WAV.
    5. **Playback.** A presigned GET URL lets the dashboard stream the original raw video
       and seek straight to a candidate clip's timestamp for review.

    This keeps the only expensive/large transfer (the raw video) off the Streamlit server
    entirely, and keeps Lambda scoped to just the ffmpeg step — no TensorFlow/YAMNet there.
    """
)

st.header("Component Responsibilities")
st.markdown(
    """
    - **Streamlit app (`dashboard.py`)** — local-audio and cloud-video tabs, triggers
      scoring, renders candidate clips and video playback.
    - **`backend/upload_helper.py`** — builds presigned S3 POST requests for direct
      browser-to-cloud upload.
    - **`backend/lambda/extract_audio_lambda.py`** — S3-triggered ffmpeg audio extraction;
      needs ephemeral storage sized above the largest raw video, since both the mp4 and wav
      land in `/tmp` temporarily.
    - **`backend/cloud_config.py`** — bucket name, region, and prefix configuration, read
      from environment variables.
    - **Inference (`preprocessing/extract_features.py` + `data/modelling/final_model/model_bundle.joblib`)** —
      YAMNet feature extraction and XGBoost scoring, shared by both upload paths.
    """
)

st.header("Model Bundle")
st.markdown(
    """
    `save_final_model.py` trains once on the complete feature set and persists everything
    the dashboard needs to score
    new footage without retraining: the trained model, decision threshold, feature column
    list, YAMNet score indices, window/stride, PCA components + mean, and the candidate-window
    settings (lookback/postroll/merge-gap).
    """
)

st.header("Operational Notes")
st.markdown(
    """
    - This is a **low-traffic prototype**, not a production service — no autoscaling or
      multi-user concerns have been designed for yet.
    - Credentials and bucket config are supplied via environment variables (`.env`), never
      hardcoded.
    - Raw video in S3 is time-limited by lifecycle rules rather than deleted immediately,
      to allow in-app playback of recently uploaded matches.
    """
)