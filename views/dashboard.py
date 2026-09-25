"""Streamlit entry point for local audio scoring and cloud video uploads."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tensorflow_hub as hub

from constants import MODEL_BUNDLE_PATH
from modelling.functions import merge_intervals
from preprocessing.extract_features import extract_yamnet_features_streaming
from backend.cloud_config import get_upload_config


def aggregate_window_features(
    scores,
    embeddings,
    start_sec,
    window_sec,
    stride_sec,
    score_indices,
    embedding_transform=None,
):
    frame_index = int(np.clip(np.rint(start_sec / stride_sec), 0, len(scores) - 1))
    selected_scores = scores[frame_index][list(score_indices)]
    if embedding_transform is None:
        return selected_scores.astype(np.float32)
    if embeddings is None:
        raise ValueError("Embeddings are required when embedding_transform is set")
    pca_components, pca_mean = embedding_transform
    selected_embeddings = embeddings[frame_index]
    selected_pca = (selected_embeddings - pca_mean) @ pca_components.T
    return np.concatenate([selected_scores, selected_pca]).astype(np.float32)


@st.cache_resource
def load_bundle(bundle_path: str) -> dict:
    return joblib.load(bundle_path)


@st.cache_resource
def load_yamnet_model():
    return hub.load("https://tfhub.dev/google/yamnet/1")


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def score_match(audio_path: Path, bundle: dict, yamnet_model=None) -> tuple[pd.DataFrame, float, int]:
    feature_columns = bundle.get("feature_columns", [])
    uses_pca = any(column.startswith("yamnet_embedding_pca_") for column in feature_columns)
    if uses_pca and not {"pca_components", "pca_mean"}.issubset(bundle):
        raise ValueError("This PCA model bundle is missing its PCA projection")
    scores, embeddings, duration = extract_yamnet_features_streaming(
        audio_path,
        include_embeddings=uses_pca,
        model=yamnet_model if yamnet_model is not None else load_yamnet_model(),
    )
    window_sec = bundle["window_sec"]
    stride_sec = bundle["stride_sec"]
    starts = np.arange(0, max(0, duration - window_sec) + 1e-6, stride_sec)

    feature_rows = [
        aggregate_window_features(
            scores,
            embeddings,
            float(start),
            window_sec,
            stride_sec,
            bundle["yamnet_score_indices"],
            (np.asarray(bundle["pca_components"]), np.asarray(bundle["pca_mean"]))
            if uses_pca else None,
        )
        for start in starts
    ]
    feature_matrix = np.vstack(feature_rows)
    probabilities = bundle["model"].predict_proba(feature_matrix)[:, 1]

    selected_starts = starts[probabilities >= bundle["threshold"]]
    intervals = merge_intervals(
        [(max(0, s - bundle["lookback"]), min(duration, s + window_sec + bundle["postroll"]))
         for s in selected_starts],
        bundle["merge_gap"],
    )
    clips = pd.DataFrame(intervals, columns=["start_sec", "end_sec"])
    clips["length_sec"] = clips["end_sec"] - clips["start_sec"]
    clips["start"] = clips["start_sec"].apply(format_timestamp)
    clips["end"] = clips["end_sec"].apply(format_timestamp)
    return clips, duration, len(starts)


def _render_upload_widget(presigned: dict) -> None:
    fields_json = json.dumps(presigned["fields"])
    components.html(
        f"""
        <div>
          <input type="file" id="crowdsense-file" accept="video/mp4" />
          <button id="crowdsense-upload-btn">Upload to cloud</button>
          <div id="crowdsense-status"></div>
        </div>
        <script>
          const fields = {fields_json};
          const uploadUrl = "{presigned['url']}";
          document.getElementById("crowdsense-upload-btn").onclick = async () => {{
            const fileInput = document.getElementById("crowdsense-file");
            const status = document.getElementById("crowdsense-status");
            if (!fileInput.files.length) {{
              status.innerText = "Choose a file first.";
              return;
            }}
            const formData = new FormData();
            Object.entries(fields).forEach(([k, v]) => formData.append(k, v));
            formData.append("file", fileInput.files[0]);
            status.innerText = "Uploading... this can take a while for a full match.";
            try {{
              const response = await fetch(uploadUrl, {{ method: "POST", body: formData }});
              status.innerText = response.ok
                ? "Uploaded. Click 'Check for extracted audio' below -- extraction runs automatically."
                : `Upload failed (status ${{response.status}}).`;
            }} catch (error) {{
              status.innerText = "Upload failed. Check the bucket CORS configuration.";
            }}
          }};
        </script>
        """,
        height=150,
    )


def _wait_for_audio(s3, bucket: str, audio_key: str, timeout_sec: int = 300, interval_sec: int = 5) -> bool:
    elapsed = 0
    with st.spinner("Waiting for audio extraction to finish..."):
        while elapsed < timeout_sec:
            try:
                s3.head_object(Bucket=bucket, Key=audio_key)
                return True
            except s3.exceptions.ClientError as error:
                if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
                    raise
                time.sleep(interval_sec)
                elapsed += interval_sec
    return False


def render_cloud_upload_tab(bundle: dict) -> None:
    config = get_upload_config()
    if config is None:
        st.info("Cloud upload is not configured. Set CROWDSENSE_UPLOAD_BUCKET to enable it.")
        return

    import boto3
    from backend.upload_helper import presigned_upload

    bucket, region, _ = config
    s3 = boto3.client("s3", region_name=region)
    st.subheader("Upload full match footage")
    st.caption("Uploads go straight to cloud storage. Audio extraction runs automatically, and raw footage is securely retained for playback until lifecycle rules clean it up.")

    uploaded_name = st.text_input("Filename (for reference -- pick the actual file below)")
    if not uploaded_name:
        st.info("Enter a filename to get an upload link.")
        return

    if st.session_state.get("presigned_filename") != uploaded_name:
        try:
            st.session_state["presigned"] = presigned_upload(uploaded_name)
        except RuntimeError as error:
            st.error(str(error))
            return
        st.session_state["presigned_filename"] = uploaded_name

    presigned = st.session_state["presigned"]
    _render_upload_widget(presigned)

    if st.button("Check for extracted audio"):
        stem = Path(presigned["key"]).stem
        audio_key = f"audio/{stem}.wav"
        if not _wait_for_audio(s3, bucket, audio_key):
            st.warning("Audio isn't ready yet (or upload didn't complete) -- try again shortly.")
            return
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            audio_path = Path(tmp_dir_name) / "audio.wav"
            s3.download_file(bucket, audio_key, str(audio_path))
            with st.spinner("Running YAMNet + classifier..."):
                clips, duration, n_windows = score_match(audio_path, bundle)
        
        # Clean up the temporary audio file object, leaving the raw video intact for streaming
        s3.delete_object(Bucket=bucket, Key=audio_key)

        st.success(f"Scored {n_windows} windows across {format_timestamp(duration)} of footage.")
        if clips.empty:
            st.warning("No candidate windows cleared the threshold.")
            return
            
        st.dataframe(
            clips[["start", "end", "length_sec"]].rename(
                columns={"start": "Start", "end": "End", "length_sec": "Length (s)"}
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("📺 Watch Highlight Clips")

        # Generate a secure presigned URL for the raw video (valid for 1 hour)
        raw_video_key = presigned["key"]
        video_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": raw_video_key},
            ExpiresIn=3600,
        )

        # Let the user pick a clip to play
        selected_clip_index = st.selectbox(
            "Select a clip to review:",
            options=clips.index,
            format_func=lambda i: f"Clip {i+1}: {clips.loc[i, 'start']} to {clips.loc[i, 'end']} ({clips.loc[i, 'length_sec']:.1f}s)",
        )

        if selected_clip_index is not None:
            clip = clips.loc[selected_clip_index]
            start_seconds = float(clip["start_sec"])
            
            st.write(f"Playing highlight from **{clip['start']}**")
            
            # Streamlit video component seeking straight to the timestamp
            st.video(video_url, start_time=start_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=MODEL_BUNDLE_PATH)
    known_args, _ = parser.parse_known_args()
    return known_args


def render_dashboard() -> None:
    args = parse_args()
    st.title("Audio Highlight Candidate Finder")
    if not args.bundle.exists():
        st.error(f"Model bundle not found: {args.bundle}. Run save_final_model.py first.")
        return
    bundle = load_bundle(str(args.bundle))

    local_tab, cloud_tab = st.tabs(["Local audio", "Cloud video"])
    with local_tab:
        uploaded_audio = st.file_uploader("Upload pre-extracted match audio (.wav)", type=["wav"])
        if uploaded_audio is not None:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                audio_path = Path(tmp_dir_name) / "audio.wav"
                audio_path.write_bytes(uploaded_audio.getbuffer())
                with st.spinner("Running YAMNet + classifier..."):
                    clips, duration, n_windows = score_match(audio_path, bundle)
            st.success(f"Scored {n_windows} windows across {format_timestamp(duration)} of footage.")
            st.metric("Candidate clips", len(clips))
            if clips.empty:
                st.warning("No candidate windows cleared the threshold.")
            else:
                st.dataframe(
                    clips[["start", "end", "length_sec"]].rename(
                        columns={"start": "Start", "end": "End", "length_sec": "Length (s)"}
                    ),
                    use_container_width=True,
                )
                st.download_button(
                    "Download candidates as CSV",
                    clips[["start_sec", "end_sec", "length_sec"]].to_csv(index=False),
                    file_name=f"{Path(uploaded_audio.name).stem}_candidates.csv",
                    mime="text/csv",
                )
    with cloud_tab:
        render_cloud_upload_tab(bundle)


render_dashboard()