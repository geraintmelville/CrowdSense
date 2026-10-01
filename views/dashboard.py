"""Streamlit entry point for cloud video uploads and highlight scoring."""

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

from constants import MODEL_BUNDLE_PATH
from modelling.functions import merge_intervals
from backend.cloud_config import get_upload_config


@st.cache_resource
def load_bundle(bundle_path: str) -> dict:
    return joblib.load(bundle_path)


@st.cache_resource
def load_yamnet_model():
    from preprocessing.extract_features import load_yamnet_model as load_model

    return load_model()


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def score_match(audio_path: Path, bundle: dict, yamnet_model=None) -> tuple[pd.DataFrame, float, int]:
    from preprocessing.extract_features import extract_yamnet_features_streaming

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

    frame_indices = np.clip(
        np.rint(starts / stride_sec).astype(np.intp), 0, len(scores) - 1
    )
    score_features = scores[frame_indices][:, list(bundle["yamnet_score_indices"])]
    feature_matrix = score_features.astype(np.float32)
    if uses_pca:
        if embeddings is None:
            raise ValueError("Embeddings are required when the model uses PCA features")
        pca_components = np.asarray(bundle["pca_components"])
        pca_mean = np.asarray(bundle["pca_mean"])
        pca_features = (embeddings[frame_indices] - pca_mean) @ pca_components.T
        feature_matrix = np.concatenate((feature_matrix, pca_features), axis=1).astype(np.float32)
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
    upload_url = json.dumps(presigned["url"])

    components.html(
        f"""
        <div>
          <input type="file" id="crowdsense-file" accept="video/mp4" />
          <button id="crowdsense-upload-btn">Upload to cloud</button>
          <div id="crowdsense-status"></div>
        </div>

        <script>
          const fields = {fields_json};
          const uploadUrl = {upload_url};

          document.getElementById("crowdsense-upload-btn").onclick = async () => {{
            const fileInput = document.getElementById("crowdsense-file");
            const status = document.getElementById("crowdsense-status");

            if (!fileInput.files.length) {{
              status.innerText = "Choose a file first.";
              return;
            }}

            const file = fileInput.files[0];
            const formData = new FormData();

            Object.entries(fields).forEach(([k, v]) => {{
              formData.append(k, v);
            }});

            formData.append("file", file);

            status.innerText =
              `Uploading ${{file.name}} (${{(file.size / 1024 / 1024).toFixed(1)}} MB)...`;

            try {{
              const response = await fetch(uploadUrl, {{
                method: "POST",
                body: formData
              }});

              const responseText = await response.text();

              console.log("S3 status:", response.status);
              console.log("S3 response:", responseText);

              if (response.ok) {{
                status.innerText =
                  "Uploaded. Click 'Check for extracted audio' below.";
              }} else {{
                status.innerText =
                  `S3 upload failed: HTTP ${{response.status}}\\n${{responseText.substring(0, 500)}}`;
              }}

            }} catch (error) {{
              console.error("S3 upload error:", error);

              status.innerText =
                `Browser upload error: ${{error.name}}: ${{error.message}}`;
            }}
          }};
        </script>
        """,
        height=180,
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
        for state_key in ("clips", "video_url", "duration", "n_windows"):
            st.session_state.pop(state_key, None)
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
        video_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": presigned["key"]},
            ExpiresIn=3600,
        )
        st.session_state["clips"] = clips
        st.session_state["video_url"] = video_url
        st.session_state["duration"] = duration
        st.session_state["n_windows"] = n_windows

    if "clips" not in st.session_state:
        return

    clips = st.session_state["clips"]
    video_url = st.session_state["video_url"]
    duration = st.session_state["duration"]
    n_windows = st.session_state["n_windows"]

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

    selected_clip_index = st.selectbox(
        "Select a clip to review:",
        options=clips.index,
        format_func=lambda i: f"Clip {i+1}: {clips.loc[i, 'start']} to {clips.loc[i, 'end']} ({clips.loc[i, 'length_sec']:.1f}s)",
    )

    if selected_clip_index is not None:
        clip = clips.loc[selected_clip_index]
        start_seconds = float(clip["start_sec"])

        st.write(f"Playing highlight from **{clip['start']}**")

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

    render_cloud_upload_tab(bundle)


render_dashboard()