"""Dataset and ingestion documentation page."""

import streamlit as st


st.title("Dataset & Ingestion")
st.caption("Raw footage, highlight clips, metadata, and how labels are derived.")

st.header("Raw Data")
raw = st.columns(2)
raw[0].metric("Raw matches", "~31 (1.5–2h each)")
raw[1].metric("Held-out test set", "8 most recent matches")
st.markdown(
    """
    Each match is a full raw video file plus a separate ZIP of editor-cut highlight clips
    (30–70 clips per match, 5–60 seconds each, e.g. "Goal", "Save", "Card"). The test split
    is deliberately the **most recent** matches, since in deployment the model always sees
    new footage, not old.
    """
)

st.header("Metadata Schema")
st.markdown(
    """
    A SQLite database (`data/metadata/clips_data.db`) links raw matches to their highlight clips:

    - **`matches`** — `match_id`, `raw_filename`, `zip_filename`, `match_date`, `length_sec`
      (raw match footage matched to its highlight ZIP by normalized opposition name + date).
    - **`clips`** — `match_id`, `clip_number`, `timestamp_formatted`, `description`, `filename`,
      `length_sec` (one row per highlight clip, parsed from the ZIP member filenames).

    Unmatched raw files or ambiguous/unparsable ZIPs are flagged during ingestion rather than
    silently dropped.
    """
)

st.header("From Editor Clips to Training Labels")
st.markdown(
    """
    A highlight clip's timestamp is the **editor's cut** — it usually includes lead-in, the
    crowd-roar spike, a quiet gap, and sometimes the restart whistle. Training on the whole
    cut as "positive" dilutes the acoustic signal the model needs to learn.

    Instead, each clip is narrowed to a short window centered on its **acoustic peak**:

    1. Take the clip's `[start, start + length]` span, expanded slightly on both sides.
    2. Within that span, sum the YAMNet Cheering + Crowd + Applause scores per audio frame
       and find the frame with the highest combined score.
    3. Emit a fixed-width label window (default 3s) centered on that peak.

    The restart whistle is deliberately excluded from this peak search — it's kept as a
    separate, later signal that can be engineered as its own feature rather than smeared
    into the positive-label definition.
    """
)

st.header("Processed Artefacts")
st.markdown(
    """
    - **Audio** — extracted per match via ffmpeg (mono, 22.05kHz WAV).
    - **Features** — one Parquet file per match at YAMNet's native cadence (0.96s window /
      0.48s stride): 11 raw YAMNet score columns + 16-dim PCA-reduced embedding, plus a
      `_meta.csv` sidecar recording the run configuration.
    - **PCA transform** — fit on training matches only, saved to `pca_transform.npz`, and
      reused (never refit) on test matches and new footage at inference time — this avoids
      leaking test-set embedding structure into the projection.
    - **Labels** — one CSV row per extracted label: match, clip, original clip bounds, peak
      time/score, and refined label window — kept alongside the original bounds for auditing.
    """
)

st.header("Data Quality & Governance")
st.info(
    "Known limitations: dataset size (~31 matches) constrains model complexity; label quality "
    "depends on the accuracy of the peak-finding heuristic; only 'Goal' clips are currently "
    "used as positive labels."
)