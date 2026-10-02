"""Preprocessing: raw footage to features and labels."""

import streamlit as st

from src.crowdsense.nav import page_header, section

P = "preprocessing"
page_header(P, "Preprocessing", "From raw footage and editor clips to model-ready features and labels.")

with section(P, "ingestion", "1 · Ingestion & clip database"):
    st.markdown(
        """
        Each match is a full raw video (~31 matches, 1.5–2h) plus a ZIP of editor-cut highlight
        clips (30–70 per match, 5–60s). `build_clip_database.py` matches raw files to ZIPs by
        normalized opposition name + date and writes a SQLite database
        (`data/metadata/clips_data.db`) plus `matches.csv` / `clips.csv`:

        - **`matches`** — `match_id`, `raw_filename`, `zip_filename`, `match_date`, `length_sec`
        - **`clips`** — `match_id`, `clip_number`, `timestamp_formatted`, `description`,
          `filename`, `length_sec`

        Only clips described as "Goal" are kept. Unmatched raw files and ambiguous or unparsable
        ZIPs are flagged during ingestion rather than silently dropped.
        """
    )

with section(P, "audio", "2 · Audio extraction"):
    st.markdown(
        """
        `extract_audio.py` uses ffmpeg to write mono, 22.05 kHz PCM WAV per match. The WAV
        duration is also the match length used as the denominator of the footage budget.
        """
    )

with section(P, "features", "3 · YAMNet features"):
    st.markdown(
        """
        Audio is resampled to 16 kHz and fed to YAMNet in bounded chunks at its **native
        cadence**: a 0.96s window every 0.48s, so each row is one untouched YAMNet frame.
        Each frame keeps 11 AudioSet class scores (Shout, Yell, Screaming, Whistling, Cheering,
        Applause, Crowd, Chatter, Hubbub, Clapping, Children shouting) plus the embedding.
        Output is one Parquet file per match and a `_meta.csv` recording the run configuration.
        """
    )

with section(P, "pca", "4 · PCA (16 dimensions) 🔒 training matches only"):
    st.markdown(
        """
        The YAMNet embedding is reduced to 16 components with `IncrementalPCA`. It is fit on
        **training matches only** and saved to `pca_transform.npz`. The same projection is
        reused, never refit, on test matches and on new footage at inference time, so no test
        embedding structure leaks into the features.
        """
    )

with section(P, "labels", "5 · Label refinement"):
    st.markdown(
        """
        A clip's timestamp is the editor's cut: lead-in, crowd-roar spike, quiet gap, sometimes
        the restart whistle. Training on the whole cut dilutes the signal, so
        `extract_labels.py` narrows each clip:

        1. Take the clip's `[start, start + length]` span, expanded by 5s on both sides.
        2. Sum the Cheering + Crowd + Applause scores per frame and find the peak frame.
        3. Emit a fixed 3s label window centred on that peak.

        Whistling is deliberately excluded; it stays available as a separate feature. The
        labels CSV keeps both the refined window (used for training targets) and the original
        clip bounds (used to score recall), plus the peak time and score for auditing.
        """
    )
