"""Preprocessing: raw footage to features and labels."""

import streamlit as st

from src.crowdsense.nav import page_header, section

P = "preprocessing"
page_header(P, "Preprocessing", "From raw footage and editor clips to model-ready features and labels.")

with section(P, "ingestion", "1 · Building the database"):
    st.markdown(
        """
        Each match is a full raw video (32 matches, 1.5–2h) each with a ZIP folder of editor-cut highlight
        clips (30–70 per match, 5–60s). Each clip has naming convention clip_number timestamp_-_description
        where description can be: 'Goal', 'Chance', 'Save' etc.). 
        `build_clip_database.py` matches each match with its clip folders using the file names 
        (opposition + date). It then parses the names of each clip to find goals, inspects those clips to 
        find duration and writes to an SQLite database and exports the tables to csv.
        (`data/metadata/clips_data.db`) plus `matches.csv` / `clips.csv`:

        - **`matches`** — `match_id`, `raw_filename`, `zip_filename`, `match_date`, `length_sec`
        - **`clips`** — `match_id`, `clip_number`, `timestamp_formatted`,
          `filename`, `length_sec`

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
        YAMNet is an open-source pretrained deep neural network from Google that classifies
         0.96s audio frames into 521 categories and generates audio embeddings. It uses a 
         MobileNetV1‑based architecture, which is trained on the AudioSet-YouTube corpus, which
         contains over 1.5 million 10s YouTube audio clips. 

        This model allows us to generate numerical representations of audio frames as well as
        predictions of the presence of various audio events such as 'cheering', 'crowd', 
        'whistle' etc. This information can then be fed into a model along with labels to 
        predict the occurrence of a goal based on audio.
        
        Audio is resampled to 16 kHz and fed to YAMNet in bounded chunks at its **native
        cadence**: a 0.96s window every 0.48s (each frame has 50% overlap with neighbours),
        so each row is one untouched YAMNet frame. CrowdSense keeps 11 relevant class
        scores (Shout, Yell, Screaming, Whistling, Cheering,
        Applause, Crowd, Chatter, Hubbub, Clapping, Children shouting) plus the embedding.
        Output is one Parquet file per match and a `_meta.csv` recording the run configuration.

        [TensorFlow's YAMNet guide](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
        documents the 16 kHz input, 0.96s frames, 0.48s hop and 1,024-value embeddings.
        Class scores are audio-event scores, not probabilities that a goal occurred.
        """
    )

with section(P, "pca", "4 · PCA (16 dimensions) 🔒 training matches only"):
    st.markdown(
        """
        The YAMNet embedding a 1024-dimensional vector, in order to reduce the size of our feature set,
        we apply PCA, which reduces the dimensionality of a dataset, whilst maintaining the underlying 
        patterns. We reduce the dimensionality to 16 components with `IncrementalPCA`, which keeps ~83%
        of the variance (information). It is fit on **training matches only** and saved to 
        `pca_transform.npz` to be reused on test matches and the demo. It is never refit so no test 
        embedding structure leaks into the features.
        """
    )

with section(P, "labels", "5 · Label refinement"):
    st.markdown(
        """
        A clip's timestamp is the editor's cut, labelling the whole cut as positive dilutes the signal,
         for the model. Instead we refine the labels, identifying the audio spike we are looking for 
         and labelling that as the positive our model is searching for.
        `extract_labels.py` narrows each clip:

        1. Take the clip's `[start, start + length]` span, expanded by 5s on both sides.
        2. Sum the Cheering + Crowd + Applause scores per frame and find the peak frame.
        3. Emit a fixed 3s label window centred on that peak. 
        
        The labels CSV keeps both the refined window (used for training targets) and the original
        clip bounds (used to evaluate model performance), plus the peak time and score for auditing.
        The heuristic assumes the strongest Cheering/Crowd/Applause peak is near the goal; it
        can miss quiet goals or select an unrelated loud reaction due to controversial decision or 
         close chance within the search span.
        """
    )
