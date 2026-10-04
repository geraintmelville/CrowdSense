"""Preprocessing: raw footage to features and labels."""

from contextlib import contextmanager
import streamlit as st

@contextmanager
def section(title):
    with st.container(border=True):
        st.subheader(title)
        yield

st.title("Preprocessing")
st.caption("From raw footage and editor clips to model-ready features and labels.")

st.markdown(
    """
    The pipeline begins with 32 full match .mp4s (1.5–2h), each with a ZIP folder of editor-cut highlight
    clips (30–70 per match, 5–60s). Each clip has naming convention clip_number timestamp_-_description
    where description can be: 'Goal', 'Chance', 'Save' etc. 
    """
)
with section("`build_clip_database.py`"):
    st.markdown(
        """
        This script matches each mp4 with its clip folder using the file names 
        (opposition + date). It then parses the names of each clip to find goals, probes those clips to 
        find duration and writes to an SQLite database and exports the tables to csv.
        (`data/metadata/clips_data.db`) plus `matches.csv` / `clips.csv`:
        
        - **`matches`** — `match_id`, `raw_filename`, `zip_filename`, `match_date`, `length_sec`
        - **`clips`** — `match_id`, `clip_number`, `timestamp_formatted`,
        `filename`, `length_sec`
    """
    )
with section("`extract_audio.py`"):
    st.markdown(
        """
        This script runs `ffmpeg` on mp4s, extract the audio into mono 22.05 kHz PCM WAV files.
        It also checks the length of the audio against the match length in
        `matches.csv` and logs any discrepancies.
        """
    )

with section("`extract_features.py`"):
    st.markdown(
        """
        This script runs YAMNet on the WAVs to extract features that can be fed into the model.
        
        YAMNet is an open-source pretrained deep neural network from Google that classifies
         0.96s audio frames into 521 categories and generates audio embeddings. It uses a 
         MobileNetV1‑based architecture, which is trained on the AudioSet-YouTube corpus, which
         contains over 1.5 million 10s YouTube audio clips. This model allows us to detect 
         various audio events such as 'cheering', 'crowd', 'whistle' etc.
        
        Audio is resampled to 16 kHz and fed to YAMNet in bounded chunks at its **native
        cadence**: a 0.96s window every 0.48s (each frame has 50% overlap with neighbours),
        so each row is one untouched YAMNet frame. CrowdSense keeps 11 relevant class
        scores (Shout, Yell, Screaming, Whistling, Cheering,
        Applause, Crowd, Chatter, Hubbub, Clapping, Children shouting) plus the embedding.

        The YAMNet embedding a 1024-dimensional vector, in order to reduce the size of our feature set,
        we apply PCA, which essentially compresses the data. We reduce the dimensionality to 16 components with `IncrementalPCA`, which keeps ~83%
        of the variance (information). It is fit on **training matches only** and the parameters are saved to 
        `pca_transform.npz` to be reused on test matches and the demo. It is never refit so no test 
        embedding structure leaks into the features.

        Th output is a parquet file per match, with one row per YAMNet frame, and columns
         consisting of the 11 class scores and the 16 PCA components.

        [TensorFlow's YAMNet guide](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
        """
    )

with section("`extract_labels.py`"):
    st.markdown(
        """
        We are aiming to build a model that identidies goals based on the crowd spike after a goal.
        In order to train a model to do this, we must manually identify these spikes to 
        provide labels it can learn from. If instead we gave it the whole clip, the signal 
         from the crowd spike would be diluted by the rest of the clip. To do this we:

        1. Take the clip's `[start, start + length]` span, expanded by 5s on both sides.
        2. Sum the Cheering + Crowd + Applause scores per frame and find the peak frame.
        3. Define a 3s window centred on that peak, that serves as a positive label. 
        
        We ouput `labels.csv` which keeps both the positive windows (used for training targets) and the original
        clip bounds (used to evaluate model performance), plus the peak time and score for auditing.
        The heuristic assumes the strongest Cheering/Crowd/Applause peak is near the goal; it
        can miss quiet goals or select an unrelated loud reaction due to controversial decision or 
         close chance within the clip.
        """
    )
