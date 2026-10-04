"""Welcome page: project summary, headline numbers, and the clickable pipeline map."""

import streamlit as st

from constants import RESULTS
from crowdsense.pipeline_map import render_pipeline_map

st.session_state.pop("focus", None)

st.title("CrowdSense - Futsal Highlight Detector")
st.caption("Audio-based highlight detection for full-match futsal footage.")

st.header("Project Summary")
st.markdown(
    """
    Manually scrubbing through hours of match footage to find highlights is slow.
    Processing multi-gigabyte video files with computer vision is expensive.
    CrowdSense takes a cheaper route: it uses **audio alone** to flag possible highlight
    clips. The model ingests (24) mp4 files, extracts the audio, and feeds them through YAMNet, a
    pre-trained deep neural-network audio embedding model. It takes the features output
    by YAMNet, and uses human curated goal clips to label them as positive or negative. This 
    data is used to train a supervised machine learning model, XGBoost, to produce candidate highlight clips,
    which should then be reviewed by a human editor. See the pipeline map at the bottom
    of this page for more details.
    """
)
st.header("Headline Figures")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 4.5rem; font-weight: 700; line-height: 1.1; }
    [data-testid="stMetricLabel"] p { font-size: 1.1rem; font-weight: 600; }
    [data-testid="stMetricDelta"] { font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(2)

with cols[0].container(border=True):
    st.metric(
        "Goal recall",
        f"{RESULTS['recall']:.0%}"
        )
    st.caption(
        "Share of Goals within a candidate clip, how well the model finds goals. "
        "Higher is better."
    )

with cols[1].container(border=True):
    st.metric(
        "Footage budget",
        f"{RESULTS['budget']:.0%}"
    )
    st.caption(
        "Candidate footage as a fraction of match length, i.e. how much the "
        "editor still has to review. Lower is better."
    )
st.markdown(
    """
    The final model's outputs used 31% of the footage
    and fully covered 91% of editor-labelled Goal clips, meaning a human is left with less
    than 1/3 of the footage to review."""
)
st.header("Context")
context = st.columns(3)

with context[0].container(border=True):
    st.subheader("What is futsal?")
    st.markdown(
        """
        Futsal is a fast-paced, five-a-side form of football played on a hard, indoor court.
        It has 30 million players worldwide, but struggles with publicity, funding and media
        coverage, especially in the UK. Clubs rely on social media to increase visibility and
         grow the game.
        """
    )

with context[1].container(border=True):
    st.subheader("What is Veo?")
    st.markdown(
        """
        Veo is the main commercial option for automated sports video: its cameras capture a
        180° view of the pitch, and its software tracks the ball to produce high quality
         footage in tight spaces like an indoor court. They provide software that generates
        highlights and statistics for sports like football, hockey and lacrosse.
        """
    )

with context[2].container(border=True):
    st.subheader("The problem")
    st.markdown(
        """
        CrowdSense was inspired by a conversation with an old coach, who explained that Veo
        used to provide highlight detection but the feature was
        eventually removed with no explanation. He also provided access to all of the club's
         Veo footage and the highlight clips they had manually curated.
        """
    )
st.markdown(
    """
    Access to a large amount of labelled data opened the door to a supervised machine
    learning solution. The initial goal was to use a computer vision model, just as Veo do.
    However, each match being 1.5–2 hours and 3-4GB in size, the compute cost would have
    made the project infeasible. In order to stick to my budget of £0, I had to reduce the
    amount of data that needed to be processed, whilst keeping enough information to train
    a model. I realised that audio alone could be used to flag possible highlight clips,
     and that became the focus of the project.
    """
)
st.header("Pipeline Map")
render_pipeline_map()
st.markdown(
    """
    This project is a proof of concept. The model itself performs quite well considering
    its design was limited by low computational resources. The app provides a demo rather
     than the ability to upload matches and export candidates. An earlier version did
      include cloud upload to an s3 bucket, a Lambda function to extract the audio and
    send it back to the app for feature extraction and inference, however this was extremely
     slow and was again hampered by the lack of budget. The demo runs on a single match,
     not involved in testing or training. Full mode extracts features from
     the WAV before scoring; Quick mode scores saved features. Both
    produce candidate windows which can be reviewed in app via YouTube.
    """
)
