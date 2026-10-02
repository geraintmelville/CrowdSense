"""Welcome page: project summary, headline numbers, and the clickable pipeline map."""

import streamlit as st

from constants import RESULTS
from src.crowdsense.nav import render_map

st.session_state.pop("focus", None)

st.title("CrowdSense - Futsal Highlight Detector")
st.caption("Audio-based highlight detection for full futsal match veo footage.")

st.header("Project Summary")
st.markdown(
    """
    Manually scrubbing through 2 hours of match footage to find highlights is slow,
    automating it by feeding 4GB video files through computer vision models is expensive.
    CrowdSense takes a cheaper route: it uses **audio alone** to flag possible highlight
    clips, reducing the amount of footage a human needs to review by ~70%.
    """
)

st.header("Context")
context = st.columns(3)

with context[0].container(border=True):
    st.subheader("What is futsal?")
    st.markdown(
        """
        Futsal is a fast-paced, five-a-side, indoor version of football. It has ~60 million players worldwide but struggles with
         publicity, funding and media coverage due to its colossal parent sport - football. 
        Clubs rely on social media to drive engagement and grow the game.
        """
    )

with context[1].container(border=True):
    st.subheader("What is Veo?")
    st.markdown(
        """
        Veo is the main commercial option for automated sports video. Its cameras capture a
        180° view of the pitch, and its software tracks the ball allowing for high quality 
        match footage. They also provide highlight detection and statistics for football, 
        hockey and lacrosse.
        """
    )

with context[2].container(border=True):
    st.subheader("The problem")
    st.markdown(
        """
        Veo used to provide
        highlights for futsal matches as well, but the feature was removed in 2024 with no
        explanation. That leaves futsal clubs scrubbing through hours of footage by hand
        to find clips for their social media.
        """
    )
st.markdown(
    """
    CrowdSense was inspired by a conversation with an old coach, who explained the dilemma
    and provided me with access to all of the club's Veo footage, along with the highlight
     clips they had manually extracted.
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
st.markdown(
    """
    This was an ambitious project and the result is a proof-of-concept. The model itself
    performs quite well considering the fairly amount of data used to train it 
    (see performance page). The app only includes a demo, rather than the ability to upload
    full matches and export candidate highlights. There is a full and fast version, both
    using a ~25 min clip from a match not used in training or testing. The full version takes 
    the audio file, extracts the features and feeds them through the model. The fast
    version skips the first, simply feeding pre-saved features through the model. Each version 
    outputs a table specifying candidate highlights, which can be streamed from an unlisted
    YouTube video. See underneath for more detail on the pipeline.
    """
)

cols = st.columns(2)
cols[0].metric("Goal recall", f"{RESULTS['recall']:.0%}")
cols[1].metric("Footage budget", f"{RESULTS['budget']:.0%}")
st.caption(
    "Recall = share of true goal clips fully covered by a candidate window. Budget = candidate "
    "footage as a fraction of raw match length. Measured on the 8 most recent held-out matches."
)

st.header("Pipeline Map")
render_map()
