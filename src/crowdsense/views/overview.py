"""Welcome page: project summary, headline numbers, and the clickable pipeline map."""

import streamlit as st

from constants import RESULTS
from src.crowdsense.nav import render_map

st.session_state.pop("focus", None)

st.title("CrowdSense - Futsal Highlight Detector")
st.caption("Audio-based highlight detection for full-match futsal footage.")

st.header("Project Summary")
st.markdown(
    """
    Manually scrubbing through 1.5–2 hours of match footage to find highlights is slow.
    Processing multi-gigabyte video files with computer vision can also be expensive.
    CrowdSense takes a cheaper route: it uses **audio alone** to flag possible highlight
    clips. At the reported test operating point, candidate windows used 28% of the footage
    and fully covered 79% of editor-labelled Goal clips, leaving about 72% less footage to review.
    """
)

st.header("Context")
context = st.columns(3)

with context[0].container(border=True):
    st.subheader("What is futsal?")
    st.markdown(
        """
        Futsal is a fast-paced, five-a-side form of football played on a hard court. FIFA's
        estimate cited by UEFA is more than 30 million players worldwide. Clubs can use
        recorded matches and short clips to share the game with their communities.
        [The FA's futsal introduction](https://www.thefa.com/get-involved/referee/laws-of-the-game)
        describes the five-a-side format; [UEFA's 2026 futsal overview](https://www.uefa.com/futsaleuro/news/02a1-1fb24d88d6af-dfa612145c89-1000--growing-momentum-futsal-s-30-year-rise/)
        cites FIFA's global participation estimate.
        """
    )

with context[1].container(border=True):
    st.subheader("What is Veo?")
    st.markdown(
        """
        Veo makes automated sports recording cameras. Veo says its cameras capture a 180°
        pitch view and use AI to create a follow-cam view. Features such as AI clips and
        analytics depend on the sport, product and plan; its current published AI clip guide
        lists different clip types by sport. See [Veo's camera overview](https://www.veo.com/product/veo-cam-3)
        and [current AI clip support](https://support.veo.com/hc/en-us/articles/28730302256785-Understanding-AI-generated-clips-in-Veo-recordings).
        """
    )

with context[2].container(border=True):
    st.subheader("The problem")
    st.markdown(
        """
        This project began after the club that supplied the recordings found it had to review
        its futsal footage manually to find shareable moments. That is the club's experience,
        not a claim that Veo removed one feature for every customer in 2024. Veo's published
        support list is sport-specific and currently does not list futsal among the sports
        with AI-generated clips.
        """
    )
st.markdown(
    """
    CrowdSense was inspired by a conversation with an old coach, who explained the dilemma
    and provided access to the club's Veo footage and the highlight clips they had manually
    extracted.
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
    This project is a proof of concept. See Performance for the held-out evaluation and its
    limitations. The app provides a demo rather than an upload-and-export service. Full mode
    extracts features from the WAV before scoring; Quick mode scores saved features. Both
    produce candidate windows for human review. The results are pooled across eight held-out
    matches, not a guarantee for every match or venue. See the pipeline map below for details.
    """
)

cols = st.columns(2)
cols[0].metric("Goal recall", f"{RESULTS['recall']:.0%}")
cols[1].metric("Footage budget", f"{RESULTS['budget']:.0%}")
st.caption(
    "Recall = share of editor-labelled Goal clips fully covered by a candidate window. Budget = "
    "candidate footage as a fraction of raw match length. Pooled over the 8 held-out matches."
)

st.caption("External sources checked 3 October 2026; linked product features and participation estimates can change.")

st.header("Pipeline Map")
render_map()
