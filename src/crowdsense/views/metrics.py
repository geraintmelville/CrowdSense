"""Evaluation metrics used to measure candidate coverage."""

from pathlib import Path
from contextlib import contextmanager

import streamlit as st
import streamlit.components.v1 as components
from constants import RECALL_BUDGET_PLOT_PATH

@contextmanager
def section(title):
    with st.container(border=True):
        st.subheader(title)
        yield
st.title("Metrics")
st.caption("How we evaluate model performance.")
st.markdown(
    """
    The goal of CrowdSense is to reduce the amount of footage a human editor must watch
    to find goals. To do this we train the model to predict the probability that an audio frame
     is a crowd spike. To create a useful prediction, we choose a threshold probability and any 
     frame over that threshold is a positive. These positive frames are then expanded to create
    a full clip that includes the buildup and goal itself, we call these *candidate windows*. 
    We then need a single metric to evaluate and compare models based on the candidate windows it produces.
    """
)
with section("Candidate windows"):
    st.markdown(
        """
        We take a positive frame (an audio spike) at `timestamp` and expand it into a
        candidate window with `(start,end) = (timestamp - lookback, timestamp + postroll)`.

        We then merge windows that are close together (within `merge_gap`) to create our
        final candidate windows, which then need to be checked by a human editor.

        We want our candidate windows to contain goals, and to be as short as possible, this
         presents two natural metrics, which can be combined to give a single metric.
        """
    )
with section("Recall"):
    st.markdown(
        """
        We look at each goal clip in a match, if it is wholly contained
        within a candidate window it is a true positive or a 'hit'. Recall is then, 
        `n_hits / n_goals`. This measure how well the model preserves goals.
        
        Note that this is a strict metric since a candidate window may contain the relevant
        footage for a goal, but if it misses any frames from the editor's cut it is counted
        as a miss.
        """
    )
with section("Budget"):
    st.markdown(
        """
        We take all the candidate windows for a match produced by our model
        and sum their durations. This is the amount of footage a human editor must check.
        Budget is then, `length_of_candidates / length_of_match`. 
        This measures how effective the model is at reducing the amount of footage to check.
        """
    )
with section("Area under the recall-budget curve"):
    explanation, chart = st.columns([0.75, 1], gap="large")
    with explanation:
        st.markdown(
            """
        We now have two competing goals: reduce the budget while maximising recall. The 
        two pull against each other, since flagging more footage will always find more 
        goals, and flagging less will always miss some, so neither number means much 
        without the other. The recall-budget curve makes this trade-off explicit.
         
        As mentioned above, the threshold controls the amount of positive predictions, 
        and therefore the recall and budget. A high threshold flags very little footage,
        giving a low budget but a low recall, while a low threshold flags almost 
        everything, giving high recall at a high budget. We can vary the threshold and
        plot the recall and budget it produces, this plot allows us to choose a desired budget 
        and read the recall achieved. 
        
        Imagine a terrible model that just guesses at random, at 100% budget, recall would
        be 100%, at 50% budget, recall would be 50% and at 0% budget, recall would 
        be 0%, it would be a straight diagonal line from (0%,0%) to (100%,100%). 
        
        We want to minimise budget at each recall, however there is a limit to how 
        good a model can be. Clearly we cannot have 100% recall with
         0% budget, the minimum budget we can have at 100% recall is `limit = confirmed_clip_seconds / match_seconds`.
         A theoretical perfect model's curve would rise steeply until it hits 100% recall at `limit`.
        
        We can therefore see that the area between our recall-budget curve and the diagonal 
        line is a measure of how good our model is, the more area the better. This
        becomes the single metric we can use to evaluate and compare models.
            """
        )
    with chart:
        diagram_path = Path(__file__).resolve().parents[1] / "recall_budget_curve_explainer_v2.html"
        components.html(
            diagram_path.read_text(encoding="utf-8"),
            height=600,
            scrolling=False,
        )
with section("25-40% band"):
    st.markdown(
        """
    In order to produce useful outputs, we are aiming for a budget of ~33%. Therefore, in practice
    we only evaluate the are under the curve between 25-40%. This is the metric that is used in
    model tuning. To report headline metrics, as in the Welcome page, we choose a fixed threshold 
    that aims for 33% budget and report the corresponding recall. Due to the discrete nature of this
     approach (we have distinct frames of audio), we cannot necessarily find a threshold that
    corresponds with exactly 33% budget, instead we choose a threshold that gets us as close as possible,
    hence the 31% figure in the headline metrics. In the performance page, the full recall-budget
    curve is reported for a full picture of model performance.
    """
    )