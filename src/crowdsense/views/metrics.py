"""Evaluation metrics used to measure candidate coverage."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from constants import RECALL_BUDGET_PLOT_PATH

from src.crowdsense.nav import page_header, section

P = "metrics"
page_header(P, "Metrics", "How we evaluate model performance.")
st.markdown(
    """
    The goal of CrowdSense is to reduce the amount of footage a human editor must watch
    to find goals. To do this we train the model to predict audio spikes from confirmed
    goal clips. The model outputs probabilities for each frame that measure the
    likelihood that a frame is an audio spike. To create predictions we choose
    a threshold, any frame with probability over that threshold is a positive. These 
    positive frames are then expanded to create a candidate window.
    """
)
with section(P, "metrics", "1 · Candidate windows"):
    st.markdown(
        """
        We take a positive frame (an audio spike) at `timestamp` and expand it into a
        candidate window with `(start,end) = (timestamp - lookback, timestamp + postroll)`.

        We then merge windows that are close together (within `merge_gap`) to create our
        final candidate windows, which then need to be checked by a human editor.

        The goal of CrowdSense is to reduce the amount of footage a human editor must watch
        whilst ensuring goals are not missed, this presents two natural metrics.
        """
    )
with section(P, "metrics", "2 · Recall"):
    st.markdown(
        """
        
        
        - **Recall** — We look at each goal clip in a match, if it is wholly contained
        within a candidate window it is a true positive or a 'hit'. Recall is then, 
        $recall = \frac{n_{hits}}{n_{goals}}$. This measure how well the model preserves goals.
        
        Note that this is a strict metric since a candidate window may contain the relevant
        footage for a goal, but if it misses any frames from the editor's cut it is counted
        as a miss.
        """
    )
with section(P, "metrics", "2 · Budget"):
    st.markdown(
        """
        - **Budget** — We take all the candidate windows for a match produced by our model
        and sum their durations. This is the amount of footage a human editor must check.
        Budget is then, $budget = \frac{n_{candidate_{seconds}}}{n_{match_{seconds}}}$. 
        This measures how effective the model is at reducing the amount of footage to check.
        """
    )
with section(P, "metrics", "3 · Area under the recall-budget curve"):
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
        
        Imagine a terrible model that just guesses at random, at $100%$ budget, recall would
        be $100%$, at $50%$ budget, recall would be $50%$ and at $0%$ budget, recall would 
        be $0%$, it would be a straight diagonal line from $(0,0)$ to $(1,1)$. There is a 
        theoretical limit to how good a model can be, we clearly cannot have $100%$ recall with
         $0%$ budget. A perfect model would have some threshold where its candidates are 
         exactly the confirmed clips, its recall is $100%$ and its budget would be the
         theoretical limit. As we vary the threshold, the curve would rise steeply from $(0,0)$
        until it hits $100%$ recall at the theoretical limit, then goes flat to $(1,1)$.
        
        We can therefore see that the area between our recall-budget curve and the diagonal 
        line is a measure of how good our model is, the more area the better. This therefore
        becomes the single metric we can use to evaluate and compare models.
        """
    )

diagram_path = Path(__file__).resolve().parents[1] / "recall_budget_curve_explainer_v2.html"
components.html(
    diagram_path.read_text(encoding="utf-8"),
    height=500,
    scrolling=False,
)