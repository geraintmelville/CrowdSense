"""Model and YAMNet pipeline documentation page."""

import streamlit as st


st.title("Model & YAMNet Pipeline")
st.caption("Feature extraction, PCA, XGBoost training, and evaluation.")

st.header("Pipeline Stages")
st.markdown(
    """
    1. **Audio preparation** — ffmpeg extracts mono 22.05kHz audio from raw match video.
    2. **YAMNet features** — audio is fed to YAMNet at its native frame cadence: a 0.96s
       window every 0.48s stride, so every output row is one untouched YAMNet frame (no
       window-aggregation step). Each frame yields 11 selected AudioSet class scores
       (Shout, Yell, Screaming, Whistling, Cheering, Applause, Crowd, Chatter, Hubbub,
       Clapping, Children shouting) plus a general-purpose audio embedding.
    3. **PCA reduction** — the embedding is reduced from its native dimensionality to 16
       components, fit once on training matches only (`IncrementalPCA`, batched) and reused
       everywhere downstream, including on the held-out test matches, to avoid leaking
       test-set structure into the projection.
    4. **Classification** — an XGBoost classifier (`binary:logistic`, `aucpr` eval metric)
       scores each window: probability that it falls inside a goal event.
    5. **Post-processing** — windows above the decision threshold are expanded (lookback
       before, postroll after) and merged if they're within `merge_gap` of each other,
       producing a short list of candidate clips per match instead of a per-window score.
    """
)

st.header("Labels & Targets")
st.markdown(
    """
    Training targets come from the peak-centered label windows described on the
    **Dataset & Ingestion** page (not the raw editor clip bounds) — a window is a positive
    if it overlaps a refined label interval for its match.
    """
)

st.header("Hyperparameter Tuning")
st.markdown(
    """
    `tune_model.py` runs a randomized search over XGBoost hyperparameters, evaluated with
    **grouped, out-of-fold (OOF) cross-validation** (`GroupKFold`, grouped by match) so no
    fold ever sees windows from a match it's also training on. Candidates are ranked by the
    **area under the recall-vs-budget curve** — swept across decision thresholds — rather
    than a single fixed operating point, so the ranking isn't sensitive to one threshold choice.

    For every sampled set of model hyperparameters, the model is fit once and the resulting
    pooled OOF probabilities are then swept against a **grid of candidate-window settings**
    (lookback × postroll × merge-gap) to find the best-performing combination — this reuses
    the same probabilities rather than refitting for every window-setting combination, since
    reshaping already-pooled probabilities into merged intervals is cheap.
    """
)

st.header("Training & Test-Set Scoring")
st.markdown(
    """
    `train_predict.py` fits the final tuned model on every match **except** the 8 held-out
    test matches, then writes per-window probabilities for those test matches — no threshold
    is applied here. `eval_model.py` sweeps thresholds over those probabilities and reports
    the recall-vs-budget curve (recall at several budget checkpoints, plus overall AUC).
    """
)

st.header("Current Results")
res = st.columns(3)
res[0].metric("Goal recall", "79%")
res[1].metric("Footage budget", "28%")
res[2].metric("Test matches", "8 (most recent)")
st.caption(
    "Up from an earlier 61% recall / 42% budget baseline, after fixing a labelling bug "
    "(full editor clip bounds → peak-centered narrow labels)."
)

st.header("Deployment Threshold")
st.markdown(
    """
    `save_final_model.py` selects a single deployment threshold: the highest-recall
    operating point whose candidate-footage budget stays within a configured limit (30% by
    default), using pooled grouped out-of-fold predictions across the training matches.
    The curve uses the same recall-vs-budget calculation as tuning and evaluation, but is
    built from OOF probabilities so each threshold score comes from a model that did not
    train on that match. The final deployment model is then refit on all training matches.
    """
)

st.header("Planned Improvements")
st.markdown(
    """
    - Temporal pattern features capturing the two-phase acoustic signature of a goal
      (crowd spike → quiet period → restart whistle) — from cheap post-hoc rolling filters
      up to a sequence model as a stretch goal.
    - Sweeping a frame-merge factor against already-cached probabilities (no re-extraction).
    - Per-match recall breakdown in `eval_model.py`, not just the pooled curve.
    """
)