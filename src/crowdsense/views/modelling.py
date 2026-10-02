"""Modelling: targets, tuning, training, final model."""

import streamlit as st

from src.crowdsense.nav import page_header, section

P = "modelling"
page_header(P, "Modelling", "Targets, grouped cross-validation, tuning, and the final model.")

with section(P, "targets", "1 · Targets & grouped CV 🔒 training matches only"):
    st.markdown(
        """
        A window is positive if it overlaps a refined (peak-centred) label interval for its
        match. The 8 most recent matches are held out as the test set, because in deployment the
        model always sees new footage. All tuning uses `GroupKFold` grouped by match, so no fold
        trains on windows from a match it is also scored on.
        """
    )

with section(P, "tuning", "2 · Hyperparameter tuning 🔒 training matches only"):
    st.markdown(
        """
        `tune_model.py` runs a randomized search over XGBoost parameters (`binary:logistic`,
        `aucpr` eval metric). Candidates are ranked by the **area under the pooled out-of-fold
        recall-vs-budget curve**, swept across thresholds, rather than by a single operating
        point, so the ranking is not sensitive to one threshold choice.
        """
    )

with section(P, "window_grid", "3 · Candidate-window grid 🔒 training matches only"):
    st.markdown(
        """
        For each sampled model, the model is fit once and its pooled OOF probabilities are
        swept against a grid of **lookback × postroll × merge-gap** settings. Reshaping existing
        probabilities into merged intervals is cheap, so no refitting is needed. The best
        combination is written to the results CSV with the winning parameters, and
        `save_final_model.py` / `eval_model.py` load it by default.
        """
    )

with section(P, "train_score", "4 · Train & test scoring"):
    st.markdown(
        """
        `train_predict.py` fits the tuned model on every match except the 8 test matches and
        writes per-window probabilities for those test matches. No threshold is applied here;
        `eval_model.py` sweeps thresholds over those probabilities.
        """
    )

with section(P, "final", "5 · Final model & deployment threshold 🔒 training matches only"):
    st.markdown(
        """
        `save_final_model.py` builds pooled grouped OOF predictions across the training matches
        and picks the operating point whose budget is closest to the configured limit (30% by
        default), tie-breaking on recall. This closest-point target can land slightly above or
        below 30%; test matches are excluded from threshold selection. The model is then refit
        on all non-test training matches and
        saved as `model.ubj` plus `model.json` (threshold, feature columns, YAMNet score
        indices, window/stride, PCA components + mean, and the candidate-window settings).
        """
    )
