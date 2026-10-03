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
        `aucpr` eval metric). Candidates are ranked by **partial recall-budget AUC (25-40%)**
        on the pooled out-of-fold curve, swept across thresholds.
        """
    )

with section(P, "window_grid", "3 · Candidate-window grid 🔒 training matches only"):
    st.markdown(
        """
        For each sampled model, the model is fit once and its pooled OOF probabilities are
        swept against a grid of **lookback × postroll × merge-gap** settings. Reshaping existing
        probabilities into merged intervals is cheap, so no refitting is needed. The best
        combination is written to the results CSV with the winning parameters, and
        `save_final_model.py` loads it from the tuning results and stores it in the model artifact;
        `eval_model.py` reads the candidate-window settings from that artifact.
        For a manual final fit, `save_final_model.py` also accepts explicit XGBoost settings
        via `--model-params` (a JSON object) and explicit `--lookback`, `--postroll`, and
        `--merge-gap` values; supplying all of them avoids reading the tuning-results CSV.
        """
    )

with section(P, "final", "4 · Final model & deployment threshold 🔒 training matches only"):
    st.markdown(
        """
        `save_final_model.py` builds pooled grouped OOF predictions across the training matches
        and picks the operating point whose budget is closest to the configured target (33%
        placeholder by default), tie-breaking on recall. This closest-point target can land
        slightly above or below the target; test matches are excluded from threshold selection. The model is then refit
        on all non-test training matches and
        saved as `model.ubj` plus `model.json` (threshold, target budget, feature columns,
        YAMNet score indices, window/stride, PCA components + mean, and candidate-window settings).
        `eval_model.py` loads that artifact, scores the held-out test matches, reports the test
        recall-budget curve, and marks the training-selected threshold on the plot.
        """
    )
