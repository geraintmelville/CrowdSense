"""Modelling: targets, tuning, training, and the final model."""

from contextlib import contextmanager

import streamlit as st


@contextmanager
def section(title):
    with st.container(border=True):
        st.subheader(title)
        yield


st.title("Modelling")
st.caption("How features + labels are used to generate a predictive model.")

with section("`XGBoost`"):
    st.markdown(
        """
        We use `XGBoost` (eXtreme Gradient Boosting), an open-source machine learning 
         algorithm with a `scikit-learn` wrapper, making it easy to implement. It
        builds a strong classifier out of many small decision trees, added one at a time.
        Each new tree is trained to correct the errors the previous trees made, by
        fitting the gradient of the loss function. The trees are individually weak 
        but together they capture non-linear patterns and feature interactions. 
        
        This model was chosen for its ease of use and strong regularisation, compared 
        with equivalent `scikit-learn` models, like `GradientBoostingClassifier` and 
        `HistGradientBoostingClassifier`, which prevents overfitting.
        """
    )

with section("`tune_model.py`"):
    st.markdown(
        """
        `XGBoost` has various hyperparameters, changing these values changes how the model
        learns, this allows the model to be used on a wide range of machine-learning
        tasks. There are also the previously mentioned parameters, `lookback`, `postroll`
        and `merge_gap` which govern how candidate windows are built from `XGBoost` ouputs.

        In order to find the optimal values for the hyperparameters, we conduct a 'random grid-search'.
        We set out a grid of possible values we want to check, then 
        iteratively take a random combination of values, train a model using them and record its 
        performance. We then look for trends in the best performing values. 
        
        To ensure we do this robustly, we use 4-fold grouped cross validation. We take 
        our 24 training matches, split it into 4 folds (sets), train the model on 3 of those sets
        and predict on the remaining set. We repeat this 4 times, changing the hold-out set each
         time, so each match has predicted probabilities from a model that did not see it during
        training. Cross-validation is essentially implementing the train-test idea
        mentioned above on the train set itself, to ensure fair evaluation.

        In order to find the optimal values for the window parameters, we produce a grid of 
        values for `lookback`, `postroll` and `merge_gap`. For each value combination, we sweep over all
        thresholds, construct the candidate windows using the values and evaluate the partial AUC (
        25-40% budget). This allows us to rank different combinations and choose the best one, the
        values for the final model are:
        """
    )
    st.markdown(
        """
        <table>
          <thead>
            <tr>
              <th colspan="3">Hyperparameters</th>
              <th>Window parameters</th>
            </tr>
            <tr>
              <th>Tree structure</th>
              <th>Regularisation</th>
              <th>Sampling &amp; imbalance</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>n_estimators</code> = 1200</td>
              <td><code>min_child_weight</code> = 20</td>
              <td><code>subsample</code> = 0.6</td>
              <td><code>lookback</code> = 20s</td>
            </tr>
            <tr>
              <td><code>max_depth</code> = 2</td>
              <td><code>gamma</code> = 0.5</td>
              <td><code>colsample_bytree</code> = 1.0</td>
              <td><code>postroll</code> = 10s</td>
            </tr>
            <tr>
              <td><code>learning_rate</code> = 0.025</td>
              <td><code>reg_lambda</code> = 10</td>
              <td><code>scale_pos_weight</code> = 4</td>
              <td><code>merge_gap</code> = 10s</td>
            </tr>
            <tr>
              <td></td>
              <td><code>reg_alpha</code> = 0.1</td>
              <td></td>
              <td></td>
            </tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
      The model is configured with `objective` = `binary:logistic` and
      `eval_metric` = `aucpr` (XGBoost's internal evaluation metric); `random_state` = 42.
      
      This `eval_metric` is not the hyperparameter-search score: models and window
      settings are ranked by the custom partial recall-budget AUC over the 25-40% budget band.
      """
      )

with section("`save_final_model.py`"):
    st.markdown(
        """
        This script take the best values as described above, and trains a model on the full 24 match
        training set. This model is saved as `model.ubj` with a companion
        `model.json`. The artifact records the threshold, target budget, feature columns,
        YAMNet score indices, window and stride, PCA components and mean, and candidate-window
        settings. This model is then used for model evaluation and the demo.
        """
    )
with section("`eval_model.py`"):
    st.markdown(
      """
      This script takes the model described above and uses it to make predictions on the 8 test
      matches. The output probabilities are then swept over to produce the recall-budget curve,
      as described in the Metrics page. This curve can be seen in the Performance page.
      """
    )
