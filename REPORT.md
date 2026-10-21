# CrowdSense: method and performance report

## Project objective

Manually searching long futsal recordings for goals is slow. CrowdSense uses audio alone to identify likely goal moments and return candidate footage for an editor to review. This keeps the processing focused on audio rather than running computer vision over multi-gigabyte video files. The output is a shortlist of candidate windows, not an automatic goal verdict or a finished highlight video.

## Data and split

The dataset contains 32 full-match recordings, each approximately 1.5–2 hours long, with 30–70 manually curated highlight clips per match. Clip names provide a match timestamp and description such as Goal, Chance, or Save. Goal clips are used to create positive training labels and as the reference clips for evaluation.

The fixed experiment uses 24 matches for model development and eight held-out matches for final evaluation. Model tuning uses four-fold grouped cross-validation on the 24 training matches, keeping every frame from a match in the same fold. The test matches are excluded from feature projection fitting, model and threshold selection, and final model fitting.

## Method

### Audio features

The match audio is extracted from the MP4 as mono, 22.05 kHz PCM WAV. YAMNet then resamples it to 16 kHz and produces an output frame for each 0.96-second audio window, with a new window every 0.48 seconds (50% overlap). YAMNet is a pretrained audio model that provides scores for 521 sound categories and a 1024-dimensional embedding for each frame.

CrowdSense keeps 11 sound-category scores associated with crowd reactions: Shout, Yell, Screaming, Whistling, Cheering, Applause, Crowd, Chatter, Hubbub, Clapping, and Children shouting. It reduces the embedding to 16 principal components using Incremental PCA, retaining approximately 83% of its variance. PCA is fitted on training matches only; its saved mean and components are reused for test and demo audio.

### Training labels

An editor's goal clip can include a buildup and other activity that dilute the acoustic signal of interest. For each goal clip, CrowdSense therefore:

1. Takes the clip's timestamp and duration and expands that interval by five seconds at each end.
2. Adds the Cheering, Crowd, and Applause scores for each frame in the expanded interval.
3. Finds the peak-scoring frame and labels the three-second interval centred on that peak as positive.

The refined three-second interval is the positive training target. The original editor clip bounds are retained for evaluation. This heuristic assumes the strongest crowd-related peak is close to the goal; a quiet goal or another loud reaction can produce a weak or misplaced label.

### Classifier and tuning

The classifier is XGBoost with a `binary:logistic` objective. It builds an ensemble of decision trees, adding trees sequentially to correct earlier errors. Tuning uses randomized hyperparameter search and four-fold grouped cross-validation. For each setting, out-of-fold probabilities are converted to candidate windows over a sweep of thresholds and window settings. Settings are ranked using the custom partial recall-budget AUC for the 25–40% footage-budget band. XGBoost's internal `aucpr` evaluation metric is not the search ranking metric.

The final settings documented in the app are:

| Category | Setting | Value |
|---|---|---:|
| Tree structure | `n_estimators` | 1200 |
| Tree structure | `max_depth` | 2 |
| Tree structure | `learning_rate` | 0.025 |
| Regularisation | `min_child_weight` | 20 |
| Regularisation | `gamma` | 0.5 |
| Regularisation | `reg_lambda` | 10 |
| Regularisation | `reg_alpha` | 0.1 |
| Sampling and imbalance | `subsample` | 0.6 |
| Sampling and imbalance | `colsample_bytree` | 1.0 |
| Sampling and imbalance | `scale_pos_weight` | 4 |
| Reproducibility | `random_state` | 42 |

The candidate-window settings are a **20-second lookback**, **10-second postroll**, and **10-second merge gap**.

## Candidate windows and metric definitions

The classifier produces a probability for each audio frame. A frame is positive if its probability is above the chosen threshold. Each positive frame at time `timestamp` creates a candidate interval:

```text
start = timestamp - lookback
end   = timestamp + postroll
```

Nearby candidate intervals are merged when separated by no more than `merge_gap`. The merged windows are the footage an editor must review.

**Goal recall** is the fraction of editor-labelled goal clips fully contained inside at least one candidate window:

```text
recall = number of fully covered goal clips / number of goal clips
```

This is a strict definition: a clip counts as a miss if any part of its original editor cut falls outside the candidate windows, even if the relevant goal moment itself is present.

**Footage budget** is the total duration of the candidate windows divided by the match duration:

```text
budget = total candidate-window duration / match duration
```

It estimates the fraction of a match an editor needs to inspect. Lower is better, while higher recall is better.

### Recall-budget curve and partial AUC

Changing the probability threshold changes how much footage is flagged. A high threshold generally produces a lower budget and lower recall; a low threshold generally produces a higher budget and higher recall. Plotting recall against budget across thresholds makes this trade-off visible.

A random detector would follow the diagonal: for example, reviewing 50% of footage would find about 50% of goals. The model is better when its curve achieves more recall at a given budget. The app summarizes performance for tuning with the area between the curve and the diagonal over the **25–40% budget band** (partial AUC).

The intended practical budget is approximately 33%. Since the frame-based predictions produce discrete candidate windows, no threshold is guaranteed to hit exactly 33%; the reported operating point uses the threshold closest to that target. The full recall-budget curve is also reported to show performance beyond that single point.

## Pipeline

The end-to-end workflow is:

1. **Build clip metadata** (`build_clip_database.py`): pair match videos and clip archives, parse goal clip timestamps, probe clip durations, and write match and clip metadata.
2. **Extract audio** (`extract_audio.py`): convert each MP4's audio to mono 22.05 kHz WAV.
3. **Extract features** (`extract_features.py`): run YAMNet, retain 11 selected sound scores, fit training-only PCA, and write one feature file per match.
4. **Extract labels** (`extract_labels.py`): locate crowd-reaction peaks around goal clips and save refined training intervals alongside original editor clip bounds.
5. **Tune** (`tune_model.py`): search classifier and window settings with grouped cross-validation, ranking candidates by partial AUC in the 25–40% budget band.
6. **Train and save** (`save_final_model.py`): select a training-only threshold near the target budget, fit the final classifier on all 24 training matches, and save the model with its inference configuration.
7. **Evaluate** (`eval_model.py`): score the eight held-out matches and produce the recall-budget performance curve.

The Streamlit app includes a pipeline map and pages describing preprocessing, modelling, metrics, and held-out results. Its demo applies the saved model and presents candidate clips for review.

## Held-out performance

On the eight held-out test matches, the app reports:

| Measure | Result | Interpretation |
|---|---:|---|
| Goal recall | **91%** | Fully covers 91% of the editor-labelled goal clips at the selected operating point. |
| Footage budget | **31%** | Flags 31% of match footage for human review. |
| Partial AUC (25–40% budget) | **0.92** | Summarizes the recall-budget curve over the budget range used for tuning. |

The recall and budget figures use the deployment threshold selected on training data to target a budget close to 33%; the observed result is 31% because the available frame thresholds yield discrete outcomes. The partial AUC gives a broader view across the stated budget band. Test results are for final assessment and should not be used to retune the model.

## Limitations

- The dataset is small: 32 matches, with performance measured on one fixed split of eight held-out matches.
- Editor clips are the reference labels, and the positive class contains Goal clips only.
- The peak-finding label heuristic can miss quiet goals or select another loud reaction.
- A possible improvement is adding temporal features for the two-phase goal signature described in the app: crowd spike, quiet period, then restart whistle.
