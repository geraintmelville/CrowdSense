# CrowdSense

CrowdSense is an audio-based futsal highlight detection model. It processes full-match audio recordings with YAMNet, a pretrained audio model, and trains an XGBoost classifier to identify likely goal moments. Due to memory constraints, the app showcases a demo which live processes a pre-extracted 25 minute audio file and produces candidate clips which can be reviewed in app through a YouTube stream.

The workflow is: match footage and manually curated goal clips → audio and clip metadata → YAMNet features and refined labels → model tuning and training → candidate windows for review. The Streamlit app's **Welcome** page contains a clickable pipeline map; the **Deep Dive Docs** pages provide a detailed explanation of preprocessing, metrics, modelling, and held-out performance.

## Requirements

- Python 3.10 or newer
- Enough disk space for the source videos, extracted WAV files, and feature files (the full-match inputs can be several gigabytes each)
- No system-wide FFmpeg install is required: `imageio-ffmpeg` supplies FFmpeg for video audio extraction and clip duration probing

The application and pipeline dependencies are listed in `requirements.txt` and `requirements-pipeline.txt`. The latter currently includes the former and is the recommended install for running every stage.

## Run the app locally

From the repository root, create and activate a virtual environment, install dependencies, and start Streamlit:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app's demo extracts YAMNet features from a WAV in `demo/raw/audio/` before scoring. It uses the trained model pair `data/modelling/final_model/model.ubj` and `data/modelling/final_model/model.json`. The JSON sidecar is part of the model bundle: it contains the feature schema, PCA projection, and candidate-window settings. If either file is absent, run the training pipeline below or provide an existing compatible model at that path.

The demo links candidates to the match on YouTube using `CROWDSENSE_DEMO_YOUTUBE_URL`. It defaults to the demonstration match configured in `constants/constants.py`; set the environment variable to use another video. Candidate timestamps are relative to the start of the selected WAV, so that audio and video need to be time-aligned.

## Training data

Training needs full-match videos **and** a ZIP of the corresponding manually curated highlight clips for each match. Goal clips are processed to find a 3s window containing the crowd-spike, which provides a positive example and is saved to `data/processed/labels/labels.csv; other periods in the match become negative examples.

Place input files here (raw media is ignored by Git):

```text
data/raw/
├── video/   # one full match MP4 per match
└── clips/   # one highlight ZIP per match
```
### File naming and clip contents

- Match video names must follow `teams-YYYY-MM-DD.mp4`, for example `mens-a-v-baku-united-2026-04-19.mp4`.
- The matching ZIP uses the same teams/opposition and date. Spaces, hyphens, and underscores are normalized; an optional `highlights` word is accepted. Example: `Men's A v Baku United_highlights_2026-04-19.zip`.
- ZIP members containing goal labels must be named like `clip_number HHMMSS_-_Goal.mp4`, for example `12 004215_-_Goal.mp4`. The timestamp is the clip's start time in the full match, formatted as `HHMMSS`; the clip duration is probed from the video. Other descriptions are ignored for positive goal labels. The clip database parser accepts extensions such as `.mp4` and `.mov`.
- The ZIP and MP4 must each identify a match unambiguously by teams and date. Unmatched or duplicate ZIP identities are flagged by `build_clip_database.py` and need to be fixed before trusting the resulting metadata.

The fixed held-out test set is `TEST_MATCH_IDS` in `constants/constants.py` (currently match IDs 6, 22, 19, 13, 30, 26, 32, and 25). The IDs are database IDs, not filenames. This repository's split was selected for its dataset; when replacing or rebuilding the dataset, verify the IDs refer to the intended eight most recent matches and update the constant consistently before feature extraction, tuning, training, and evaluation. The pipeline fails if any configured test ID is missing.

## Run the training pipeline

Install the full pipeline dependencies in the activated environment:

```bash
python -m pip install -r requirements-pipeline.txt
```

Create the raw input folders if needed, place the MP4s and ZIPs as described above, then run these commands in order from the project root. Defaults write all generated artifacts to the `data/` locations shown below.

```bash
python -m preprocessing.extract_audio
python -m preprocessing.build_clip_database
python -m preprocessing.extract_features
python -m preprocessing.extract_labels
python -m modelling.tune_model
python -m modelling.save_final_model
python -m modelling.eval_model
```

`extract_audio.py` skips WAV files that already exist. The other stages should be rerun when their inputs or upstream outputs change. In particular, rerun feature extraction if the raw audio, split, YAMNet configuration, or PCA basis changes. Tuning performs randomized XGBoost search and grouped cross-validation, so it is the most compute-intensive step. It can be reduced for a quick iteration with `--n-iter` (the default is 60); use the full search for a final run.

The scripts accept paths and selected settings as command-line options. Run `python -m preprocessing.extract_features --help`, `python -m preprocessing.extract_labels --help`, `python -m modelling.tune_model --help`, `python -m modelling.save_final_model --help`, or `python -m modelling.eval_model --help` for the available overrides.

### Generated artefacts

| Output | Produced by | Used for |
| --- | --- | --- |
| `data/raw/audio/<match>.wav` | `extract_audio` | Match duration, YAMNet features, training |
| `data/metadata/clips_data.db`, `matches.csv`, `clips.csv` | `build_clip_database` | Match/clip metadata, duration denominator, labels |
| `data/processed/features/096_048/<match>.parquet`, `_meta.csv` | `extract_features` | Per-frame model inputs |
| `data/processed/features/pca/pca_transform.npz` | `extract_features` | Reuse the training-only PCA projection for new/demo audio |
| `data/processed/labels/labels.csv` | `extract_labels` | Refined training labels and original clip bounds for evaluation |
| `data/modelling/parameters/model_search_results.csv` | `tune_model` | Ranked model parameters and window settings |
| `data/modelling/final_model/model.ubj`, `model.json` | `save_final_model` | Inference model and required configuration sidecar |
| `data/modelling/performance/recall_budget_curve_test.png` | `eval_model` | Held-out test performance plot |

## Pipeline stages and scripts

### 1. Extract match audio — `preprocessing/extract_audio.py`

Reads every MP4 in `data/raw/video/` and writes one mono, 22.05 kHz, 16-bit PCM WAV with the same filename stem to `data/raw/audio/`. FFmpeg comes from `imageio-ffmpeg`. Existing WAVs are skipped. YAMNet later resamples the audio to its required 16 kHz rate in memory.

### 2. Pair footage and clips — `preprocessing/build_clip_database.py`

Matches each video and ZIP using normalized team names and date. It reads the goal clip timestamp from the member filename, probes clip duration, and records match and clip rows in SQLite. It also exports `matches.csv` and `clips.csv` next to the database under `data/metadata/`. Unmatched, ambiguous, malformed, or unreadable inputs are reported with `[FLAG]` messages. `matches.csv` includes `audio_length_sec`, so run audio extraction first; missing WAVs leave that duration empty.

### 3. Extract features — `preprocessing/extract_features.py`

Runs the vendored YAMNet SavedModel over each match WAV in bounded chunks. Audio is resampled to 16 kHz; YAMNet emits 0.96-second frames every 0.48 seconds. Each output row represents one frame and stores its `start_sec`, the 11 selected AudioSet scores (including cheering, applause, crowd, whistling, chatter, and shouting), and 16 PCA components derived from YAMNet's 1024-dimensional embedding.

Incremental PCA is fit using training matches only, excluding `TEST_MATCH_IDS`, and then transforms every match. The learned components and mean are saved in `pca_transform.npz` for inference, preventing test-match information from entering the feature projection. Results are stored as one Parquet file per match, plus `_meta.csv`.

### 4. Refine labels — `preprocessing/extract_labels.py`

Uses each goal clip's timestamp and duration to find its span in the match feature file. It expands that span by five seconds on either side, sums the YAMNet Cheering, Crowd, and Applause scores, and locates the strongest frame. A three-second interval centered on that peak (clipped to the available audio) becomes the positive training label. This targets the acoustic crowd response instead of treating the whole editor's cut as positive. The output CSV retains both refined bounds and the original clip bounds, plus peak time and score for inspection. Refined bounds train the model; original clip bounds are used to measure whether a candidate fully contains each edited goal clip.

### 5. Tune the classifier — `modelling/tune_model.py`

Runs a randomized search over XGBoost hyperparameters (60 candidates by default). Each candidate is evaluated with grouped cross-validation, keeping all frames from a match in the same fold. Out-of-fold predictions are pooled, then candidate windows are swept over combinations of `lookback`, `postroll`, and `merge_gap` without retraining the classifier. Candidates are ranked by mean recall across the 25–40% footage-budget band (partial recall-budget AUC), rather than by one arbitrary threshold. The eight fixed test matches are excluded. Ranked settings are written to `model_search_results.csv`; subsequent training uses the top row.

### 6. Select threshold and save model — `modelling/save_final_model.py`

Loads the best hyperparameters and candidate-window settings from the search results unless overridden. It creates training-only out-of-fold predictions and chooses the decision threshold whose candidate footage is closest to the requested budget (33% by default), breaking ties by recall. It then fits XGBoost on all non-test matches and saves `model.ubj` and the adjacent `model.json`. The test set is excluded from both threshold selection and final fitting. The JSON stores the threshold, target budget, feature columns, YAMNet frame settings, candidate-window settings, and PCA transform required for new audio.

### 7. Evaluate held-out matches — `modelling/eval_model.py`

Loads the saved bundle and scores only the eight held-out matches. It measures goal recall using the original editor clip bounds and computes the candidate footage budget relative to raw audio duration. The script reports recall at several budgets and the partial recall-budget AUC, marks the training-selected operating threshold, and writes the test curve plot. Test results are for final assessment; do not use them to retune model settings.

### Supporting modules and app

- `preprocessing/functions.py` contains shared filename parsing, clip probing, audio/YAMNet extraction, Parquet construction, and PCA helpers.
- `modelling/functions.py` contains feature and label loading, target construction, candidate merging, grouped cross-validation, recall-budget metrics, model persistence, and threshold selection.
- `constants/constants.py` defines project paths, YAMNet and label settings, the fixed test match IDs, tuning ranges, and budget defaults.
- `src/crowdsense/views/dashboard.py` provides Quick and Full demo scoring. `src/crowdsense/demo_functions.py` runs inference, selects a threshold for approximately 30% candidate footage, and merges detections into candidate ranges. `app.py` registers the Streamlit pages.

## Prepare demo data (optional)

The committed feature cache in `demo/features/` supports Quick demo without raw demo media. To regenerate it or use Full demo, place the demonstration MP4 and its matching highlight ZIP under `demo/raw/video/` and `demo/raw/clips/`, then run:

```bash
python -m demo.prepare_demo
```

This extracts demo audio, writes demo-local clip metadata, rebases clip timestamps by the configured `DEMO_START_OFFSET_SEC` (18 minutes 4 seconds) to account for the demo media's trim, and extracts YAMNet features using the training PCA transform. If your demo footage has a different trim or timestamp origin, adjust that offset in `demo/prepare_demo.py` before preparing it. Raw demo media is ignored by Git; generated demo feature files are stored in `demo/features/`.

## Notes

- This is a proof of concept for audio-based candidate detection, not automatic goal verification or a full video editing/export workflow. A human should review candidates.
- The dataset and evaluation use a fixed split with eight held-out matches. Performance may differ across teams, venues, microphones, and recording conditions.
- Goal labels depend on the strongest Cheering/Crowd/Applause peak being near the goal; a quiet goal or another loud reaction can produce a weak or misplaced label.
- `archive/` contains a retired cloud upload workflow and is not used by the current local pipeline or app.
- No project-specific license file is currently included. Add a license before publishing if you intend to distribute the project.
