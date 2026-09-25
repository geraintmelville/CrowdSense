# CrowdSense

CrowdSense is an audio-driven futsal highlight detection project that helps reviewers find likely goal moments in long match footage using crowd noise and match audio cues.

The app combines:

- audio extraction from raw video
- YAMNet-based audio feature extraction
- PCA dimensionality reduction
- XGBoost classification
- candidate-window merging and review in a Streamlit dashboard

This is designed to reduce the amount of footage a human has to watch when building highlight reels from full-length matches.

## Project overview

The application is built around a Streamlit UI in `app.py`, with project pages under `views/` for overview, dashboard, architecture, dataset information, and the model pipeline.

The workflow is:

1. Raw video or audio is uploaded into the system.
2. Audio is extracted or loaded for scoring.
3. Audio windows are transformed into features using YAMNet embeddings.
4. PCA reduces the embedding space.
5. A tuned XGBoost model scores each window for likely goal activity.
6. Nearby high-probability windows are merged into candidate clips.
7. Reviewers inspect the resulting clips in the dashboard.

## Key features

- Local audio upload support for direct analysis
- Cloud upload flow with S3 + Lambda audio extraction
- Streamlit dashboard for candidate review
- Model bundle with trained classifier and scoring parameters
- Data preparation scripts for labels, clips, and feature generation

## Repository structure

```text
.
├── app.py                     # Main Streamlit app entry point
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package metadata
├── backend/
│   ├── cloud_config.py       # AWS config and environment loading
│   ├── upload_helper.py      # Presigned upload helpers
│   └── lambda/
│       └── extract_audio_lambda.py
├── constants/
│   └── constants.py
├── data/
│   ├── metadata/
│   ├── modelling/
│   ├── processed/
│   └── raw/
├── modelling/
│   ├── eval_model.py
│   ├── functions.py
│   ├── save_final_model.py
│   ├── train_predict.py
│   └── tune_model.py
├── preprocessing/
│   ├── build_clip_database.py
│   ├── extract_audio.py
│   ├── extract_features.py
│   ├── extract_labels.py
│   └── functions.py
├── views/
│   ├── architecture.py
│   ├── dashboard.py
│   ├── dataset_info.py
│   ├── model_pipeline.py
│   └── overview.py
└── tree_diagram.txt
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the app

From the project root:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

The app launches a multi-page Streamlit interface with:

- Overview / Holistic Project Summary
- Live Highlight Finder Dashboard
- System Architecture / Cloud Infrastructure
- Model & YAMNet Pipeline
- Dataset & Ingestion

## Cloud upload configuration

The project supports a cloud-based workflow for uploading raw match video and having audio extracted automatically via AWS.

The app reads configuration from environment variables and Streamlit secrets. Relevant settings include:

- `AWS_REGION` or `AWS_DEFAULT_REGION`
- `CROWDSENSE_UPLOAD_BUCKET`
- optional `AWS_ACCESS_KEY_ID`
- optional `AWS_SECRET_ACCESS_KEY`

A local `.env` file can be used for local development, and the dashboard also supports configuration loaded from `st.secrets`.

## Data and model assets

The project expects data in the `data/` directory, including:

- raw match audio and video files
- metadata for clips and matches
- processed features and labels
- model artifacts in `data/modelling/final_model/`

The published model bundle is used by the dashboard to score new footage without retraining on the fly.

## Typical workflow

### Local analysis

- upload a WAV file through the dashboard
- the system scores the audio windows
- candidate clips are merged and displayed

### Cloud ingestion flow

- upload raw MP4 to S3 using a presigned URL
- an S3-triggered Lambda extracts the audio
- the app polls for the processed audio file
- scoring proceeds using the same model pipeline

## Notes

This project is a research/prototype workflow rather than a production-grade deployment system. It is intended to demonstrate a complete ML and cloud-assisted highlight-detection pipeline in a single codebase.

## License

This project does not currently include a project-specific license file. If you are publishing or sharing this repo publicly, add a license that matches your intended usage and distribution requirements.
