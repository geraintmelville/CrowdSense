# CrowdSense

CrowdSense is an audio-driven futsal highlight detection project that helps reviewers find likely goal moments in long match footage using crowd noise and match audio cues.

The app combines YAMNet audio features, PCA dimensionality reduction, XGBoost classification, and candidate-window merging.

## Live demo

1. Put the full-match MP4 in `demo/raw/video/` and prepare the demo with `python -m demo.prepare_demo`.
2. Run `streamlit run app.py` from the project root.
3. On the dashboard, choose Quick demo to score the prepared feature file or Full demo to select the matching WAV and extract features from the audio before scoring.

The dashboard uses [this demo match](https://youtu.be/W8DhX4CIdKM) for timestamped links and in-page playback by default. To use a different match, set `CROWDSENSE_DEMO_YOUTUBE_URL` in the environment. Candidate times are relative to the selected match audio, which should start at the beginning of that video.

To prepare a manually trimmed demo match and its highlight ZIP, place the MP4 in
`demo/raw/video/` and the matching ZIP in `demo/raw/clips/`, then run
`python -m demo.prepare_demo`. The script extracts audio into `demo/raw/audio/`, writes
`clips_data.db` and CSV metadata into `demo/`, and writes model-compatible feature files
into `demo/features/`. Raw media is ignored by Git; generated features are tracked.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the virtual environment with `.venv\Scripts\Activate.ps1` before installing dependencies and running the app.

The dashboard loads `data/modelling/final_model/model.ubj` and its adjacent `model.json` configuration to score demo audio without retraining.

## Archived workflow

The retired AWS upload, S3 configuration, Lambda extraction, and related tests are retained locally under `archive/cloud_upload/`. The entire `archive/` directory is ignored by Git and is not imported by the application.

## License

This project does not currently include a project-specific license file. If you are publishing or sharing this repo publicly, add a license that matches your intended usage and distribution requirements.
