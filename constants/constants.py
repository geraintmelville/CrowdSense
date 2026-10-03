"""Single source of truth for active preprocessing and modelling configuration."""

from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parents[2].name == "src":
    ROOT_DIR = _THIS_FILE.parents[3]
else:
    ROOT_DIR = _THIS_FILE.parents[1]

# Repository data layout.
DATA_DIR = ROOT_DIR / "data"
DEMO_DIR = ROOT_DIR / "demo"
DEMO_RAW_DIR = DEMO_DIR / "raw"
DEMO_RAW_VIDEO_DIR = DEMO_RAW_DIR / "video"
DEMO_RAW_AUDIO_DIR = DEMO_RAW_DIR / "audio"
DEMO_RAW_CLIPS_DIR = DEMO_RAW_DIR / "clips"
DEMO_FEATURES_DIR = DEMO_DIR / "features"
DEMO_GENERATED_DIR = DEMO_DIR / "generated"
METADATA_DIR = DATA_DIR / "metadata"
RAW_DIR = DATA_DIR / "raw"
RAW_VIDEO_DIR = RAW_DIR / "video"
RAW_CLIPS_DIR = RAW_DIR / "clips"
RAW_AUDIO_DIR = RAW_DIR / "audio"
FEATURES_DIR = DATA_DIR / "processed" / "features" / "096_048"
PCA_DIR = DATA_DIR / "processed" / "features" / "pca"
LABELS_PATH = DATA_DIR / "processed" / "labels" / "labels.csv"
MATCHES_PATH = METADATA_DIR / "matches.csv"
CLIP_DATABASE_PATH = METADATA_DIR / "clips_data.db"
MODELLING_DIR = DATA_DIR / "modelling"
MODEL_RESULTS_PATH = MODELLING_DIR / "parameters" / "model_search_results.csv"
MODEL_PERFORMANCE_DIR = MODELLING_DIR / "performance"
RECALL_BUDGET_PLOT_PATH = MODEL_PERFORMANCE_DIR / "recall_budget_curve_test.png"
MODEL_PATH = MODELLING_DIR / "final_model" / "model.ubj"

# YAMNet feature extraction.
YAMNET_SAMPLE_RATE = 16_000
YAMNET_WINDOW_SEC = 0.96
YAMNET_STRIDE_SEC = 0.48
YAMNET_CHUNK_SEC = 28.8
YAMNET_LOOKAHEAD_SEC = YAMNET_WINDOW_SEC
PCA_COMPONENTS = 16
PCA_BATCH_SIZE = 4096

# AudioSet classes used by the active feature and label pipelines.
SCORE_INDICES = {
    6: "Shout",
    9: "Yell",
    11: "Screaming",
    35: "Whistling",
    61: "Cheering",
    62: "Applause",
    64: "Crowd",
    63: "Chatter",
    65: "Hubbub, speech noise, speech babble",
    58: "Clapping",
    10: "Children shouting",
}
PEAK_LABELS = ("Cheering", "Crowd", "Applause")

# The fixed test split is shared by extraction, tuning, training, and evaluation.
TEST_MATCH_IDS = (6, 22, 19, 13, 30, 26, 32, 25)

# Label refinement defaults.
LABEL_WINDOW_SEC = 3.0
SEARCH_MARGIN_SEC = 5.0
MIN_PEAK_SCORE = 0.0

# Candidate-window and evaluation defaults.
LOOKBACK_SEC = 50.0
POSTROLL_SEC = 8.0
MERGE_GAP_SEC = 5.0
BUDGET_CHECKPOINTS = (0.1, 0.2, 0.3, 0.4, 0.5)
MINIMUM_RECALL = 0.95
BUDGET_BAND = (0.25, 0.40)
TARGET_BUDGET = 0.33
DEPLOYMENT_BUDGET = TARGET_BUDGET
CURVE_N_THRESHOLDS = 400
FINAL_CURVE_N_THRESHOLDS = 5000

# Reproducible model-search defaults.
RANDOM_STATE = 42
N_ITER = 60
CV_FOLDS = 4
MODEL_N_JOBS = 4
MODEL_PARAM_DISTRIBUTIONS = {
    "n_estimators": [600, 900, 1200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.015, 0.025, 0.05],
    "min_child_weight": [10, 20, 30],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "scale_pos_weight": [4, 8, 12],
    "gamma": [0.25, 0.5, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [5, 10, 20],
}
CANDIDATE_PARAM_GRID = {
    "lookback": [10.0, 15.0, 20.0, 30.0],
    "postroll": [0.0, 1.0, 3.0, 5.0, 10.0],
    "merge_gap": [5.0, 10.0, 15.0],
}
RESULTS = {
    "recall": 0.79,
    "budget": 0.28,
    "baseline_recall": 0.61,
    "baseline_budget": 0.42,
}

__all__ = [
    name for name in globals()
    if not name.startswith("_")
]
