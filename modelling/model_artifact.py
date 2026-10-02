"""Native XGBoost model persistence with JSON inference metadata."""

import json
from pathlib import Path

from xgboost import XGBClassifier


def save_model_artifact(model: XGBClassifier, metadata: dict, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    config_path = model_path.with_suffix(".json")
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(metadata, config_file, indent=2)


def load_model_artifact(model_path: str | Path) -> dict:
    model_path = Path(model_path)
    model = XGBClassifier()
    model.load_model(model_path)
    config_path = model_path.with_suffix(".json")
    with config_path.open(encoding="utf-8") as config_file:
        metadata = json.load(config_file)
    return {"model": model, **metadata}