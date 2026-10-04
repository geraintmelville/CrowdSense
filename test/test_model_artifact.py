import numpy as np
from xgboost import XGBClassifier

from modelling.functions import load_model_artifact, save_model_artifact


def test_native_model_round_trip_preserves_predictions_and_metadata(tmp_path):
    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1])
    model = XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1, random_state=0)
    model.fit(features, labels)
    expected = model.predict_proba(features)
    metadata = {"threshold": 0.42, "feature_columns": ["left", "right"]}
    model_path = tmp_path / "model.ubj"

    save_model_artifact(model, metadata, model_path)
    loaded = load_model_artifact(model_path)

    np.testing.assert_allclose(loaded["model"].predict_proba(features), expected)
    assert loaded["threshold"] == metadata["threshold"]
    assert loaded["feature_columns"] == metadata["feature_columns"]
