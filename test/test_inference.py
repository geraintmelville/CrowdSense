import numpy as np
import pandas as pd

from crowdsense import demo_functions as inference


class StubClassifier:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)
        self.features = None

    def predict_proba(self, features):
        self.features = features
        return np.column_stack((1 - self.probabilities, self.probabilities))


def test_score_match_builds_clips_from_injected_features(monkeypatch):
    scores = np.asarray([[0.1, 0.8], [0.2, 0.9], [0.3, 0.1]])
    classifier = StubClassifier([0.2, 0.9, 0.1])
    calls = {}

    def extract_features(audio_path, include_embeddings, model):
        calls.update(audio_path=audio_path, include_embeddings=include_embeddings, model=model)
        return scores, None, 2.0

    monkeypatch.setattr(inference, "extract_yamnet_features_streaming", extract_features)
    bundle = {
        "feature_columns": ["yamnet_score_001"],
        "yamnet_score_indices": [1],
        "model": classifier,
        "threshold": 0.5,
        "window_sec": 0.96,
        "stride_sec": 0.48,
        "lookback": 0.2,
        "postroll": 0.1,
        "merge_gap": 0.0,
    }

    clips, duration, n_windows = inference.score_match("match.wav", bundle, yamnet_model="yamnet")

    assert calls == {"audio_path": "match.wav", "include_embeddings": False, "model": "yamnet"}
    np.testing.assert_allclose(classifier.features[:, 0], scores[:, 1])
    np.testing.assert_allclose(clips[["start_sec", "end_sec"]].to_numpy(), [[0.28, 1.54]])
    assert clips[["start", "end"]].to_numpy().tolist() == [["00:00:00", "00:00:01"]]
    assert (duration, n_windows) == (2.0, 3)


def test_score_precomputed_features_uses_bundle_column_order():
    classifier = StubClassifier([0.9, 0.1])
    features = pd.DataFrame({
        "start_sec": [0.48, 0.0],
        "first": [2.0, 1.0],
        "second": [20.0, 10.0],
    })
    bundle = {
        "feature_columns": ["second", "first"],
        "model": classifier,
        "threshold": 0.5,
        "window_sec": 0.96,
        "lookback": 0.2,
        "postroll": 0.1,
        "merge_gap": 0.0,
    }

    clips, duration, n_windows = inference.score_precomputed_features(features, bundle, 1.44)

    np.testing.assert_allclose(classifier.features, [[10.0, 1.0], [20.0, 2.0]])
    np.testing.assert_allclose(clips[["start_sec", "end_sec"]].to_numpy(), [[0.0, 1.06]])
    assert (duration, n_windows) == (1.44, 2)
