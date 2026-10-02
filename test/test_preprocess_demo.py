import csv
import sqlite3

import numpy as np
import pandas as pd

from demo import prepare_demo
from preprocessing.build_clip_database import process_all
from preprocessing.functions import build_feature_dataframe


def test_prepare_demo_uses_raw_directories_and_writes_features(tmp_path, monkeypatch):
    demo_dir = tmp_path / "demo"
    raw_dir = demo_dir / "raw"
    video_dir = raw_dir / "video"
    audio_dir = raw_dir / "audio"
    clips_dir = raw_dir / "clips"
    features_dir = demo_dir / "features"
    database_path = demo_dir / "clips_data.db"
    pca_path = tmp_path / "pca_transform.npz"
    np.savez(pca_path, components=np.eye(2, dtype=np.float32), mean=np.zeros(2, dtype=np.float32))
    calls = []

    monkeypatch.setattr(prepare_demo, "DEMO_DIR", demo_dir)
    monkeypatch.setattr(prepare_demo, "DEMO_RAW_DIR", raw_dir)
    monkeypatch.setattr(prepare_demo, "DEMO_RAW_VIDEO_DIR", video_dir)
    monkeypatch.setattr(prepare_demo, "DEMO_RAW_AUDIO_DIR", audio_dir)
    monkeypatch.setattr(prepare_demo, "DEMO_RAW_CLIPS_DIR", clips_dir)
    monkeypatch.setattr(prepare_demo, "DEMO_FEATURES_DIR", features_dir)
    monkeypatch.setattr(prepare_demo, "DEMO_DATABASE_PATH", database_path)
    monkeypatch.setattr(prepare_demo, "DEMO_MATCHES_PATH", demo_dir / "matches.csv")
    monkeypatch.setattr(prepare_demo, "DEMO_CLIPS_PATH", demo_dir / "clips.csv")
    monkeypatch.setattr(prepare_demo, "DEMO_PCA_PATH", pca_path)
    monkeypatch.setattr(
        prepare_demo,
        "extract_audio",
        lambda source_dir, output_dir: calls.append(("audio", source_dir, output_dir)),
    )
    monkeypatch.setattr(
        prepare_demo,
        "process_all",
        lambda **kwargs: calls.append(("database", kwargs)),
    )
    monkeypatch.setattr(prepare_demo, "list_matches", lambda path: [(4, "match.mp4")])
    monkeypatch.setattr(prepare_demo, "load_yamnet_model", lambda: "yamnet")
    monkeypatch.setattr(
        prepare_demo,
        "extract_yamnet_match",
        lambda filename, directory, model: (
            np.array([0.0]), np.zeros((1, len(prepare_demo.SCORE_INDICES))), np.ones((1, 2)),
        ),
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda frame, path, index: calls.append(("features", path, index)),
    )

    prepare_demo.prepare_demo()

    assert calls == [
        ("audio", video_dir, audio_dir),
        (
            "database",
            {
                "raw_dir": video_dir,
                "clips_dir": clips_dir,
                "audio_dir": audio_dir,
                "db_path": database_path,
                "matches_csv_path": demo_dir / "matches.csv",
                "clips_csv_path": demo_dir / "clips.csv",
            },
        ),
        ("features", features_dir / "match.parquet", False),
    ]
    assert all(path.is_dir() for path in (raw_dir, video_dir, audio_dir, clips_dir, features_dir))


def test_build_feature_dataframe_uses_production_columns_and_saved_pca():
    score_rows = np.arange(len(prepare_demo.SCORE_INDICES), dtype=np.float32).reshape(1, -1)
    embeddings = np.array([[2.0, 4.0]], dtype=np.float32)
    components = np.array([[1.0, 0.0]], dtype=np.float32)
    mean = np.array([[1.0, 1.0]], dtype=np.float32)

    features = build_feature_dataframe(
        4, "match.mp4", np.array([0.0]), score_rows, embeddings, components, mean
    )

    assert features.loc[0, "match_id"] == 4
    assert features.loc[0, "raw_filename"] == "match.mp4"
    assert features.loc[0, "start_sec"] == 0.0
    assert features.loc[0, "yamnet_score_006"] == score_rows[0, 0]
    assert features.loc[0, "yamnet_embedding_pca_00"] == 1.0


def test_process_all_exports_matches_and_clips_from_database(tmp_path):
    database_path = tmp_path / "clips_data.db"
    raw_dir = tmp_path / "raw"
    clips_dir = tmp_path / "zips"
    audio_dir = tmp_path / "audio"
    raw_dir.mkdir()
    clips_dir.mkdir()
    audio_dir.mkdir()
    with sqlite3.connect(database_path) as conn:
        conn.executescript(
            """
            CREATE TABLE matches (
                match_id INTEGER PRIMARY KEY,
                raw_filename TEXT,
                zip_filename TEXT,
                match_date TEXT,
                length_sec REAL
            );
            CREATE TABLE clips (
                match_id INTEGER,
                clip_number INTEGER,
                timestamp_formatted TEXT,
                description TEXT,
                filename TEXT,
                length_sec REAL
            );
            INSERT INTO matches VALUES (7, 'match.mp4', 'highlights.zip', '2026-04-19', 123.5);
            INSERT INTO clips VALUES (7, 2, '00:01:00', 'Goal', '02_goal.mp4', 12.0);
            """
        )

    process_all(raw_dir, clips_dir, audio_dir, database_path)

    matches_path = tmp_path / "matches.csv"
    clips_path = tmp_path / "clips.csv"

    with matches_path.open(newline="", encoding="utf-8") as csv_file:
        matches = list(csv.DictReader(csv_file))
    with clips_path.open(newline="", encoding="utf-8") as csv_file:
        clips = list(csv.DictReader(csv_file))

    assert matches == [{
        "match_id": "7",
        "raw_filename": "match.mp4",
        "zip_filename": "highlights.zip",
        "match_date": "2026-04-19",
        "audio_length_sec": "123.5",
    }]
    assert clips == [{
        "clip_id": "1",
        "match_id": "7",
        "clip_number": "2",
        "timestamp_formatted": "00:01:00",
        "description": "Goal",
        "filename": "02_goal.mp4",
        "length_sec": "12.0",
    }]