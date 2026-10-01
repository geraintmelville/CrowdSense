import os

import backend.cloud_config as cloud_config


def test_placeholder_credentials_are_rejected(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "your-access-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "your-secret-access-key")

    assert cloud_config._is_placeholder_aws_value("AWS_ACCESS_KEY_ID", "your-access-key-id") is True
    assert cloud_config._is_placeholder_aws_value("AWS_SECRET_ACCESS_KEY", "your-secret-access-key") is True
    assert cloud_config._has_real_aws_credentials() is False


def test_store_env_overrides_placeholder_values(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "your-access-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "your-secret-access-key")

    cloud_config._store_env("AWS_ACCESS_KEY_ID", "real-access-key")
    cloud_config._store_env("AWS_SECRET_ACCESS_KEY", "real-secret")

    assert os.environ["AWS_ACCESS_KEY_ID"] == "real-access-key"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "real-secret"
