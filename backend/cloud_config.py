"""Config for the cloud upload -> Lambda extraction -> app scoring flow."""

import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

_PLACEHOLDER_AWS_VALUES = {
    "AWS_ACCESS_KEY_ID": {"your-access-key-id", "replace-me", "changeme"},
    "AWS_SECRET_ACCESS_KEY": {"your-secret-access-key", "replace-me", "changeme"},
}


def _is_placeholder_aws_value(key: str, value: str | None) -> bool:
    if value is None:
        return True
    value = str(value).strip().strip("\"'")
    if not value:
        return True
    return value.lower() in _PLACEHOLDER_AWS_VALUES.get(key, set())


def _has_real_aws_credentials() -> bool:
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    return bool(access_key) and bool(secret_key) and not (
        _is_placeholder_aws_value("AWS_ACCESS_KEY_ID", access_key)
        or _is_placeholder_aws_value("AWS_SECRET_ACCESS_KEY", secret_key)
    )


def _store_env(key: str, value: str | None) -> None:
    if value is None:
        return
    value = str(value).strip().strip("\"'")
    if not value or _is_placeholder_aws_value(key, value):
        return
    current = os.environ.get(key)
    if current is None or _is_placeholder_aws_value(key, current):
        os.environ[key] = value


def _load_local_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        _store_env(key.strip(), value.strip())


def _load_streamlit_secrets() -> None:
    try:
        secrets = st.secrets
    except StreamlitSecretNotFoundError:
        return

    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "CROWDSENSE_UPLOAD_BUCKET",
    ):
        value = secrets.get(key)
        _store_env(key, value)


_load_local_env()
_load_streamlit_secrets()

AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get(
    "AWS_DEFAULT_REGION", "eu-west-2"
)
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "")

# raw/<key>.mp4   <- browser uploads here directly (presigned URL)
# audio/<key>.wav <- Lambda writes the extracted audio here, then deletes raw/
UPLOAD_BUCKET = os.environ.get("CROWDSENSE_UPLOAD_BUCKET")
RAW_PREFIX = "raw/"
AUDIO_PREFIX = "audio/"


def get_upload_config() -> tuple[str, str, str] | None:
    if not UPLOAD_BUCKET:
        return None
    return UPLOAD_BUCKET, AWS_REGION, AWS_ENDPOINT_URL
