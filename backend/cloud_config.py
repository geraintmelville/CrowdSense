"""Config for the cloud upload -> Lambda extraction -> app scoring flow."""

import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _load_local_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


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
        if value:
            os.environ.setdefault(key, str(value))


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
