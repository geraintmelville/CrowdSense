"""Presigned-POST upload so raw match footage goes browser -> S3 directly,
never through the Streamlit server's memory. Streamlit itself can't issue
a raw HTML <form> upload (see project constraints), so this is meant to be
used from a small static/JS upload widget embedded via st.components, or
from a separate lightweight upload page -- not Streamlit's file_uploader.
"""

import re
import uuid
from datetime import datetime

import boto3

from backend.cloud_config import AWS_REGION, RAW_PREFIX, UPLOAD_BUCKET

_s3 = boto3.client("s3", region_name=AWS_REGION)

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


def make_match_key(original_filename: str) -> str:
    """A storage key that's stable and traceable back to the original
    filename (reuses the same opposition-date naming convention as the
    existing raw filenames) plus a short uuid to avoid collisions."""
    stem = _SAFE_NAME.sub("-", original_filename.rsplit(".", 1)[0])
    date_prefix = datetime.utcnow().strftime("%Y%m%d")
    return f"{RAW_PREFIX}{date_prefix}-{stem}-{uuid.uuid4().hex[:8]}.mp4"


def presigned_upload(original_filename: str, max_bytes: int = 3 * 1024 * 1024 * 1024) -> dict:
    """Returns {url, fields, key} for a presigned POST. The browser POSTs
    the file directly to `url` with `fields` plus the file, and the object
    lands at `key` in UPLOAD_BUCKET -- no server-side handling of the bytes.
    """
    if not UPLOAD_BUCKET:
        raise RuntimeError("CROWDSENSE_UPLOAD_BUCKET is not configured")
    if boto3.Session().get_credentials() is None:
        raise RuntimeError(
            "AWS credentials are not configured. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY, or configure an AWS profile."
        )
    key = make_match_key(original_filename)
    presigned = _s3.generate_presigned_post(
        Bucket=UPLOAD_BUCKET,
        Key=key,
        Conditions=[["content-length-range", 0, max_bytes]],
        ExpiresIn=3600,
    )
    return {"url": presigned["url"], "fields": presigned["fields"], "key": key}
