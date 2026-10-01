from pathlib import Path
import sys

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.upload_helper import presigned_upload

filename = Path(__file__).resolve().with_name("test.mp4")
if not filename.is_file():
    raise FileNotFoundError(f"Upload fixture not found: {filename}")

presigned = presigned_upload(filename.name)

print(f"Uploading {filename.name} ({filename.stat().st_size} bytes) to S3")
try:
    with filename.open("rb") as f:
        response = requests.post(
            presigned["url"],
            data=presigned["fields"],
            files={
                "file": (
                    filename.name,
                    f,
                    "video/mp4",
                )
            },
            timeout=(15, 120),
        )
except requests.exceptions.RequestException as error:
    print(f"Upload failed before receiving an HTTP response ({type(error).__name__}): {error}")
    raise

print("\nSTATUS:")
print(response.status_code)

print("\nRESPONSE:")
print(response.text)

print("\nHEADERS:")
print(dict(response.headers))