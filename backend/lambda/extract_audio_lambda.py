"""AWS Lambda handler: triggered by S3 ObjectCreated on raw/<key>.mp4,
extracts audio via ffmpeg, writes it to audio/<key>.wav, deletes the raw
video. Only ffmpeg runs here -- no TensorFlow/YAMNet -- which is what
makes Lambda a reasonable fit for this one step (it isn't for scoring).

Needs an ffmpeg binary available at FFMPEG_PATH (default /opt/bin/ffmpeg),
provided via a Lambda Layer -- there's no apt/yum in the Lambda runtime,
and imageio_ffmpeg's download-on-import behaviour doesn't work against
Lambda's read-only filesystem outside /tmp. Several public ffmpeg-layer
projects exist, or you can build one from a static ffmpeg binary
(e.g. johnvansickle.com/ffmpeg's static builds) zipped under bin/ffmpeg.

Also needs ephemeral storage (/tmp) sized above your largest raw video --
the mp4 and the resulting wav both land there temporarily. Set this under
the function's "Ephemeral storage" config (up to 10,240 MB). Memory should
scale with file size too, since Lambda's CPU allocation is tied to it.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import boto3

s3 = boto3.client("s3")

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "/opt/bin/ffmpeg")
AUDIO_PREFIX = "audio/"


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    command = [
        FFMPEG_PATH, "-y", "-i", str(video_path), "-vn",
        "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", str(audio_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def handler(event, context):
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]  # e.g. raw/20260924-team-a-vs-b-abc123.mp4
        stem = Path(key).stem
        audio_key = f"{AUDIO_PREFIX}{stem}.wav"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            video_path = tmp_dir / Path(key).name
            audio_path = tmp_dir / f"{stem}.wav"

            print(f"[{stem}] downloading raw footage")
            s3.download_file(bucket, key, str(video_path))

            print(f"[{stem}] extracting audio")
            _extract_audio(video_path, audio_path)

            print(f"[{stem}] uploading extracted audio -> {audio_key}")
            s3.upload_file(str(audio_path), bucket, audio_key)
            # video_path/audio_path removed automatically on block exit.

    return {"statusCode": 200}
