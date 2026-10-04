import os
import subprocess
from pathlib import Path

import imageio_ffmpeg

from constants.constants import RAW_AUDIO_DIR, RAW_VIDEO_DIR

def create_demo_clip(input_video_path, output_video_path, start_seconds, duration_seconds):
    """Create a 720p H.264 demo segment from a full match video."""
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    if not input_video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {input_video_path}")
    if start_seconds < 0 or duration_seconds <= 0:
        raise ValueError("Clip start must be non-negative and duration must be positive")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(input_video_path),
        "-t",
        str(duration_seconds),
        "-vf",
        "scale=-2:720",
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_video_path),
    ]
    _run_ffmpeg(command)
    return output_video_path


def extract_audio_file(input_video_path, output_audio_path):
    """Extract one video to mono 22.05 kHz PCM WAV."""
    input_video_path = Path(input_video_path)
    output_audio_path = Path(output_audio_path)
    if not input_video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {input_video_path}")

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(input_video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "22050",
        "-ac",
        "1",
        str(output_audio_path),
    ]
    _run_ffmpeg(command)
    return output_audio_path


def _run_ffmpeg(command):
    """Run ffmpeg quietly on success and include its diagnostics on failure."""
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, end="" if error.stderr.endswith("\n") else "\n")
        raise


def extract_audio(input_dir, output_dir):
    """
    Extracts audio from all MP4 files in input_dir and saves them as WAV files in output_dir.
    Skips extraction if the output WAV file already exists.
    """
    # Ensure target directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all MP4 files in the target directory
    mp4_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        print(f"No .mp4 files found in {input_dir}")
        return

    print(f"Found {len(mp4_files)} MP4 file(s). Starting extraction...\n")

    for filename in mp4_files:
        base_name = os.path.splitext(filename)[0]
        mp4_path = os.path.join(input_dir, filename)
        wav_path = os.path.join(output_dir, f"{base_name}.wav")

        # Skip logic if output file already exists
        if os.path.exists(wav_path):
            print(f"[SKIP] Already converted: {base_name}.wav")
            continue

        print(f"[PROCESSING] Extracting audio from {filename}...")

        try:
            extract_audio_file(mp4_path, wav_path)
            print(f"[SUCCESS] Saved to {wav_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to convert {filename}: {e}")

if __name__ == "__main__":
    extract_audio(RAW_VIDEO_DIR, RAW_AUDIO_DIR)
