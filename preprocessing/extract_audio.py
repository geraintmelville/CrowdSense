import os
import subprocess
import imageio_ffmpeg

from constants.constants import RAW_AUDIO_DIR, RAW_VIDEO_DIR

def extract_audio(input_dir, output_dir):
    """
    Extracts audio from all MP4 files in input_dir and saves them as WAV files in output_dir.
    Skips extraction if the output WAV file already exists.
    """
    # Ensure target directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

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

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            mp4_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "22050",
            "-ac",
            "1",
            wav_path,
        ]

        try:
            # Suppress standard output to keep console logs clean
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[SUCCESS] Saved to {wav_path}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to convert {filename}: {e}")

if __name__ == "__main__":
    extract_audio(RAW_VIDEO_DIR, RAW_AUDIO_DIR)