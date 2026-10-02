from preprocessing import extract_audio


def test_create_demo_clip_builds_720p_h264_command(tmp_path, monkeypatch):
    source = tmp_path / "full-match.mp4"
    output = tmp_path / "generated" / "demo.mp4"
    source.touch()
    monkeypatch.setattr(extract_audio.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

    monkeypatch.setattr(extract_audio.subprocess, "run", fake_run)
    result = extract_audio.create_demo_clip(source, output, 600, 1500)

    assert result == output
    command = captured["command"]
    assert command[command.index("-ss") + 1] == "600"
    assert command[command.index("-t") + 1] == "1500"
    assert command[command.index("-vf") + 1] == "scale=-2:720"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "28"


def test_extract_audio_file_builds_mono_22050_hz_command(tmp_path, monkeypatch):
    source = tmp_path / "demo.mp4"
    output = tmp_path / "generated" / "demo.wav"
    source.touch()
    monkeypatch.setattr(extract_audio.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

    monkeypatch.setattr(extract_audio.subprocess, "run", fake_run)
    result = extract_audio.extract_audio_file(source, output)

    assert result == output
    command = captured["command"]
    assert command[command.index("-ar") + 1] == "22050"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-acodec") + 1] == "pcm_s16le"