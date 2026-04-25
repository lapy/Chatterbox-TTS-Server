from __future__ import annotations

import pytest

from audio_pipeline import _get_ffmpeg_stream_command


def test_ffmpeg_opus_stream_command_uses_standard_audio_profile():
    cmd = _get_ffmpeg_stream_command(
        "ffmpeg", input_sample_rate=24000, output_format="opus"
    )

    assert cmd[:1] == ["ffmpeg"]
    assert ["-c:a", "libopus"] == cmd[cmd.index("-c:a") : cmd.index("-c:a") + 2]
    assert cmd[cmd.index("-application") + 1] == "audio"
    assert cmd[cmd.index("-f") + 1] == "s16le"
    assert cmd[-1] == "pipe:1"
    assert "-page_duration" in cmd


def test_ffmpeg_mp3_stream_command_disables_stream_unfriendly_headers():
    cmd = _get_ffmpeg_stream_command(
        "ffmpeg", input_sample_rate=24000, output_format="mp3"
    )

    assert "libmp3lame" in cmd
    assert cmd[cmd.index("-write_xing") + 1] == "0"
    assert cmd[cmd.index("-id3v2_version") + 1] == "0"
    assert cmd[-1] == "pipe:1"


def test_ffmpeg_stream_command_rejects_unsupported_format():
    with pytest.raises(ValueError):
        _get_ffmpeg_stream_command(
            "ffmpeg", input_sample_rate=24000, output_format="wav"
        )
