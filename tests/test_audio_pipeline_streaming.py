from __future__ import annotations

import asyncio

import pytest

from audio_pipeline import _get_ffmpeg_stream_command, _stream_encoded_audio_from_pcm


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


def test_stream_encoded_audio_writer_cleanup_does_not_reference_removed_stop_event(
    monkeypatch,
):
    class FakeStdin:
        def __init__(self):
            self.closed = False

        def write(self, _data):
            return None

        async def drain(self):
            return None

        def close(self):
            self.closed = True

    class FakeReader:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        async def read(self, _size):
            if self.chunks:
                return self.chunks.pop(0)
            return b""

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = FakeReader([b"encoded-audio"])
            self.stderr = FakeReader([])
            self.returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return FakeProcess()

    async def scenario():
        chunks = []
        async for item in _stream_encoded_audio_from_pcm(
            text_chunks=["hello"],
            target_sample_rate=24000,
            output_format="mp3",
            sse=False,
            log_prefix="test",
            locked_synthesis={
                "audio_prompt_path": None,
                "temperature": 0.8,
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "seed": 0,
                "language": "en",
                "speed_factor": 1.0,
            },
        ):
            chunks.append(item)
        return chunks

    monkeypatch.setattr("audio_pipeline.shutil.which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "audio_pipeline.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    assert asyncio.run(scenario()) == [b"encoded-audio"]
