"""Integration tests for /tts and OpenAI speech routes (stub engine)."""

from __future__ import annotations

import pytest


@pytest.fixture
def voice_file(client, tmp_path):
    path = tmp_path / "voices" / "stub_voice.wav"
    path.write_bytes(b"RIFF")
    return "stub_voice.wav"


def test_tts_returns_wav(client, voice_file):
    r = client.post(
        "/tts",
        json={
            "text": "Hello there",
            "voice_mode": "predefined",
            "predefined_voice_id": voice_file,
            "output_format": "wav",
            "split_text": False,
        },
    )
    assert r.status_code == 200
    assert "audio" in r.headers.get("content-type", "")
    assert len(r.content) > 100


def test_tts_rejects_voice_path_traversal(client):
    r = client.post(
        "/tts",
        json={
            "text": "Hello there",
            "voice_mode": "predefined",
            "predefined_voice_id": "../config.yaml",
            "output_format": "wav",
            "split_text": False,
        },
    )
    assert r.status_code == 400


def test_tts_rejects_oversized_text(client, monkeypatch):
    monkeypatch.setattr("utils.config_manager.get_int", lambda key, default=0: 5)
    r = client.post(
        "/tts",
        json={
            "text": "too long",
            "voice_mode": "predefined",
            "predefined_voice_id": "missing.wav",
            "output_format": "wav",
            "split_text": False,
        },
    )
    assert r.status_code == 413


def test_tts_rejects_stream_wav(client, voice_file):
    r = client.post(
        "/tts",
        json={
            "text": "Hello",
            "voice_mode": "predefined",
            "predefined_voice_id": voice_file,
            "output_format": "wav",
            "split_text": False,
            "stream": True,
        },
    )
    assert r.status_code == 400


def test_tts_stream_opus_contract(client, voice_file, monkeypatch):
    import server_routes.tts_routes as tts_routes

    async def fake_stream(**_kwargs):
        yield b"ogg-data"

    monkeypatch.setattr(tts_routes.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(tts_routes, "_stream_encoded_audio_from_pcm", fake_stream)

    r = client.post(
        "/tts",
        json={
            "text": "Hello streamed audio",
            "voice_mode": "predefined",
            "predefined_voice_id": voice_file,
            "output_format": "opus",
            "split_text": False,
            "stream": True,
        },
    )

    assert r.status_code == 200
    assert "audio/ogg" in r.headers.get("content-type", "")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"
    assert "attachment; filename=" in r.headers["content-disposition"]
    assert r.content == b"ogg-data"


def test_tts_stream_requires_ffmpeg(client, voice_file, monkeypatch):
    import server_routes.tts_routes as tts_routes

    monkeypatch.setattr(tts_routes.shutil, "which", lambda _name: None)

    r = client.post(
        "/tts",
        json={
            "text": "Hello streamed audio",
            "voice_mode": "predefined",
            "predefined_voice_id": voice_file,
            "output_format": "mp3",
            "split_text": False,
            "stream": True,
        },
    )

    assert r.status_code == 503


def test_openai_speech_wav(client, voice_file):
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hello OpenAI",
            "voice": voice_file,
            "response_format": "wav",
        },
    )
    assert r.status_code == 200
    assert len(r.content) > 100


def test_openai_speech_rejects_voice_path_traversal(client):
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hello OpenAI",
            "voice": "../config.yaml",
            "response_format": "wav",
        },
    )
    assert r.status_code == 400


def test_openai_speech_validates_reference_audio(client, tmp_path, monkeypatch):
    ref = tmp_path / "reference_audio" / "too_long.wav"
    ref.write_bytes(b"RIFF")
    monkeypatch.setattr(
        "server_routes.openai_routes.utils.validate_reference_audio",
        lambda *_args, **_kwargs: (False, "too long"),
    )

    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hello OpenAI",
            "voice": "too_long.wav",
            "response_format": "wav",
        },
    )
    assert r.status_code == 400
    assert "Invalid reference audio" in r.json()["detail"]


def test_openai_speech_rejects_oversized_input(client, monkeypatch):
    monkeypatch.setattr("utils.config_manager.get_int", lambda key, default=0: 5)
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "too long",
            "voice": "missing.wav",
            "response_format": "wav",
        },
    )
    assert r.status_code == 413


def test_openai_speech_sse_pcm_contract(client, voice_file):
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hello OpenAI",
            "voice": voice_file,
            "response_format": "pcm",
            "stream_format": "sse",
        },
    )

    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-sample-rate"] == "24000"
    assert b"speech.audio.delta" in r.content
    assert b"speech.audio.done" in r.content


def test_openai_auto_streams_multichunk_compressed_audio(
    client, voice_file, monkeypatch
):
    import server_routes.openai_routes as openai_routes

    async def fake_stream(**_kwargs):
        yield b"mp3-data"

    monkeypatch.setattr(openai_routes.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(openai_routes, "_stream_encoded_audio_from_pcm", fake_stream)

    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hello. " * 120,
            "voice": voice_file,
            "response_format": "mp3",
        },
    )

    assert r.status_code == 200
    assert "audio/mpeg" in r.headers.get("content-type", "")
    assert r.headers["x-stream-audio-codec"] == "mp3"
    assert r.headers["x-stream-container"] == "mp3"
    assert r.content == b"mp3-data"


def test_openai_speech_voice_missing(client):
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "Hi",
            "voice": "nonexistent.wav",
            "response_format": "wav",
        },
    )
    assert r.status_code == 404


def test_ui_initial_data_redacts_secrets(client):
    r = client.get("/api/ui/initial-data")
    assert r.status_code == 200
    config = r.json()["config"]
    assert config["server"]["auth_password"] == ""
    assert config["asr"]["access_token"] == ""
