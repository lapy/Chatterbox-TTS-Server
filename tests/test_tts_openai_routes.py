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
