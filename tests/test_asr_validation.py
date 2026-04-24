from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

import asr_validation


def test_normalize_text_for_asr_compare():
    assert asr_validation.normalize_text_for_asr_compare("Hello, World!") == "hello world"


def test_transcribe_openai_compatible_json(monkeypatch):
    class Resp:
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "  hi there  "}

    monkeypatch.setattr("asr_validation.requests.post", lambda *a, **kwargs: Resp())
    out = asr_validation.transcribe_openai_compatible(
        np.ones(800, dtype=np.float32) * 0.1,
        24000,
        base_url="https://example.com/v1",
        access_token="tok",
        model="whisper-1",
        timeout_sec=30.0,
        language="en",
    )
    assert out == "hi there"


def test_transcribe_raises_on_http_error(monkeypatch):
    class Resp:
        def raise_for_status(self):
            raise requests.HTTPError("bad")

    monkeypatch.setattr("asr_validation.requests.post", lambda *a, **kwargs: Resp())
    with pytest.raises(requests.HTTPError):
        asr_validation.transcribe_openai_compatible(
            np.ones(100, dtype=np.float32),
            16000,
            base_url="https://x/v1",
            access_token="t",
            model="m",
            timeout_sec=5.0,
        )


def test_transcription_mismatch_reason_disabled(monkeypatch):
    monkeypatch.setattr(
        "asr_validation.config_manager.get_bool", lambda k, d=None: False
    )
    assert (
        asr_validation.transcription_mismatch_reason(
            np.ones(1000, dtype=np.float32), 24000, "hello world"
        )
        is None
    )


def test_transcription_mismatch_reason_low_similarity(monkeypatch):
    class CM:
        def get_bool(self, key, default=None):
            if key == "asr.enabled":
                return True
            return bool(default)

        def get(self, key, default=None):
            return {
                "asr.openai_compatible_base_url": "https://ex/v1",
                "asr.access_token": "x",
                "asr.model": "whisper-1",
                "asr.language": "",
            }.get(key, default)

        def get_int(self, key, default=None):
            if key == "asr.skip_if_text_shorter_than":
                return 4
            return int(default or 0)

        def get_float(self, key, default=None):
            if key == "asr.min_similarity":
                return 0.95
            if key == "asr.timeout_sec":
                return 30.0
            return float(default or 0.0)

    monkeypatch.setattr(asr_validation, "config_manager", CM())
    monkeypatch.setattr(
        "asr_validation.transcribe_openai_compatible",
        lambda *a, **k: "something unrelated",
    )
    r = asr_validation.transcription_mismatch_reason(
        np.ones(2000, dtype=np.float32) * 0.1,
        24000,
        "The quick brown fox jumps.",
    )
    assert r is not None
    assert r.startswith("asr_low_similarity")


def test_transcription_mismatch_skips_short_text(monkeypatch):
    class CM:
        def get_bool(self, key, default=None):
            return key == "asr.enabled" or bool(default)

        def get(self, key, default=None):
            return {
                "asr.openai_compatible_base_url": "https://ex/v1",
                "asr.access_token": "x",
                "asr.model": "m",
                "asr.language": "",
            }.get(key, default)

        def get_int(self, key, default=None):
            if key == "asr.skip_if_text_shorter_than":
                return 50
            return int(default or 0)

        def get_float(self, key, default=None):
            return float(default or 0.0)

    monkeypatch.setattr(asr_validation, "config_manager", CM())
    spy = MagicMock()
    monkeypatch.setattr(asr_validation, "transcribe_openai_compatible", spy)
    assert (
        asr_validation.transcription_mismatch_reason(
            np.ones(500, dtype=np.float32), 24000, "short"
        )
        is None
    )
    spy.assert_not_called()
