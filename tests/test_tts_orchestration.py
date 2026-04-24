"""Unit tests for shared TTS orchestration (chunking, params, synthesis partial)."""

from __future__ import annotations

import sys
import types
from functools import partial

import pytest

from models import CustomTTSRequest, OpenAISpeechRequest
from tts_orchestration import (
    ResolvedSynthesisParams,
    build_text_chunks,
    resolve_synthesis_params_custom,
    resolve_synthesis_params_openai,
)


def test_build_text_chunks_no_split_when_disabled():
    text = "Hello. " * 100
    chunks = build_text_chunks(
        text, split_enabled=False, chunk_size=50, chunk_size_max=200
    )
    assert chunks == [text]


def test_build_text_chunks_no_split_when_below_threshold():
    # threshold = 100 * 1.5 = 150; text shorter than that
    text = "short"
    chunks = build_text_chunks(
        text, split_enabled=True, chunk_size=100, chunk_size_max=200
    )
    assert chunks == [text]


def test_build_text_chunks_splits_when_long(monkeypatch):
    def fake_chunk(text, size):
        assert size == 100
        return ["first", "second"]

    monkeypatch.setitem(
        sys.modules,
        "utils",
        types.SimpleNamespace(chunk_text_by_sentences=fake_chunk),
    )
    text = "Hello. " * 40
    assert len(text) > 100 * 1.5
    chunks = build_text_chunks(
        text, split_enabled=True, chunk_size=100, chunk_size_min=50, chunk_size_max=200
    )
    assert chunks == ["first", "second"]


def test_build_text_chunks_clamps_chunk_size(monkeypatch):
    sizes = []

    def fake_chunk(text, size):
        sizes.append(size)
        mid = max(1, len(text) // 2)
        return [text[:mid], text[mid:]]

    monkeypatch.setitem(
        sys.modules,
        "utils",
        types.SimpleNamespace(chunk_text_by_sentences=fake_chunk),
    )
    # Long enough to split for both chunk_size cases (threshold = chunk_size * 1.5)
    text = "x. " * 400
    build_text_chunks(
        text, split_enabled=True, chunk_size=10, chunk_size_min=50, chunk_size_max=200
    )
    build_text_chunks(
        text, split_enabled=True, chunk_size=9999, chunk_size_min=50, chunk_size_max=200
    )
    assert sizes == [50, 200]


def test_resolve_synthesis_params_custom_explicit():
    req = CustomTTSRequest(
        text="hi",
        temperature=0.1,
        exaggeration=0.2,
        cfg_weight=0.3,
        seed=42,
        language="fr",
        speed_factor=1.25,
        predefined_voice_id="v.wav",
    )
    p = resolve_synthesis_params_custom(req)
    assert p == ResolvedSynthesisParams(
        temperature=0.1,
        exaggeration=0.2,
        cfg_weight=0.3,
        seed=42,
        language="fr",
        speed_factor=1.25,
    )


def test_resolve_synthesis_params_openai_speed_clamped(monkeypatch):
    monkeypatch.setattr("tts_orchestration.get_gen_default_seed", lambda: 0)
    monkeypatch.setattr(
        "tts_orchestration.get_gen_default_temperature", lambda: 0.8
    )
    monkeypatch.setattr(
        "tts_orchestration.get_gen_default_exaggeration", lambda: 0.5
    )
    monkeypatch.setattr("tts_orchestration.get_gen_default_cfg_weight", lambda: 0.5)
    monkeypatch.setattr("tts_orchestration.get_gen_default_language", lambda: "en")
    monkeypatch.setattr("tts_orchestration.config_manager.get_float", lambda k, d: d)
    monkeypatch.setattr("tts_orchestration.config_manager.get", lambda k, d=None: None)

    monkeypatch.setattr(
        "tts_orchestration.get_gen_default_speed_factor", lambda: 2.0
    )
    req = OpenAISpeechRequest(
        model="m", input="x", voice="v.wav", speed=4.0, seed=7
    )
    p = resolve_synthesis_params_openai(req)
    assert p.seed == 7
    assert p.speed_factor == 4.0

    monkeypatch.setattr(
        "tts_orchestration.get_gen_default_speed_factor", lambda: 0.5
    )
    req2 = OpenAISpeechRequest(
        model="m", input="x", voice="v.wav", speed=0.25, seed=None
    )
    p2 = resolve_synthesis_params_openai(req2)
    assert p2.speed_factor == 0.25


def test_resolve_synthesis_params_openai_reads_ui_language(monkeypatch):
    monkeypatch.setattr(
        "tts_orchestration.get_gen_default_speed_factor", lambda: 1.0
    )
    monkeypatch.setattr("tts_orchestration.get_gen_default_seed", lambda: 0)
    monkeypatch.setattr(
        "tts_orchestration.get_gen_default_temperature", lambda: 0.8
    )
    monkeypatch.setattr(
        "tts_orchestration.get_gen_default_exaggeration", lambda: 0.5
    )
    monkeypatch.setattr("tts_orchestration.get_gen_default_cfg_weight", lambda: 0.5)
    monkeypatch.setattr("tts_orchestration.get_gen_default_language", lambda: "en")

    def _get_float(key, default=None):
        return default if default is not None else 0.5

    def _get(key, default=None):
        if key == "ui_state.last_language":
            return "de"
        return default

    monkeypatch.setattr("tts_orchestration.config_manager.get_float", _get_float)
    monkeypatch.setattr("tts_orchestration.config_manager.get", _get)

    req = OpenAISpeechRequest(model="m", input="x", voice="v.wav")
    p = resolve_synthesis_params_openai(req)
    assert p.language == "de"


def test_synthesize_partial_keyword_binding_contract():
    """Regression: audio path must be keyword-bound so chunk text stays first positional."""

    def stub(
        chunk_text,
        audio_prompt_path_str=None,
        *,
        temperature,
        exaggeration,
        cfg_weight,
        seed,
        language,
        speed_factor,
    ):
        return chunk_text, audio_prompt_path_str, language

    params = ResolvedSynthesisParams(0.8, 0.5, 0.5, 0, "en", 1.0)
    fn = partial(
        stub,
        audio_prompt_path_str="/voices/a.wav",
        temperature=params.temperature,
        exaggeration=params.exaggeration,
        cfg_weight=params.cfg_weight,
        seed=params.seed,
        language=params.language,
        speed_factor=params.speed_factor,
    )
    out = fn("[sigh] test phrase")
    assert out == ("[sigh] test phrase", "/voices/a.wav", "en")
