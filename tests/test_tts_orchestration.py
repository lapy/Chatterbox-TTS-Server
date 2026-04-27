"""Unit tests for shared TTS orchestration (chunking, params, synthesis partial)."""

from __future__ import annotations

import asyncio
import sys
import types
from functools import partial

import engine as engine_module
import numpy as np
import pytest
import torch

import tts_orchestration
from config import config_manager
from models import CustomTTSRequest, OpenAISpeechRequest
from tts_orchestration import (
    ResolvedSynthesisParams,
    build_text_chunks,
    resolve_synthesis_params_custom,
    resolve_synthesis_params_openai,
    synthesize_text_chunks_async,
)


def test_build_text_chunks_no_split_when_disabled():
    text = "**Hello.** " * 100
    chunks = build_text_chunks(
        text, split_enabled=False, chunk_size=50, chunk_size_max=200
    )
    assert chunks == [("Hello. " * 100).strip()]


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
        types.SimpleNamespace(
            chunk_text_by_sentences=fake_chunk,
            normalize_markdown_for_tts=lambda value: value,
        ),
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
        types.SimpleNamespace(
            chunk_text_by_sentences=fake_chunk,
            normalize_markdown_for_tts=lambda value: value,
        ),
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


def test_synthesize_text_chunks_batch_preserves_input_order(monkeypatch):
    params = ResolvedSynthesisParams(0.8, 0.5, 0.5, 0, "en", 1.0)
    text_chunks = ["slow", "fast", "medium"]
    calls = []

    def fake_synthesize_batch(jobs, perf_monitor=None, log_prefix=""):
        calls.append([job["text"] for job in jobs])
        mapping = {
            "slow": torch.tensor([1.0], dtype=torch.float32),
            "fast": torch.tensor([2.0], dtype=torch.float32),
            "medium": torch.tensor([3.0], dtype=torch.float32),
        }
        return [(mapping[job["text"]], 24000) for job in jobs]

    monkeypatch.setattr(engine_module, "synthesize_batch", fake_synthesize_batch)
    # 0 => single batch with all chunks
    monkeypatch.setattr("tts_orchestration.config_manager.get_int", lambda _k, _d: 0)

    segments, sr = asyncio.run(
        synthesize_text_chunks_async(
            text_chunks,
            audio_prompt_path_str=None,
            params=params,
            perf_monitor=None,
            log_prefix="test",
        )
    )

    assert sr == 24000
    assert [float(seg.reshape(-1)[0]) for seg in segments] == [1.0, 2.0, 3.0]
    assert calls == [["slow", "fast", "medium"]]


def test_synthesize_text_chunks_batch_splits_when_chunk_batch_size_set(monkeypatch):
    params = ResolvedSynthesisParams(0.8, 0.5, 0.5, 0, "en", 1.0)
    text_chunks = ["slow", "fast", "medium"]
    calls = []

    def fake_synthesize_batch(jobs, perf_monitor=None, log_prefix=""):
        calls.append([job["text"] for job in jobs])
        mapping = {
            "slow": torch.tensor([1.0], dtype=torch.float32),
            "fast": torch.tensor([2.0], dtype=torch.float32),
            "medium": torch.tensor([3.0], dtype=torch.float32),
        }
        return [(mapping[job["text"]], 24000) for job in jobs]

    monkeypatch.setattr(engine_module, "synthesize_batch", fake_synthesize_batch)

    def get_int_batch2(key, default=0):
        if key == "tts_engine.chunk_batch_size":
            return 2
        return default

    monkeypatch.setattr("tts_orchestration.config_manager.get_int", get_int_batch2)

    segments, sr = asyncio.run(
        synthesize_text_chunks_async(
            text_chunks,
            audio_prompt_path_str=None,
            params=params,
            perf_monitor=None,
            log_prefix="test",
        )
    )

    assert sr == 24000
    assert [float(seg.reshape(-1)[0]) for seg in segments] == [1.0, 2.0, 3.0]
    assert calls == [["slow", "fast"], ["medium"]]


def test_chunk_quality_retry_invokes_synthesize(monkeypatch):
    """Near-silent first pass should trigger heuristic retry."""
    params = ResolvedSynthesisParams(0.8, 0.5, 0.5, 99, "en", 1.0)
    synth_calls = []

    def fake_batch(jobs, perf_monitor=None, log_prefix=""):
        return [(torch.zeros(4800, dtype=torch.float32), 24000) for _ in jobs]

    def fake_synthesize(text, audio_prompt_path=None, *args, **kwargs):
        synth_calls.append((text, audio_prompt_path))
        return torch.ones(8000, dtype=torch.float32) * 0.2, 24000

    monkeypatch.setattr(engine_module, "synthesize_batch", fake_batch)
    monkeypatch.setattr(engine_module, "synthesize", fake_synthesize)

    def get_int_q(key, default=0):
        if key == "tts_engine.chunk_quality_max_retries":
            return 2
        if key == "tts_engine.chunk_batch_size":
            return 0
        return default

    monkeypatch.setattr("tts_orchestration.config_manager.get_int", get_int_q)

    segments, sr = asyncio.run(
        synthesize_text_chunks_async(
            ["This is a longer phrase that should not be silent."],
            None,
            params,
            perf_monitor=None,
            log_prefix="test",
        )
    )
    assert sr == 24000
    assert synth_calls
    assert float(np.max(np.abs(segments[0]))) > 0.01


def test_chunk_asr_mismatch_triggers_resynthesis(monkeypatch):
    """ASR validation failure should schedule a chunk resynthesis like heuristic failure."""
    synth_calls = []

    def fake_batch(jobs, perf_monitor=None, log_prefix=""):
        return [(torch.ones(96_000, dtype=torch.float32) * 0.12, 24000) for _ in jobs]

    def fake_synthesize(text, audio_prompt_path=None, *args, **kwargs):
        synth_calls.append(text)
        return torch.ones(96_000, dtype=torch.float32) * 0.12, 24000

    monkeypatch.setattr(engine_module, "synthesize_batch", fake_batch)
    monkeypatch.setattr(engine_module, "synthesize", fake_synthesize)

    asr_calls = {"n": 0}

    def fake_tm(seg, sr, text):
        asr_calls["n"] += 1
        if asr_calls["n"] == 1:
            return "asr_low_similarity_0.10"
        return None

    monkeypatch.setattr(
        tts_orchestration.asr_validation,
        "transcription_mismatch_reason",
        fake_tm,
    )

    orig_gb = config_manager.get_bool

    def get_bool_patched(key, default=None):
        if key == "asr.enabled":
            return True
        return orig_gb(key, default)

    monkeypatch.setattr(config_manager, "get_bool", get_bool_patched)

    orig_get_int = config_manager.get_int

    def get_int_patched(key, default=0):
        if key == "tts_engine.chunk_quality_max_retries":
            return 2
        if key == "tts_engine.chunk_batch_size":
            return 0
        return orig_get_int(key, default)

    monkeypatch.setattr(config_manager, "get_int", get_int_patched)

    params = ResolvedSynthesisParams(0.8, 0.5, 0.5, 1, "en", 1.0)
    segments, sr = asyncio.run(
        synthesize_text_chunks_async(
            ["This is enough text for both heuristics and ASR length gates."],
            None,
            params,
            perf_monitor=None,
            log_prefix="test",
        )
    )
    assert sr == 24000
    assert len(synth_calls) == 1
    assert asr_calls["n"] == 2
    assert segments[0].size > 0


def test_chunked_reference_path_only_on_first_job(monkeypatch):
    """Later chunks must not pass audio_prompt_path so conditioning is reused."""
    params = ResolvedSynthesisParams(0.8, 0.5, 0.5, 0, "en", 1.0)
    captured = []

    def fake_synthesize_batch(jobs, perf_monitor=None, log_prefix=""):
        captured.append([(j.get("audio_prompt_path"), j["text"]) for j in jobs])
        return [(torch.tensor([1.0], dtype=torch.float32), 24000) for _ in jobs]

    monkeypatch.setattr(engine_module, "synthesize_batch", fake_synthesize_batch)

    def get_int_seq(key, default=0):
        if key == "tts_engine.chunk_batch_size":
            return 0
        return default

    monkeypatch.setattr("tts_orchestration.config_manager.get_int", get_int_seq)

    asyncio.run(
        synthesize_text_chunks_async(
            ["chunk a", "chunk b"],
            "/voices/ref.wav",
            params,
            perf_monitor=None,
            log_prefix="test",
        )
    )
    assert captured == [[("/voices/ref.wav", "chunk a"), (None, "chunk b")]]


def test_stale_parallel_workers_config_is_ignored(monkeypatch):
    """Chunk synthesis stays sequential even if an old config still has parallel workers."""
    params = ResolvedSynthesisParams(0.8, 0.5, 0.5, 42, "en", 1.0)
    batch_calls = []

    def fake_batch(jobs, perf_monitor=None, log_prefix=""):
        batch_calls.append([(j.get("audio_prompt_path"), j["seed"]) for j in jobs])
        return [
            (torch.tensor([float(i + 1)], dtype=torch.float32), 24000)
            for i in range(len(jobs))
        ]

    monkeypatch.setattr(engine_module, "synthesize_batch", fake_batch)

    def get_int_par(key, default=0):
        if key == "tts_engine.parallel_chunk_workers":
            return 3
        if key == "tts_engine.chunk_batch_size":
            return 0
        return default

    monkeypatch.setattr("tts_orchestration.config_manager.get_int", get_int_par)

    segments, sr = asyncio.run(
        synthesize_text_chunks_async(
            ["chunk a", "chunk b"],
            "/voices/ref.wav",
            params,
            perf_monitor=None,
            log_prefix="test",
        )
    )
    assert sr == 24000
    assert [float(s.reshape(-1)[0]) for s in segments] == [1.0, 2.0]
    assert batch_calls == [[("/voices/ref.wav", 42), (None, 42)]]


def test_synthesize_text_chunks_batch_does_not_apply_speed_factor(monkeypatch):
    """Speed is applied post-stitch in route handlers, not in orchestration."""
    params = ResolvedSynthesisParams(0.8, 0.5, 0.5, 0, "en", 1.5)
    text_chunks = ["a", "b"]
    speed_calls = []

    def fake_synthesize_batch(jobs, perf_monitor=None, log_prefix=""):
        return [
            (torch.tensor([1.0], dtype=torch.float32), 24000),
            (torch.tensor([2.0], dtype=torch.float32), 24000),
        ]

    def fake_apply_speed_factor(audio_tensor, sample_rate, speed_factor):
        speed_calls.append((float(audio_tensor[0]), sample_rate, speed_factor))
        return audio_tensor + 10, sample_rate

    monkeypatch.setattr(engine_module, "synthesize_batch", fake_synthesize_batch)
    monkeypatch.setattr("tts_orchestration.config_manager.get_int", lambda _k, _d: 0)
    monkeypatch.setitem(
        sys.modules,
        "utils",
        types.SimpleNamespace(apply_speed_factor=fake_apply_speed_factor),
    )

    segments, sr = asyncio.run(
        synthesize_text_chunks_async(
            text_chunks,
            audio_prompt_path_str=None,
            params=params,
            perf_monitor=None,
            log_prefix="test",
        )
    )

    assert sr == 24000
    assert [float(seg.reshape(-1)[0]) for seg in segments] == [1.0, 2.0]
    assert speed_calls == []
