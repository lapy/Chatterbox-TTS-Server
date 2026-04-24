from __future__ import annotations

import numpy as np

import utils


def test_detect_glitch_empty_and_non_finite():
    assert utils.detect_chunk_audio_glitch(np.array([], dtype=np.float32), 24000, "hello") == "empty_waveform"
    bad = np.array([0.1, np.nan, 0.1], dtype=np.float32)
    assert utils.detect_chunk_audio_glitch(bad, 24000, "hello") == "non_finite_samples"


def test_detect_glitch_near_silent():
    w = np.full(4800, 1e-6, dtype=np.float32)
    assert utils.detect_chunk_audio_glitch(w, 24000, "hello there") == "near_silent"


def test_detect_glitch_too_short_for_long_text():
    # Long enough to pass min_wav_duration_sec, still far below expected speech length
    w = np.random.default_rng(0).normal(0, 0.02, 2400).astype(np.float32)
    text = "abcdefghijklmnopqrstuvwxyz" * 3
    assert utils.detect_chunk_audio_glitch(w, 24000, text) == "too_short_for_text"


def test_detect_glitch_excessive_leading_silence():
    sr = 24000
    n_pre = int(1.1 * sr)
    n_voice = int(0.4 * sr)
    w = np.concatenate(
        [np.zeros(n_pre, dtype=np.float32), np.sin(np.linspace(0, 80, n_voice)).astype(np.float32) * 0.2]
    )
    text = "This is enough text to run the leading silence heuristic on the chunk."
    assert utils.detect_chunk_audio_glitch(w, sr, text) == "excessive_leading_silence"


def test_detect_glitch_ok_for_typical_tone():
    sr = 24000
    w = np.sin(np.linspace(0, 200, int(1.5 * sr))).astype(np.float32) * 0.15
    text = "A reasonable sentence for speech synthesis quality checks."
    assert utils.detect_chunk_audio_glitch(w, sr, text) is None


def test_derive_chunk_retry_seed_zero_base():
    assert utils.derive_chunk_retry_seed(0, 1, 1) == 0


def test_derive_chunk_retry_seed_deterministic():
    assert utils.derive_chunk_retry_seed(42, 2, 1) == utils.derive_chunk_retry_seed(42, 2, 1)
    assert utils.derive_chunk_retry_seed(42, 2, 1) != utils.derive_chunk_retry_seed(42, 2, 2)
