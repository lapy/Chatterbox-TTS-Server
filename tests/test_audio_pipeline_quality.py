from __future__ import annotations

import numpy as np

import audio_pipeline


def test_true_peak_limiter_scales_when_peak_exceeds_limit():
    wave = np.array([0.1, -0.8, 1.2, -1.0], dtype=np.float32)
    limited = audio_pipeline._apply_true_peak_limiter(wave, peak_limit=0.95)
    assert float(np.max(np.abs(limited))) <= 0.950001


def test_gentle_cleanup_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(
        "audio_pipeline.config_manager.get_bool",
        lambda key, default=None: False
        if key == "audio_processing.gentle_cleanup.enabled"
        else default,
    )
    wave = np.array([0.2, -0.3, 0.4], dtype=np.float32)
    cleaned = audio_pipeline._apply_gentle_voice_cleanup(wave, 24000)
    assert np.allclose(cleaned, wave)


def test_gentle_cleanup_applies_limiter_only_when_requested(monkeypatch):
    bool_values = {
        "audio_processing.gentle_cleanup.enabled": True,
        "audio_processing.gentle_cleanup.enable_highpass": False,
        "audio_processing.gentle_cleanup.enable_loudness_normalization": False,
        "audio_processing.gentle_cleanup.enable_true_peak_limiter": True,
    }

    monkeypatch.setattr(
        "audio_pipeline.config_manager.get_bool",
        lambda key, default=None: bool_values.get(key, default),
    )
    monkeypatch.setattr(
        "audio_pipeline.config_manager.get_float",
        lambda key, default=None: 0.5
        if key == "audio_processing.gentle_cleanup.true_peak_limit"
        else default,
    )

    wave = np.array([0.1, -1.0, 0.9], dtype=np.float32)
    cleaned = audio_pipeline._apply_gentle_voice_cleanup(wave, 24000)
    assert float(np.max(np.abs(cleaned))) <= 0.500001
