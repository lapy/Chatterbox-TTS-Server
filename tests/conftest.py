"""Pytest fixtures: isolate config directory and inject a stub `engine` module for API tests."""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch


def make_engine_stub():
    """Minimal engine API used by server, routes, and audio_pipeline tests."""

    def _fake_synthesize(
        text="",
        audio_prompt_path=None,
        temperature=0.8,
        exaggeration=0.5,
        cfg_weight=0.5,
        seed=0,
        language="en",
        **_,
    ):
        return torch.zeros(8000, dtype=torch.float32), 24000

    def _fake_synthesize_batch(jobs, perf_monitor=None, log_prefix=""):
        return [_fake_synthesize(**job) for job in jobs]

    def _fake_iter(chunks, path, *args, **kwargs):
        for _c in chunks:
            yield torch.zeros(4000, dtype=torch.float32), 24000

    def _info():
        return {
            "class_name": "StubModel",
            "type": "turbo",
            "sample_rate": 24000,
            "loaded": True,
            "device": "cpu",
            "supports_paralinguistic_tags": True,
        }

    stub = types.ModuleType("engine")
    stub.load_model = lambda: True
    stub.reload_model = lambda: True
    stub.unload_model = lambda: True
    stub.synthesize = _fake_synthesize
    stub.synthesize_batch = _fake_synthesize_batch
    stub.iter_synthesize_under_lock = _fake_iter
    stub.get_model_info = _info
    stub.MODEL_LOADED = True
    return stub


def ensure_test_engine_stub():
    """
    Install stub before any test imports `audio_pipeline` / `tts_orchestration`
    (avoids importing real `engine.py`, which requires chatterbox).

    Set CHATTERBOX_TEST_USE_REAL_ENGINE=1 to skip stubbing when running against a real engine.
    """
    if os.environ.get("CHATTERBOX_TEST_USE_REAL_ENGINE") == "1":
        return
    existing = sys.modules.get("engine")
    if existing is not None and getattr(existing, "__file__", None):
        return
    sys.modules["engine"] = make_engine_stub()


ensure_test_engine_stub()


def pytest_collection_modifyitems(config, items):
    """Skip TestClient tests when optional audio deps are missing (minimal dev env)."""
    try:
        import pydub  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(
            reason="Install requirements.txt (pydub, soundfile, …) for integration tests."
        )
        for item in items:
            if "client" in getattr(item, "fixturenames", ()):
                item.add_marker(skip)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient without importing real `engine.py` (no chatterbox dependency)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "index.html").write_text(
        "<!DOCTYPE html><html><body>test</body></html>"
    )
    (tmp_path / "ui" / "presets.yaml").write_text("[]\n")
    (tmp_path / "voices").mkdir()
    (tmp_path / "reference_audio").mkdir()
    (tmp_path / "outputs").mkdir()

    sys.modules["engine"] = make_engine_stub()

    to_drop = [
        m
        for m in list(sys.modules)
        if m == "server"
        or m.startswith("server_routes")
        or m == "audio_pipeline"
        or m == "tts_orchestration"
    ]
    for m in to_drop:
        sys.modules.pop(m, None)

    from fastapi.testclient import TestClient

    import server

    with TestClient(server.app) as tc:
        yield tc
