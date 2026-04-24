"""Pytest fixtures: isolate config directory and inject a stub `engine` module for API tests."""

from __future__ import annotations

import sys
import types

import pytest
import torch


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

    def _fake_synthesize(**kwargs):
        return torch.zeros(8000, dtype=torch.float32), 24000

    def _info():
        return {
            "class_name": "StubModel",
            "type": "turbo",
            "sample_rate": 24000,
        }

    stub = types.ModuleType("engine")
    stub.load_model = lambda: True
    stub.reload_model = lambda: True
    stub.unload_model = lambda: True
    stub.synthesize = _fake_synthesize
    stub.get_model_info = _info
    stub.MODEL_LOADED = True
    sys.modules["engine"] = stub

    to_drop = [
        m
        for m in list(sys.modules)
        if m == "server"
        or m.startswith("server_routes")
        or m == "audio_pipeline"
    ]
    for m in to_drop:
        del sys.modules[m]

    from fastapi.testclient import TestClient

    import server

    with TestClient(server.app) as tc:
        yield tc
