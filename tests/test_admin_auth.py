from unittest.mock import patch

import pytest
from fastapi import HTTPException

from admin_auth import verify_admin_access


def test_admin_auth_skipped_when_disabled():
    with patch("config.config_manager.get_bool", return_value=False):
        verify_admin_access(None)  # no exception


def test_admin_auth_missing_credentials():
    with patch("config.config_manager.get_bool", return_value=True):
        with pytest.raises(HTTPException) as exc:
            verify_admin_access(None)
        assert exc.value.status_code == 401


def test_restart_server_uses_threadpool(client, monkeypatch):
    import server_routes.admin_routes as admin_routes

    calls = []

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        calls.append(fn)
        return fn(*args, **kwargs)

    monkeypatch.setattr(admin_routes, "run_in_threadpool", fake_run_in_threadpool)

    r = client.post("/restart_server")

    assert r.status_code == 200
    assert calls == [admin_routes.engine.reload_model]


def test_unload_model_uses_threadpool(client, monkeypatch):
    import server_routes.admin_routes as admin_routes

    calls = []

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        calls.append(fn)
        return fn(*args, **kwargs)

    monkeypatch.setattr(admin_routes, "run_in_threadpool", fake_run_in_threadpool)

    r = client.post("/api/unload")

    assert r.status_code == 200
    assert calls == [admin_routes.engine.unload_model]
