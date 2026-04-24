from config import get_app_version


def test_version_file():
    assert get_app_version() == "2.0.2"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "2.0.2"
    assert "ffmpeg_available" in body
    assert isinstance(body["ffmpeg_available"], bool)


def test_ready_when_model_patched(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
