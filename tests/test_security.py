"""Tests for API hardening: error responses and CORS.

Defects:
  * The global exception handler returned `traceback.format_exc()` in the
    response body, disclosing paths, SQL and library internals.
  * CORS was `allow_origins=["*"]` + `allow_credentials=True` -- a
    combination browsers reject, and which exposes an unauthenticated
    token-spending endpoint to every site on the internet.
"""
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _boom_route():
    """Register a route that raises, on a throwaway app sharing the config."""

    @app.get("/_test_boom")
    def _boom():
        raise ValueError("SECRET_DB_PASSWORD=hunter2 /srv/app/storage/x")

    @app.get("/_test_boom_custom")
    def _boom2():
        from fastapi import HTTPException
        raise HTTPException(status_code=418, detail="teapot detail")

    return app


def test_unhandled_error_does_not_leak_traceback(client):
    _boom_route()
    r = client.get("/_test_boom")
    assert r.status_code == 500
    body = r.text
    assert "SECRET_DB_PASSWORD" not in body
    assert "Traceback" not in body
    assert "ValueError" not in body
    assert "/srv/app" not in body
    assert r.json() == {"message": "Internal server error"}


def test_client_errors_keep_their_detail(client):
    """HTTPException detail is a deliberate, safe message -- keep it."""
    _boom_route()
    r = client.get("/_test_boom_custom")
    assert r.status_code == 418
    assert r.json()["detail"] == "teapot detail"


def test_cors_is_not_wildcard(client):
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    # A wildcard allow-list would echo the requesting origin back.
    assert r.headers.get("access-control-allow-origin") != "https://evil.example"
    assert "access-control-allow-origin" not in r.headers or \
        r.headers["access-control-allow-origin"] in {
            "http://localhost:5173", "http://localhost:5174",
            "http://localhost:5175", "http://127.0.0.1:5173",
        }


def test_cors_allows_local_dashboard(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_credentials_not_enabled_with_wildcard(client):
    """`*` + credentials is rejected by browsers; must not be configured."""
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-credentials") is None


def test_allowed_origins_are_configurable(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example, https://alt.example")
    import importlib
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    assert main_mod._allowed_origins() == ["https://app.example", "https://alt.example"]


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "healthy"}
