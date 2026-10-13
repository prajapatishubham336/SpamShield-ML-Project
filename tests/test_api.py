from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


def test_home_page():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "SpamShield ML" in response.text


def test_favicon_endpoint():
    with TestClient(app) as client:
        response = client.get("/favicon.ico")

    assert response.status_code == 204


def test_prediction_without_model_returns_service_unavailable(monkeypatch,):
    import app.main as main

    with TestClient(app) as client:
        # Set MODEL to None after application startup.
        monkeypatch.setattr(main, "MODEL", None)

        response = client.post(
            "/predict",
            json={"text": "Hello, how are you?"},
        )

    assert response.status_code == 503
    assert "Model is not trained" in response.json()["detail"]
