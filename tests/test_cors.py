from fastapi.testclient import TestClient


def test_cors_preflight(client: TestClient):
    headers = {
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type,Authorization",
    }

    response = client.options("/status", headers=headers)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "Content-Type" in response.headers["access-control-allow-headers"]
    assert "Authorization" in response.headers["access-control-allow-headers"]
