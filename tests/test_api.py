from fastapi.testclient import TestClient

from app.main import app


def test_health_and_system_routes_do_not_load_models() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/models/status")
    assert response.status_code == 200
    assert response.json()["loaded_model"] is None


def test_docs_are_available() -> None:
    response = TestClient(app).get("/docs")
    assert response.status_code == 200


def test_openapi_declares_generated_media_types() -> None:
    schema = TestClient(app).get("/openapi.json").json()
    assert "audio/wav" in schema["paths"]["/generate/audio"]["get"]["responses"]["200"]["content"]
    assert "image/png" in schema["paths"]["/generate/image"]["get"]["responses"]["200"]["content"]
    assert "video/mp4" in schema["paths"]["/generate/video"]["post"]["responses"]["200"]["content"]
    assert "model/obj" in schema["paths"]["/generate/3d"]["get"]["responses"]["200"]["content"]


def test_prompt_and_upload_validation_rejects_invalid_input() -> None:
    client = TestClient(app)
    assert client.get("/generate/text", params={"prompt": ""}).status_code == 422
    response = client.post(
        "/generate/video",
        files={"image": ("invalid.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 422


def test_web_ui_serves_html_and_rag_workspace() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    html = response.text
    for tab in ["Text", "Audio", "Image", "Video", "3D", "RAG"]:
        assert f'name: "{tab}"' in html or f"name: '{tab}'" in html
    assert "rag-workspace" in html
    assert "Knowledge Base Documents" in html
    assert "Semantic Q&A Workspace" in html
