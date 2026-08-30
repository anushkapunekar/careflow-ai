import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200


def test_chat_endpoint():
    response = client.post(
        "/api/chat",
        json={
            "conversation_id": "test-health-conversation",
            "message": "what are your clinic hours"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data is not None


def test_chat_walk_in_question():
    response = client.post(
        "/api/chat",
        json={
            "conversation_id": "test-walk-in-conversation",
            "message": "do you accept walk-ins"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data is not None


def test_chat_empty_message():
    response = client.post(
        "/api/chat",
        json={
            "conversation_id": "test-empty-conversation",
            "message": ""
        }
    )

    assert response.status_code == 422