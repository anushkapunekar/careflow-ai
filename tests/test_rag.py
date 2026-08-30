import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.rag import create_chunks, normalize_query


def test_create_chunks():
    text = """
CLINIC HOURS

The clinic is open Monday through Friday.

INSURANCE

The clinic accepts selected insurance plans.
"""

    chunks = create_chunks(text)

    assert len(chunks) == 2
    assert chunks[0].startswith("CLINIC HOURS")
    assert chunks[1].startswith("INSURANCE")


def test_normalize_walk_in_query():
    result = normalize_query(
        "can I come without an appointment"
    )

    assert "walk-in appointments" in result
    assert "walk in without an appointment" in result


def test_normalize_insurance_query():
    result = normalize_query(
        "do you accept insurance"
    )

    assert "insurance accepted" in result
    assert "insurance coverage" in result


def test_normalize_appointment_preparation_query():
    result = normalize_query(
        "what should I bring"
    )

    assert "what to bring" in result
    assert "photo ID" in result


def test_normalize_unknown_query():
    query = "tell me something random"

    result = normalize_query(query)

    assert result == query