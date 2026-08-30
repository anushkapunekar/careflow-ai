import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.tools import (
    check_appointment_availability,
    get_available_appointments,
    book_appointment,
)


def test_get_available_appointments():
    result = get_available_appointments(
        "2026-09-21"
    )

    assert result["available"] is True
    assert result["date"] == "2026-09-21"
    assert len(result["slots"]) > 0


def test_check_available_appointment():
    result = check_appointment_availability(
        "2026-09-21",
        "11:30"
    )

    assert result["available"] is True
    assert result["date"] == "2026-09-21"
    assert result["time"] == "11:30"
    assert "appointment_id" in result


def test_check_nonexistent_time():
    result = check_appointment_availability(
        "2026-09-21",
        "23:59"
    )

    assert result["available"] is False
    assert result["requested_time"] == "23:59"


def test_invalid_appointment_id():
    result = book_appointment(
        0,
        "Test Patient"
    )

    assert result["success"] is False


def test_invalid_patient_name():
    result = book_appointment(
        999999,
        ""
    )

    assert result["success"] is False


def test_invalid_placeholder_patient_name():
    result = book_appointment(
        999999,
        "patient"
    )

    assert result["success"] is False


def test_booking_nonexistent_appointment():
    result = book_appointment(
        999999,
        "Test Patient"
    )

    assert result["success"] is False