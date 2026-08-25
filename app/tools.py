from datetime import datetime


def check_appointment_availability(
    date: str,
    preferred_time: str
) -> dict:

    # Demo appointment data.
    # Later this will be replaced by a real database/API.

    available_slots = [
        "09:00",
        "10:00",
        "11:30",
        "14:00",
        "15:30"
    ]

    if preferred_time in available_slots:
        return {
            "available": True,
            "date": date,
            "time": preferred_time
        }

    return {
        "available": False,
        "date": date,
        "requested_time": preferred_time,
        "alternative_slots": available_slots
    }