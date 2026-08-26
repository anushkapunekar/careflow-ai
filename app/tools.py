from app.database import get_connection


def check_appointment_availability(
    date: str,
    preferred_time: str
) -> dict:

    connection = get_connection()

    appointment = connection.execute(
        """
        SELECT id, appointment_date, appointment_time, status
        FROM appointments
        WHERE appointment_date = ?
        AND appointment_time = ?
        """,
        (date, preferred_time)
    ).fetchone()

    connection.close()

    if appointment is None:
        return {
            "available": False,
            "date": date,
            "requested_time": preferred_time,
            "message": "No appointment slot exists at that time."
        }

    if appointment["status"] == "available":
        return {
            "available": True,
            "appointment_id": appointment["id"],
            "date": appointment["appointment_date"],
            "time": appointment["appointment_time"]
        }

    return {
        "available": False,
        "appointment_id": appointment["id"],
        "date": appointment["appointment_date"],
        "requested_time": appointment["appointment_time"],
        "message": "This appointment slot is already booked."
    }


def get_available_appointments(
    date: str
) -> dict:

    connection = get_connection()

    appointments = connection.execute(
        """
        SELECT id, appointment_date, appointment_time
        FROM appointments
        WHERE appointment_date = ?
        AND status = 'available'
        ORDER BY appointment_time
        """,
        (date,)
    ).fetchall()

    connection.close()

    if not appointments:
        return {
            "available": False,
            "date": date,
            "slots": [],
            "message": (
                "There are no available appointment slots "
                "on this date."
            )
        }

    slots = [
        {
            "appointment_id": appointment["id"],
            "time": appointment["appointment_time"]
        }
        for appointment in appointments
    ]

    return {
        "available": True,
        "date": date,
        "slots": slots
    }


def book_appointment(
    appointment_id: int,
    patient_name: str
) -> dict:

    patient_name = patient_name.strip()

    if not patient_name:
        return {
            "success": False,
            "message": "Patient name is required."
        }

    connection = get_connection()

    # Only book the slot if it is still available.
    cursor = connection.execute(
        """
        UPDATE appointments
        SET patient_name = ?,
            status = 'booked'
        WHERE id = ?
        AND status = 'available'
        """,
        (patient_name, appointment_id)
    )

    connection.commit()

    if cursor.rowcount == 0:

        connection.close()

        return {
            "success": False,
            "message": (
                "Appointment slot was not found or "
                "is no longer available."
            )
        }

    appointment = connection.execute(
        """
        SELECT id, appointment_date, appointment_time,
               patient_name, status
        FROM appointments
        WHERE id = ?
        """,
        (appointment_id,)
    ).fetchone()

    connection.close()

    return {
        "success": True,
        "appointment_id": appointment["id"],
        "date": appointment["appointment_date"],
        "time": appointment["appointment_time"],
        "patient_name": appointment["patient_name"],
        "status": appointment["status"]
    }