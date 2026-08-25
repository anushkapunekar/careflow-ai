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

def book_appointment(
    appointment_id: int,
    patient_name: str
) -> dict:

    connection = get_connection()

    appointment = connection.execute(
        """
        SELECT id, appointment_date, appointment_time, status
        FROM appointments
        WHERE id = ?
        """,
        (appointment_id,)
    ).fetchone()

    if appointment is None:
        connection.close()

        return {
            "success": False,
            "message": "Appointment slot was not found."
        }

    if appointment["status"] != "available":
        connection.close()

        return {
            "success": False,
            "message": "Appointment slot is no longer available."
        }

    connection.execute(
        """
        UPDATE appointments
        SET patient_name = ?,
            status = 'booked'
        WHERE id = ?
        """,
        (patient_name, appointment_id)
    )

    connection.commit()
    connection.close()

    return {
        "success": True,
        "appointment_id": appointment_id,
        "date": appointment["appointment_date"],
        "time": appointment["appointment_time"],
        "patient_name": patient_name,
        "status": "booked"
    }