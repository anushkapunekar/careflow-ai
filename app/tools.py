from app.database import (
    get_connection,
    ensure_appointment_slots
)


# ==================================================
# CHECK SPECIFIC APPOINTMENT AVAILABILITY
# ==================================================

def check_appointment_availability(
    date: str,
    preferred_time: str
) -> dict:

    ensure_appointment_slots(date)

    connection = get_connection()

    try:

        appointment = connection.execute(
            """
            SELECT
                id,
                appointment_date,
                appointment_time,
                status
            FROM appointments
            WHERE appointment_date = ?
            AND appointment_time = ?
            """,
            (
                date,
                preferred_time
            )
        ).fetchone()

    finally:

        connection.close()


    if appointment is None:

        return {
            "available": False,
            "date": date,
            "requested_time": preferred_time,
            "message": (
                "No appointment slot exists at that time."
            )
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
        "message": (
            "This appointment slot is already booked."
        )
    }


# ==================================================
# GET AVAILABLE APPOINTMENTS
# ==================================================

def get_available_appointments(
    date: str
) -> dict:

    ensure_appointment_slots(date)

    connection = get_connection()

    try:

        appointments = connection.execute(
            """
            SELECT
                id,
                appointment_date,
                appointment_time
            FROM appointments
            WHERE appointment_date = ?
            AND status = 'available'
            ORDER BY appointment_time
            """,
            (date,)
        ).fetchall()

    finally:

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

#====================================================
#GET SCHEDULED APPOINTMENTS BY THE PATIENT NAME
#====================================================
def get_patient_appointments(
    patient_name: str
) -> dict:

    if not isinstance(patient_name, str):
        return {
            "success": False,
            "appointments": [],
            "message": "A valid patient name is required."
        }

    patient_name = patient_name.strip()

    if not patient_name:
        return {
            "success": False,
            "appointments": [],
            "message": "A valid patient name is required."
        }

    connection = get_connection()

    try:

        appointments = connection.execute(
            """
            SELECT
                id,
                appointment_date,
                appointment_time,
                patient_name,
                status
            FROM appointments
            WHERE LOWER(patient_name) = LOWER(?)
            AND status = 'booked'
            ORDER BY appointment_date, appointment_time
            """,
            (patient_name,)
        ).fetchall()

    finally:

        connection.close()

    if not appointments:

        return {
            "success": True,
            "appointments": [],
            "message": (
                f"No scheduled appointments were found "
                f"for {patient_name}."
            )
        }

    return {
        "success": True,
        "appointments": [
            {
                "appointment_id": appointment["id"],
                "date": appointment["appointment_date"],
                "time": appointment["appointment_time"],
                "patient_name": appointment["patient_name"],
                "status": appointment["status"],
            }
            for appointment in appointments
        ]
    }

# ==================================================
# BOOK APPOINTMENT
# ==================================================

def book_appointment(
    appointment_id: int,
    patient_name: str
) -> dict:

    # --------------------------------------------------
    # VALIDATE APPOINTMENT ID
    # --------------------------------------------------

    if not isinstance(
        appointment_id,
        int
    ) or appointment_id <= 0:

        return {
            "success": False,
            "message": (
                "A valid appointment slot is required."
            )
        }


    # --------------------------------------------------
    # VALIDATE PATIENT NAME
    # --------------------------------------------------

    if not isinstance(
        patient_name,
        str
    ):

        return {
            "success": False,
            "message": (
                "Patient name is required."
            )
        }


    patient_name = patient_name.strip()


    invalid_names = {
        "",
        "john doe",
        "patient",
        "patient name",
        "[patient_name]",
        "[to be provided by user]",
        "unknown",
        "n/a",
        "na",
        "none",
        "null"
    }


    if patient_name.lower() in invalid_names:

        return {
            "success": False,
            "message": (
                "A valid patient name must be provided."
            )
        }


    # --------------------------------------------------
    # OPEN DATABASE CONNECTION
    # --------------------------------------------------

    connection = get_connection()

    try:

        # --------------------------------------------------
        # ATOMIC BOOKING OPERATION
        # --------------------------------------------------
        #
        # The slot is booked only if it is STILL available.
        #
        # This means the database remains the final authority.
        #
        # Even if availability was checked a moment earlier,
        # another request cannot successfully book the same
        # slot after its status has changed.
        #

        cursor = connection.execute(
            """
            UPDATE appointments
            SET
                patient_name = ?,
                status = 'booked'
            WHERE id = ?
            AND status = 'available'
            """,
            (
                patient_name,
                appointment_id
            )
        )


        # --------------------------------------------------
        # BOOKING FAILED
        # --------------------------------------------------

        if cursor.rowcount == 0:

            connection.rollback()

            return {
                "success": False,
                "message": (
                    "Appointment slot was not found "
                    "or is no longer available."
                )
            }


        # --------------------------------------------------
        # COMMIT SUCCESSFUL BOOKING
        # --------------------------------------------------

        connection.commit()


        # --------------------------------------------------
        # FETCH CONFIRMED BOOKING
        # --------------------------------------------------

        appointment = connection.execute(
            """
            SELECT
                id,
                appointment_date,
                appointment_time,
                patient_name,
                status
            FROM appointments
            WHERE id = ?
            """,
            (appointment_id,)
        ).fetchone()


        if appointment is None:

            connection.rollback()

            return {
                "success": False,
                "message": (
                    "The appointment could not be "
                    "confirmed."
                )
            }


        # --------------------------------------------------
        # RETURN CONFIRMED RESULT
        # --------------------------------------------------

        return {

            "success": True,

            "appointment_id":
                appointment["id"],

            "date":
                appointment["appointment_date"],

            "time":
                appointment["appointment_time"],

            "patient_name":
                appointment["patient_name"],

            "status":
                appointment["status"]
        }


    except Exception:

        # --------------------------------------------------
        # ROLLBACK ON UNEXPECTED DATABASE ERROR
        # --------------------------------------------------

        connection.rollback()

        return {
            "success": False,
            "message": (
                "The appointment could not be booked "
                "because of a database error."
            )
        }


    finally:

        connection.close()


# ==================================================
# CANCEL APPOINTMENT
# ==================================================

# ==================================================
# CANCEL APPOINTMENT
# ==================================================

def cancel_appointment(
    appointment_id: int
) -> dict:

    if not isinstance(appointment_id, int) or appointment_id <= 0:
        return {
            "success": False,
            "message": "A valid appointment slot is required."
        }

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            UPDATE appointments
            SET
                patient_name = NULL,
                status = 'available'
            WHERE id = ?
            AND status = 'booked'
            """,
            (appointment_id,)
        )

        if cursor.rowcount == 0:
            connection.rollback()

            return {
                "success": False,
                "message": "The appointment was not found or is not currently booked."
            }

        connection.commit()

        appointment = connection.execute(
            """
            SELECT
                id,
                appointment_date,
                appointment_time,
                patient_name,
                status
            FROM appointments
            WHERE id = ?
            """,
            (appointment_id,)
        ).fetchone()

        if appointment is None:
            return {
                "success": False,
                "message": "The appointment could not be confirmed as cancelled."
            }

        return {
            "success": True,
            "appointment_id": appointment["id"],
            "date": appointment["appointment_date"],
            "time": appointment["appointment_time"],
            "status": appointment["status"]
        }

    except Exception:
        connection.rollback()

        return {
            "success": False,
            "message": "The appointment could not be cancelled because of a database error."
        }

    finally:
        connection.close()

  # ==================================================
# RESCHEDULE APPOINTMENT
# ==================================================

def reschedule_appointment(
    appointment_id: int,
    new_appointment_id: int
) -> dict:

    if (
        not isinstance(appointment_id, int)
        or appointment_id <= 0
        or not isinstance(new_appointment_id, int)
        or new_appointment_id <= 0
    ):
        return {
            "success": False,
            "message": "Valid appointment slots are required."
        }

    if appointment_id == new_appointment_id:
        return {
            "success": False,
            "message": "The new appointment must be different from the current appointment."
        }

    connection = get_connection()

    try:

        # --------------------------------------------------
        # GET CURRENT BOOKED APPOINTMENT
        # --------------------------------------------------

        current = connection.execute(
            """
            SELECT
                id,
                appointment_date,
                appointment_time,
                patient_name,
                status
            FROM appointments
            WHERE id = ?
            AND status = 'booked'
            """,
            (appointment_id,)
        ).fetchone()

        if current is None:

            return {
                "success": False,
                "message": (
                    "The current appointment was not found "
                    "or is not currently booked."
                )
            }


        # --------------------------------------------------
        # CHECK NEW APPOINTMENT
        # --------------------------------------------------

        new_slot = connection.execute(
            """
            SELECT
                id,
                appointment_date,
                appointment_time,
                status
            FROM appointments
            WHERE id = ?
            """,
            (new_appointment_id,)
        ).fetchone()

        if new_slot is None:

            return {
                "success": False,
                "message": "The new appointment slot was not found."
            }


        if new_slot["status"] != "available":

            return {
                "success": False,
                "message": "The new appointment slot is no longer available."
            }


        # --------------------------------------------------
        # ATOMIC RESCHEDULE
        # --------------------------------------------------

        connection.execute(
            """
            UPDATE appointments
            SET
                patient_name = NULL,
                status = 'available'
            WHERE id = ?
            AND status = 'booked'
            """,
            (appointment_id,)
        )


        cursor = connection.execute(
            """
            UPDATE appointments
            SET
                patient_name = ?,
                status = 'booked'
            WHERE id = ?
            AND status = 'available'
            """,
            (
                current["patient_name"],
                new_appointment_id
            )
        )


        if cursor.rowcount == 0:

            connection.rollback()

            return {
                "success": False,
                "message": (
                    "The new appointment slot is no longer available."
                )
            }


        connection.commit()


        # --------------------------------------------------
        # FETCH CONFIRMED RESCHEDULE
        # --------------------------------------------------

        updated = connection.execute(
            """
            SELECT
                id,
                appointment_date,
                appointment_time,
                patient_name,
                status
            FROM appointments
            WHERE id = ?
            """,
            (new_appointment_id,)
        ).fetchone()


        if updated is None:

            connection.rollback()

            return {
                "success": False,
                "message": (
                    "The rescheduled appointment could not be confirmed."
                )
            }


        return {
            "success": True,
            "old_appointment_id": current["id"],
            "old_date": current["appointment_date"],
            "old_time": current["appointment_time"],
            "appointment_id": updated["id"],
            "date": updated["appointment_date"],
            "time": updated["appointment_time"],
            "patient_name": updated["patient_name"],
            "status": updated["status"]
        }


    except Exception:

        connection.rollback()

        return {
            "success": False,
            "message": (
                "The appointment could not be rescheduled "
                "because of a database error."
            )
        }


    finally:

        connection.close()      