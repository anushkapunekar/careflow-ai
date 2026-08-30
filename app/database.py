import sqlite3
from pathlib import Path
from datetime import datetime, date


# ==================================================
# DATABASE CONFIGURATION
# ==================================================

DATABASE_PATH = Path("careflow.db")


# ==================================================
# DEFAULT CLINIC SCHEDULE
# ==================================================

DEFAULT_APPOINTMENT_TIMES = (
    "09:00",
    "10:00",
    "11:30",
    "14:00",
    "15:30",
)


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_connection():
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ==================================================
# DATABASE INITIALIZATION
# ==================================================

def initialize_database():

    connection = get_connection()

    try:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                appointment_date TEXT NOT NULL,

                appointment_time TEXT NOT NULL,

                patient_name TEXT,

                status TEXT NOT NULL
                    DEFAULT 'available',

                UNIQUE (
                    appointment_date,
                    appointment_time
                )
            )
            """
        )


        # ==================================================
        # CONVERSATION STATE
        # ==================================================
        #
        # Persistent state allows the application to survive
        # a server restart instead of losing booking context.
        #

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_state (

                conversation_id TEXT PRIMARY KEY,

                appointment_date TEXT,

                appointment_time TEXT,

                appointment_id INTEGER,

                patient_name TEXT,

                confirmed INTEGER NOT NULL
                    DEFAULT 0,

                updated_at TEXT NOT NULL,

                FOREIGN KEY (
                    appointment_id
                )
                REFERENCES appointments(id)
            )
            """
        )


        connection.commit()

    finally:

        connection.close()


# ==================================================
# SEED INITIAL DEMONSTRATION DATA
# ==================================================

def seed_appointments():
    """
    Create a small initial set of demonstration slots.

    Future dates are NOT hardcoded.

    Future weekday dates are created dynamically by
    ensure_appointment_slots().
    """

    connection = get_connection()

    try:

        existing_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM appointments
            """
        ).fetchone()[0]


        if existing_count == 0:

            appointments = [

                ("2026-08-27", "09:00"),
                ("2026-08-27", "10:00"),
                ("2026-08-27", "11:30"),
                ("2026-08-27", "14:00"),
                ("2026-08-27", "15:30"),

                ("2026-08-28", "09:00"),
                ("2026-08-28", "10:00"),
                ("2026-08-28", "11:30"),
                ("2026-08-28", "14:00"),
                ("2026-08-28", "15:30"),

            ]


            connection.executemany(
                """
                INSERT OR IGNORE INTO appointments (
                    appointment_date,
                    appointment_time
                )
                VALUES (?, ?)
                """,
                appointments
            )


            connection.commit()

    finally:

        connection.close()


# ==================================================
# VALIDATE APPOINTMENT DATE
# ==================================================

def validate_appointment_date(
    appointment_date: str
) -> date:

    try:

        parsed_date = datetime.strptime(
            appointment_date,
            "%Y-%m-%d"
        ).date()

    except ValueError:

        raise ValueError(
            "Appointment date must be in YYYY-MM-DD format."
        )


    return parsed_date


# ==================================================
# ENSURE DATE HAS APPOINTMENT SLOTS
# ==================================================

def ensure_appointment_slots(
    appointment_date: str
):
    """
    Dynamically create standard appointment slots
    for a valid future/current weekday.

    Existing appointments are never replaced.

    Weekends do not receive appointment slots.

    Past dates do not receive new appointment slots.
    """

    requested_date = validate_appointment_date(
        appointment_date
    )


    # --------------------------------------------------
    # DO NOT CREATE SLOTS IN THE PAST
    # --------------------------------------------------

    if requested_date < date.today():

        return


    # --------------------------------------------------
    # WEEKENDS HAVE NO APPOINTMENTS
    # --------------------------------------------------

    if requested_date.weekday() >= 5:

        return


    connection = get_connection()

    try:

        # --------------------------------------------------
        # CHECK WHETHER DATE ALREADY HAS SLOTS
        # --------------------------------------------------

        existing_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM appointments
            WHERE appointment_date = ?
            """,
            (
                appointment_date,
            )
        ).fetchone()[0]


        if existing_count > 0:

            return


        # --------------------------------------------------
        # CREATE STANDARD CLINIC SLOTS
        # --------------------------------------------------

        appointments = [

            (
                appointment_date,
                appointment_time
            )

            for appointment_time
            in DEFAULT_APPOINTMENT_TIMES

        ]


        connection.executemany(
            """
            INSERT OR IGNORE INTO appointments (
                appointment_date,
                appointment_time
            )
            VALUES (?, ?)
            """,
            appointments
        )


        connection.commit()

    finally:

        connection.close()


# ==================================================
# SAVE CONVERSATION STATE
# ==================================================

def save_conversation_state(
    conversation_id: str,
    appointment_date: str | None = None,
    appointment_time: str | None = None,
    appointment_id: int | None = None,
    patient_name: str | None = None,
    confirmed: bool = False
):

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO conversation_state (
                conversation_id,
                appointment_date,
                appointment_time,
                appointment_id,
                patient_name,
                confirmed,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(conversation_id)
            DO UPDATE SET

                appointment_date =
                    excluded.appointment_date,

                appointment_time =
                    excluded.appointment_time,

                appointment_id =
                    excluded.appointment_id,

                patient_name =
                    excluded.patient_name,

                confirmed =
                    excluded.confirmed,

                updated_at =
                    excluded.updated_at
            """,
            (
                conversation_id,
                appointment_date,
                appointment_time,
                appointment_id,
                patient_name,
                1 if confirmed else 0,
                datetime.utcnow().isoformat()
            )
        )


        connection.commit()

    finally:

        connection.close()


# ==================================================
# GET CONVERSATION STATE
# ==================================================

def get_conversation_state(
    conversation_id: str
) -> dict:

    connection = get_connection()

    try:

        state = connection.execute(
            """
            SELECT
                conversation_id,
                appointment_date,
                appointment_time,
                appointment_id,
                patient_name,
                confirmed,
                updated_at
            FROM conversation_state
            WHERE conversation_id = ?
            """,
            (
                conversation_id,
            )
        ).fetchone()

    finally:

        connection.close()


    if state is None:

        return {}


    return {

        "conversation_id":
            state["conversation_id"],

        "appointment_date":
            state["appointment_date"],

        "appointment_time":
            state["appointment_time"],

        "appointment_id":
            state["appointment_id"],

        "patient_name":
            state["patient_name"],

        "confirmed":
            bool(state["confirmed"]),

        "updated_at":
            state["updated_at"]

    }


# ==================================================
# CLEAR CONVERSATION STATE
# ==================================================

def clear_conversation_state(
    conversation_id: str
):

    connection = get_connection()

    try:

        connection.execute(
            """
            DELETE FROM conversation_state
            WHERE conversation_id = ?
            """,
            (
                conversation_id,
            )
        )

        connection.commit()

    finally:

        connection.close()