import sqlite3
from pathlib import Path


DATABASE_PATH = Path("careflow.db")


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            appointment_date TEXT NOT NULL,
            appointment_time TEXT NOT NULL,
            patient_name TEXT,
            status TEXT NOT NULL DEFAULT 'available'
        )
        """
    )

    connection.commit()
    connection.close()


def seed_appointments():
    connection = get_connection()

    existing_count = connection.execute(
        "SELECT COUNT(*) FROM appointments"
    ).fetchone()[0]

    if existing_count == 0:
        appointments = [
            ("2026-08-26", "09:00"),
            ("2026-08-26", "10:00"),
            ("2026-08-26", "11:30"),
            ("2026-08-26", "14:00"),
            ("2026-08-26", "15:30"),
        ]

        connection.executemany(
            """
            INSERT INTO appointments (
                appointment_date,
                appointment_time
            )
            VALUES (?, ?)
            """,
            appointments
        )

        connection.commit()

    connection.close()