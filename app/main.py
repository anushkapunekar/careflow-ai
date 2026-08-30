import json
import logging
import os
import re
from datetime import date, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.rag import (
    build_index,
    search_knowledge,
)

from app.llm import (
    ask_llm,
    SYSTEM_PROMPT,
    TOOLS,
)

from app.tools import (
    check_appointment_availability,
    get_available_appointments,
    book_appointment,
)

from app.database import (
    initialize_database,
    seed_appointments,
)


# ==================================================
# ENVIRONMENT CONFIGURATION
# ==================================================

load_dotenv()

ADMIN_API_KEY = os.getenv("CARE_FLOW_ADMIN_KEY")

if not ADMIN_API_KEY:
    raise RuntimeError(
        "CARE_FLOW_ADMIN_KEY is not configured. "
        "Add it to the project's .env file."
    )


# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("careflow")


# ==================================================
# APPLICATION
# ==================================================

app = FastAPI(
    title="CareFlow AI"
)


# ==================================================
# DATABASE INITIALIZATION
# ==================================================

initialize_database()
seed_appointments()


# ==================================================
# REQUEST MODELS
# ==================================================

class ChatRequest(BaseModel):

    conversation_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )


# ==================================================
# CONVERSATION MEMORY
# ==================================================

conversations: dict[str, list[dict]] = {}


# ==================================================
# APPOINTMENT SESSION STATE
# ==================================================
#
# This is deliberately kept outside the LLM.
#
# The backend, not the model, owns:
#
# - requested date
# - selected appointment
# - confirmation state
# - patient name
#
# This makes booking deterministic and much safer.
# ==================================================

booking_states: dict[str, dict] = {}


# Example:
#
# {
#     "voice-123": {
#         "date": "2026-09-16",
#         "appointment_id": 44,
#         "time": "14:00",
#         "confirmed": False,
#         "patient_name": None
#     }
# }


# ==================================================
# VERIFIED AVAILABLE SLOTS
# ==================================================
#
# These are appointment IDs actually returned by
# the database during this conversation.
# ==================================================

available_slots: dict[str, set[int]] = {}


# ==================================================
# CONVERSATION SAFETY LIMITS
# ==================================================

MAX_CONVERSATIONS = 500

MAX_MESSAGES_PER_CONVERSATION = 10

MAX_TRACKED_SLOTS_PER_CONVERSATION = 50


# ==================================================
# DATE HELPERS
# ==================================================

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


# ==================================================
# CONVERSATION LIMITS
# ==================================================

def enforce_conversation_limits(
    conversation_id: str
):
    conversation = conversations.get(
        conversation_id
    )

    if not conversation:
        return

    if len(conversation) <= MAX_MESSAGES_PER_CONVERSATION:
        return

    system_message = conversation[0]

    recent_messages = conversation[
        -(MAX_MESSAGES_PER_CONVERSATION - 1):
    ]

    conversations[
        conversation_id
    ] = [
        system_message,
        *recent_messages,
    ]


def enforce_slot_limits(
    conversation_id: str
):
    slots = available_slots.get(
        conversation_id
    )

    if not slots:
        return

    if len(slots) <= MAX_TRACKED_SLOTS_PER_CONVERSATION:
        return

    available_slots[
        conversation_id
    ] = set(
        list(slots)[
            -MAX_TRACKED_SLOTS_PER_CONVERSATION:
        ]
    )


# ==================================================
# BOOKING STATE
# ==================================================

def get_booking_state(
    conversation_id: str
) -> dict:

    if conversation_id not in booking_states:

        booking_states[
            conversation_id
        ] = {
            "date": None,
            "appointment_id": None,
            "time": None,
            "confirmed": False,
            "patient_name": None,
        }

    return booking_states[
        conversation_id
    ]


def clear_booking_state(
    conversation_id: str
):
    booking_states[
        conversation_id
    ] = {
        "date": None,
        "appointment_id": None,
        "time": None,
        "confirmed": False,
        "patient_name": None,
    }


# ==================================================
# DATE RESOLUTION
# ==================================================

def resolve_date(
    message: str
) -> str | None:

    normalized = message.lower().strip()

    today = date.today()


    # --------------------------------------------------
    # TODAY
    # --------------------------------------------------

    if re.search(
        r"\btoday\b",
        normalized,
    ):

        return today.isoformat()


    # --------------------------------------------------
    # TOMORROW
    # --------------------------------------------------

    if re.search(
        r"\btomorrow\b",
        normalized,
    ):

        return (
            today +
            timedelta(days=1)
        ).isoformat()


    # --------------------------------------------------
    # DAY AFTER TOMORROW
    # --------------------------------------------------

    if (
        "day after tomorrow" in normalized
        or re.search(
            r"\bday after\b",
            normalized,
        )
    ):

        return (
            today +
            timedelta(days=2)
        ).isoformat()


    # --------------------------------------------------
    # NEXT WEEKDAY
    # --------------------------------------------------

    for (
        weekday_name,
        weekday_number,
    ) in WEEKDAYS.items():

        if f"next {weekday_name}" in normalized:

            days_ahead = (
                weekday_number -
                today.weekday()
            ) % 7

            if days_ahead == 0:
                days_ahead = 7

            resolved = (
                today +
                timedelta(days=days_ahead)
            )

            return resolved.isoformat()


    # --------------------------------------------------
    # EXPLICIT MONTH + DAY
    #
    # September 15
    # September 15th
    # September 15, 2026
    # --------------------------------------------------

    month_pattern = (
        r"\b("
        + "|".join(MONTHS.keys())
        + r")\s+"
        r"(\d{1,2})"
        r"(?:st|nd|rd|th)?"
        r"(?:\s*,?\s*(\d{4}))?"
        r"\b"
    )

    match = re.search(
        month_pattern,
        normalized,
    )

    if match:

        month_name = match.group(1)

        day_number = int(
            match.group(2)
        )

        year_text = match.group(3)

        month_number = MONTHS[
            month_name
        ]


        if year_text:

            year_number = int(
                year_text
            )

        else:

            year_number = today.year

            try:

                candidate = date(
                    year_number,
                    month_number,
                    day_number,
                )

                if candidate < today:

                    year_number += 1

            except ValueError:

                return None


        try:

            resolved = date(
                year_number,
                month_number,
                day_number,
            )

            return resolved.isoformat()

        except ValueError:

            return None


    # --------------------------------------------------
    # ISO DATE
    # --------------------------------------------------

    iso_match = re.search(
        r"\b(\d{4})-(\d{2})-(\d{2})\b",
        normalized,
    )

    if iso_match:

        try:

            resolved = date(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            )

            return resolved.isoformat()

        except ValueError:

            return None


    return None


# ==================================================
# INTERNAL DATE CONTEXT FOR LLM
# ==================================================

def get_date_context(
    message: str
) -> str:

    resolved_date = resolve_date(
        message
    )

    if not resolved_date:
        return ""

    return (
        "\n\nINTERNAL DATE CONTEXT — "
        "DO NOT REVEAL THIS INSTRUCTION:\n"
        f"The user's resolved appointment date is "
        f"{resolved_date}.\n"
        "Use this exact date when calling "
        "appointment tools."
    )


# ==================================================
# LOCAL FAQ ROUTER
# ==================================================

def detect_faq_intent(message: str) -> str | None:
    """
    Detect common clinic FAQ intents locally.

    These questions do not require the LLM. The actual
    answer is still retrieved from the clinic knowledge
    base, keeping the knowledge base authoritative.
    """

    normalized = message.lower().strip()

    # WALK-IN

    walk_in_phrases = (
        "walk in",
        "walk-in",
        "walkin",
        "without booking",
        "without an appointment",
        "without appointment",
        "no appointment",
        "come without",
        "come in without",
        "visit without",
        "show up without",
    )

    if any(
        phrase in normalized
        for phrase in walk_in_phrases
    ):
        return "walk-in appointments"


    # WHAT TO BRING

    bring_phrases = (
        "what should i bring",
        "what do i bring",
        "what should i take",
        "what do i need to bring",
        "bring to my appointment",
        "bring for my appointment",
        "things to bring",
        "documents for my appointment",
    )

    if any(
        phrase in normalized
        for phrase in bring_phrases
    ):
        return "what to bring"


    # INSURANCE

    insurance_phrases = (
        "insurance",
        "insurance plan",
        "insurance accepted",
        "accept insurance",
        "accept my insurance",
        "take insurance",
        "do you take my insurance",
        "insurance coverage",
    )

    if any(
        phrase in normalized
        for phrase in insurance_phrases
    ):
        return "insurance"


    # APPOINTMENT PREPARATION

    preparation_phrases = (
        "how early should i arrive",
        "when should i arrive",
        "how long before my appointment",
        "how early do i need to arrive",
        "arrive before my appointment",
        "appointment preparation",
        "prepare for my appointment",
    )

    if any(
        phrase in normalized
        for phrase in preparation_phrases
    ):
        return "appointment preparation"


    # CLINIC HOURS

    hours_phrases = (
        "clinic hours",
        "office hours",
        "what time do you open",
        "what time do you close",
        "when are you open",
        "when is the clinic open",
        "are you open",
    )

    if any(
        phrase in normalized
        for phrase in hours_phrases
    ):
        return "clinic hours"


    # SERVICES

    service_phrases = (
        "what services",
        "what do you offer",
        "services do you provide",
        "what kind of appointments",
        "what type of appointments",
        "what can i book",
    )

    if any(
        phrase in normalized
        for phrase in service_phrases
    ):
        return "services"


    # CANCELLATION

    cancellation_phrases = (
        "cancel my appointment",
        "cancel an appointment",
        "cancel appointment",
        "cancellation",
    )

    if any(
        phrase in normalized
        for phrase in cancellation_phrases
    ):
        return "cancellation"


    # RESCHEDULING

    reschedule_phrases = (
        "reschedule my appointment",
        "reschedule an appointment",
        "reschedule appointment",
        "change my appointment",
        "move my appointment",
    )

    if any(
        phrase in normalized
        for phrase in reschedule_phrases
    ):
        return "rescheduling"


    return None


# ==================================================
# TIME RESOLUTION
# ==================================================

def resolve_time(
    message: str
) -> str | None:

    normalized = (
        message
        .lower()
        .strip()
    )


    # --------------------------------------------------
    # 11:30 AM
    # 11:30 a.m.
    # --------------------------------------------------

    match = re.search(
        r"\b("
        r"\d{1,2}"
        r")\s*:\s*"
        r"(\d{2})"
        r"\s*"
        r"(a\.?m\.?|p\.?m\.?)"
        r"\b",
        normalized,
    )

    if match:

        hour = int(
            match.group(1)
        )

        minute = int(
            match.group(2)
        )

        period = (
            match.group(3)
            .replace(".", "")
        )

        if hour < 1 or hour > 12:
            return None

        if minute < 0 or minute > 59:
            return None

        if period == "pm" and hour != 12:
            hour += 12

        if period == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"


    # --------------------------------------------------
    # 11 AM
    # 2 PM
    # --------------------------------------------------

    match = re.search(
        r"\b("
        r"\d{1,2}"
        r")\s*"
        r"(a\.?m\.?|p\.?m\.?)"
        r"\b",
        normalized,
    )

    if match:

        hour = int(
            match.group(1)
        )

        period = (
            match.group(2)
            .replace(".", "")
        )

        if hour < 1 or hour > 12:
            return None

        if period == "pm" and hour != 12:
            hour += 12

        if period == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:00"


    # --------------------------------------------------
    # Bare hour:
    #
    # "11:30"
    # --------------------------------------------------

    match = re.search(
        r"\b(\d{1,2})\s*:\s*(\d{2})\b",
        normalized,
    )

    if match:

        hour = int(
            match.group(1)
        )

        minute = int(
            match.group(2)
        )

        if (
            0 <= hour <= 23
            and 0 <= minute <= 59
        ):

            return f"{hour:02d}:{minute:02d}"


    return None


# ==================================================
# YES / CONFIRMATION DETECTION
# ==================================================

def is_confirmation(
    message: str
) -> bool:

    normalized = (
        message
        .lower()
        .strip()
    )

    confirmation_patterns = [

        r"^yes$",

        r"^yes[!.]?$",

        r"^yes go ahead$",

        r"^go ahead$",

        r"^book it$",

        r"^book$",

        r"^sure$",

        r"^sure thing$",

        r"^that's correct$",

        r"^that is correct$",

        r"^correct$",

        r"^please do$",

        r"^do it$",

        r"^absolutely$",

        r"^yep$",

        r"^yeah$",

        r"^yeah go ahead$",
    ]

    return any(
        re.fullmatch(
            pattern,
            normalized,
        )
        for pattern
        in confirmation_patterns
    )


# ==================================================
# NO / NEGATIVE DETECTION
# ==================================================

def is_negative(
    message: str
) -> bool:

    normalized = (
        message
        .lower()
        .strip()
    )

    return normalized in {
        "no",
        "nope",
        "nah",
        "not now",
        "cancel",
        "never mind",
        "nevermind",
    }


# ==================================================
# PATIENT NAME DETECTION
# ==================================================

def looks_like_name(
    message: str
) -> bool:

    value = message.strip()

    if not value:
        return False

    normalized = value.lower()

    invalid = {
        "yes",
        "no",
        "nope",
        "okay",
        "ok",
        "sure",
        "go ahead",
        "book it",
        "book",
        "patient",
        "patient name",
        "name",
        "unknown",
        "none",
        "null",
        "n/a",
        "na",
        "tomorrow",
        "today",
        "morning",
        "afternoon",
        "evening",
    }

    if normalized in invalid:
        return False

    if len(value) > 100:
        return False

    # Names should not contain numbers.

    if re.search(
        r"\d",
        value,
    ):
        return False

    # Avoid treating questions as names.

    if "?" in value:
        return False

    # Allow normal alphabetic names with spaces,
    # apostrophes and hyphens.

    if not re.fullmatch(
        r"[A-Za-z][A-Za-z '\-]{0,98}",
        value,
    ):
        return False

    return True


# ==================================================
# EMERGENCY DETECTION
# ==================================================

def is_emergency(
    message: str
) -> bool:

    normalized = (
        message
        .lower()
        .strip()
    )

    emergency_terms = [

        "chest pain",

        "difficulty breathing",

        "can't breathe",

        "cannot breathe",

        "trouble breathing",

        "shortness of breath",

        "severe bleeding",

        "unconscious",

        "passed out",

        "stroke",

        "seizure",

        "heart attack",

        "critical patient",

        "patient is critical",

        "patient critical",

        "life threatening",

        "life-threatening",

        "emergency",
    ]

    return any(
        term in normalized
        for term in emergency_terms
    )


# ==================================================
# CLINIC KNOWLEDGE DETECTION
# ==================================================

def looks_like_clinic_question(
    message: str
) -> bool:

    normalized = (
        message
        .lower()
        .strip()
    )

    clinic_terms = [

        "walk in",

        "walk-in",

        "walkins",

        "walk-ins",

        "insurance",

        "bring",

        "what should i bring",

        "documents",

        "paperwork",

        "clinic hours",

        "opening hours",

        "closing time",

        "open",

        "close",

        "address",

        "location",

        "where are you",

        "where is the clinic",

        "services",

        "service",

        "procedure",

        "preparation",

        "prepare",

        "prescription",

        "medicine",

        "medication",

        "payment",

        "cash",

        "credit card",

        "parking",

        "parking lot",

        "telehealth",

        "virtual appointment",

        "doctor",

        "provider",
    ]

    return any(
        term in normalized
        for term in clinic_terms
    )


# ==================================================
# FORMAT RAG RESPONSE
# ==================================================

def format_knowledge_response(
    results: list
) -> str | None:

    if not results:
        return None

    first = results[0]

    content = first.get(
        "content",
        ""
    )

    if not content:
        return None

    content = content.strip()

    # Remove obvious document headings.

    lines = content.splitlines()

    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        # Skip all-uppercase heading lines.

        if (
            stripped.isupper()
            and len(stripped) < 80
        ):
            continue

        cleaned_lines.append(
            stripped
        )

    if not cleaned_lines:
        return content

    return " ".join(
        cleaned_lines
    )


# ==================================================
# DIRECT CLINIC KNOWLEDGE LOOKUP
# ==================================================

def answer_from_knowledge(
    message: str
) -> str | None:

    try:

        results = search_knowledge(
            query=message
        )

        return format_knowledge_response(
            results
        )

    except Exception:

        logger.exception(
            "Knowledge search failed"
        )

        return None


# ==================================================
# ADMIN AUTHENTICATION
# ==================================================

def verify_admin_key(
    x_admin_key: str | None
):

    if not x_admin_key:

        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    if x_admin_key != ADMIN_API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy"
    }


# ==================================================
# LOCAL APPOINTMENT HANDLER
# ==================================================
#
# This is the biggest performance improvement.
#
# Common appointment operations do NOT require Groq.
# ==================================================

def handle_local_appointment_request(
    conversation_id: str,
    message: str
) -> str | None:

    normalized = (
        message
        .lower()
        .strip()
    )

    state = get_booking_state(
        conversation_id
    )


    # ==================================================
    # CANCELLATION
    # ==================================================

    if (
        "cancel my appointment" in normalized
        or "cancel an appointment" in normalized
        or normalized == "cancel appointment"
    ):

        return (
            "Appointment cancellation is not currently "
            "supported by CareFlow AI."
        )


    # ==================================================
    # EXPLICIT DATE
    # ==================================================

    resolved_date = resolve_date(
        message
    )

    if resolved_date:

        state["date"] = resolved_date

        # Changing the date invalidates the previous
        # selected appointment.

        state["appointment_id"] = None
        state["time"] = None
        state["confirmed"] = False
        state["patient_name"] = None


    # ==================================================
    # USER WANTS TO BOOK
    # ==================================================

    booking_request = any(
        phrase in normalized
        for phrase in [
            "book an appointment",
            "book appointment",
            "schedule an appointment",
            "schedule appointment",
            "make an appointment",
            "i want an appointment",
            "i need an appointment",
            "book me",
            "schedule me",
        ]
    )


    if booking_request:

        if not state["date"]:

            return (
                "I'd be happy to help you schedule "
                "an appointment. What date works best "
                "for you?"
            )

        return (
            "What time would you prefer for your "
            "appointment?"
        )


    # ==================================================
    # NO DATE YET
    # ==================================================

    if not state["date"]:

        return None


    # ==================================================
    # USER ASKS FOR AVAILABLE TIMES
    # ==================================================

    availability_request = any(
        phrase in normalized
        for phrase in [
            "what times are available",
            "what time is available",
            "what times do you have",
            "available times",
            "availability",
            "what appointments are available",
            "what appointment times",
            "show available times",
            "show me the available times",
            "what slots are available",
            "available slots",
        ]
    )


    if availability_request:

        result = get_available_appointments(
            date=state["date"]
        )

        logger.info(
            "LOCAL AVAILABILITY RESULT: %s",
            result
        )


        if not result.get("available"):

            return (
                f"There are no available appointment "
                f"slots on "
                f"{format_spoken_date(state['date'])}."
            )


        slots = result.get(
            "slots",
            []
        )

        times = [
            format_spoken_time(
                slot["time"]
            )
            for slot in slots
        ]


        # Store every verified slot.

        available_slots[
            conversation_id
        ] = {
            slot["appointment_id"]
            for slot in slots
        }


        enforce_slot_limits(
            conversation_id
        )


        return (
            f"I have {format_time_list(times)} "
            f"available on "
            f"{format_spoken_date(state['date'])}. "
            f"Which time works best for you?"
        )


    # ==================================================
    # SPECIFIC TIME
    # ==================================================

    resolved_time = resolve_time(
        message
    )

    if resolved_time:

        result = check_appointment_availability(
            date=state["date"],
            preferred_time=resolved_time
        )

        logger.info(
            "LOCAL TIME CHECK RESULT: %s",
            result
        )


        if not result.get("available"):

            # If the exact time doesn't exist,
            # give the actual available options.

            available = get_available_appointments(
                date=state["date"]
            )

            if available.get("available"):

                times = [
                    format_spoken_time(
                        slot["time"]
                    )
                    for slot
                    in available.get(
                        "slots",
                        []
                    )
                ]

                available_slots[
                    conversation_id
                ] = {
                    slot["appointment_id"]
                    for slot
                    in available.get(
                        "slots",
                        []
                    )
                }

                return (
                    f"I don't have a slot at "
                    f"{format_spoken_time(resolved_time)} "
                    f"on "
                    f"{format_spoken_date(state['date'])}. "
                    f"The available times are "
                    f"{format_time_list(times)}."
                )

            return (
                f"There are no available appointment "
                f"slots on "
                f"{format_spoken_date(state['date'])}."
            )


        appointment_id = result.get(
            "appointment_id"
        )


        state["appointment_id"] = (
            appointment_id
        )

        state["time"] = (
            result.get("time")
        )

        state["confirmed"] = False
        state["patient_name"] = None


        available_slots[
            conversation_id
        ].add(
            appointment_id
        )


        return (
            f"Just to confirm, would you like me "
            f"to book an appointment for you on "
            f"{format_spoken_date(state['date'])} "
            f"at "
            f"{format_spoken_time(state['time'])}?"
        )


    # ==================================================
    # CONFIRMATION
    # ==================================================

    if is_confirmation(message):

        if (
            state["date"]
            and state["appointment_id"] is not None
            and state["time"]
        ):

            state["confirmed"] = True

            if not state["patient_name"]:

                return (
                    "Absolutely. What is your full name?"
                )


    # ==================================================
    # PATIENT NAME
    # ==================================================

    if (
        state["confirmed"]
        and not state["patient_name"]
        and looks_like_name(message)
    ):

        patient_name = message.strip()

        state["patient_name"] = (
            patient_name
        )


        appointment_id = (
            state["appointment_id"]
        )


        # --------------------------------------------------
        # FINAL BACKEND VALIDATION
        # --------------------------------------------------

        if (
            appointment_id is None
            or appointment_id
            not in available_slots.get(
                conversation_id,
                set()
            )
        ):

            state["appointment_id"] = None
            state["time"] = None
            state["confirmed"] = False
            state["patient_name"] = None

            return (
                "That appointment slot is no longer "
                "confirmed as available. Please choose "
                "another available time."
            )


        # --------------------------------------------------
        # ACTUAL DATABASE BOOKING
        # --------------------------------------------------

        result = book_appointment(
            appointment_id=appointment_id,
            patient_name=patient_name
        )


        logger.info(
            "LOCAL BOOKING RESULT: %s",
            result
        )


        if result.get("success"):

            available_slots[
                conversation_id
            ].discard(
                appointment_id
            )


            booked_date = (
                result.get("date")
                or state["date"]
            )

            booked_time = (
                result.get("time")
                or state["time"]
            )


            clear_booking_state(
                conversation_id
            )


            return (
                f"Your appointment for "
                f"{format_spoken_date(booked_date)} "
                f"at "
                f"{format_spoken_time(booked_time)} "
                f"has been booked, {patient_name}."
            )


        # --------------------------------------------------
        # BOOKING FAILED
        # --------------------------------------------------

        clear_booking_state(
            conversation_id
        )

        return (
            "I couldn't book that appointment because "
            "the slot is no longer available."
        )


    # ==================================================
    # USER SAYS NO TO CONFIRMATION
    # ==================================================

    if (
        is_negative(message)
        and state["appointment_id"] is not None
        and not state["confirmed"]
    ):

        state["appointment_id"] = None
        state["time"] = None
        state["confirmed"] = False
        state["patient_name"] = None

        return (
            "No problem. What time would you prefer instead?"
        )


    return None


# ==================================================
# SPOKEN DATE FORMAT
# ==================================================

def format_spoken_date(
    iso_date: str
) -> str:

    try:

        parsed = date.fromisoformat(
            iso_date
        )

        suffix = get_ordinal_suffix(
            parsed.day
        )

        return (
            f"{parsed.strftime('%B')} "
            f"{parsed.day}{suffix}"
        )

    except Exception:

        return iso_date


def get_ordinal_suffix(
    day: int
) -> str:

    if 10 <= day % 100 <= 20:
        return "th"

    return {
        1: "st",
        2: "nd",
        3: "rd",
    }.get(
        day % 10,
        "th"
    )


# ==================================================
# SPOKEN TIME FORMAT
# ==================================================

def format_spoken_time(
    time_value: str
) -> str:

    try:

        hour, minute = map(
            int,
            time_value.split(":")
        )

        if hour == 0:

            spoken_hour = 12
            period = "AM"

        elif hour < 12:

            spoken_hour = hour
            period = "AM"

        elif hour == 12:

            spoken_hour = 12
            period = "PM"

        else:

            spoken_hour = hour - 12
            period = "PM"


        if minute == 0:

            return (
                f"{spoken_hour} {period}"
            )

        return (
            f"{spoken_hour}:{minute:02d} {period}"
        )

    except Exception:

        return time_value


# ==================================================
# TIME LIST FORMATTER
# ==================================================

def format_time_list(
    times: list[str]
) -> str:

    if not times:
        return ""

    if len(times) == 1:
        return times[0]

    if len(times) == 2:

        return (
            f"{times[0]} and {times[1]}"
        )

    return (
        ", ".join(times[:-1])
        + ", and "
        + times[-1]
    )


# ==================================================
# CHAT ENDPOINT
# ==================================================

@app.post("/api/chat")
def chat(
    request: ChatRequest
):

    conversation_id = (
        request.conversation_id
    )

    message = (
        request.message
        .strip()
    )


    try:

        # ==================================================
        # LOCAL FAQ ROUTER
        # ==================================================
        #
        # Common clinic FAQ questions are handled locally.
        # This avoids unnecessary LLM calls and therefore
        # avoids Groq tokens/rate limits.
        #

        faq_intent = detect_faq_intent(
            request.message
        )

        if faq_intent:

            logger.info(
                "LOCAL FAQ ROUTER: intent=%s",
                faq_intent
            )


        # ==================================================
        # LOCAL FAQ ANSWERS
        # ==================================================

        FAQ_ANSWERS = {

            "walk-in appointments": (
                "Walk-in appointments are not currently "
                "supported by the clinic. Patients should "
                "schedule an appointment through CareFlow AI."
            ),

            "what to bring": (
                "Patients should bring a valid photo ID and "
                "any relevant medical records or medication "
                "information needed for their visit."
            ),

            "insurance": (
                "The clinic accepts selected insurance plans. "
                "Insurance coverage depends on the patient's "
                "individual plan. Patients should verify their "
                "coverage before the appointment."
            ),

            "clinic hours": (
                "The clinic is open Monday through Friday "
                "from 9 AM to 5 PM. The clinic is closed on "
                "Saturdays and Sundays."
            ),

            "appointment preparation": (
                "Patients should arrive approximately 10 minutes "
                "before their scheduled appointment."
            ),

            "cancellation": (
                "Appointment cancellation is not currently "
                "supported by the CareFlow AI prototype."
            ),

            "rescheduling": (
                "Appointment rescheduling is not currently "
                "supported by the CareFlow AI prototype."
            ),

            "location": (
                "The CareFlow Demo Clinic does not have a "
                "real-world physical location. It is a fictional "
                "demonstration clinic."
            ),

            "human assistance": (
                "The current CareFlow AI prototype does not have "
                "a live human handoff or phone-transfer capability."
            ),

            "reminders": (
                "The current CareFlow AI prototype does not "
                "provide SMS, email, telephone, or other "
                "appointment reminders."
            ),

            "patient records": (
                "The current prototype does not have access to "
                "real patient medical records or external "
                "healthcare systems."
            ),

            "prescriptions": (
                "CareFlow AI does not prescribe medication or "
                "provide prescription recommendations."
            ),
        }


        faq_answer = FAQ_ANSWERS.get(
            faq_intent
        )


        if faq_answer:

            logger.info(
                "LOCAL FAQ ANSWER: intent=%s",
                faq_intent
            )

            return {
                "conversation_id":
                    request.conversation_id,

                "response":
                    faq_answer
            }


        if faq_intent:

            logger.warning(
                "FAQ intent detected but no "
                "answer configured: %s",
                faq_intent
            )

            return {
                "conversation_id":
                    request.conversation_id,

                "response": (
                    "That information is not currently "
                    "available in the CareFlow AI prototype."
                )
            }


        # ==================================================
        # CREATE CONVERSATION
        # ==================================================

        if conversation_id not in conversations:

            if (
                len(conversations)
                >= MAX_CONVERSATIONS
            ):

                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Conversation capacity "
                        "temporarily reached."
                    )
                )


            conversations[
                conversation_id
            ] = [

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }

            ]


        # ==================================================
        # CREATE STATE
        # ==================================================

        get_booking_state(
            conversation_id
        )


        if conversation_id not in available_slots:

            available_slots[
                conversation_id
            ] = set()


        # ==================================================
        # MEMORY LIMIT
        # ==================================================

        enforce_conversation_limits(
            conversation_id
        )


        conversation = conversations[
            conversation_id
        ]


        # ==================================================
        # STORE USER MESSAGE
        # ==================================================

        conversation.append(

            {
                "role": "user",
                "content": message,
            }

        )


        # ==================================================
        # SAFETY FIRST
        # ==================================================

        if is_emergency(message):

            response_text = (
                "Chest pain or other serious symptoms "
                "can be a medical emergency. Please seek "
                "immediate medical attention by calling "
                "emergency services or going to the nearest "
                "emergency department. Do not wait for an "
                "appointment."
            )

            conversation.append(

                {
                    "role": "assistant",
                    "content": response_text,
                }

            )

            return {
                "conversation_id":
                    conversation_id,

                "response":
                    response_text
            }


        # ==================================================
        # LOCAL APPOINTMENT ROUTER
        # ==================================================

        local_response = (
            handle_local_appointment_request(
                conversation_id,
                message
            )
        )


        if local_response is not None:

            logger.info(
                "LOCAL ROUTER handled request for %s",
                conversation_id
            )


            response_text = (
                local_response
            )


            conversation.append(

                {
                    "role": "assistant",
                    "content": response_text,
                }

            )


            enforce_conversation_limits(
                conversation_id
            )


            return {

                "conversation_id":
                    conversation_id,

                "response":
                    response_text
            }


        # ==================================================
        # DIRECT RAG ROUTER
        # ==================================================
        #
        # Clinic-specific questions go directly to the
        # knowledge base instead of consuming Groq tokens.
        # ==================================================

        if looks_like_clinic_question(
            message
        ):

            knowledge_response = (
                answer_from_knowledge(
                    message
                )
            )


            if knowledge_response:

                response_text = (
                    knowledge_response
                )

            else:

                response_text = (
                    "That information is not currently "
                    "available in the CareFlow AI prototype."
                )


            conversation.append(

                {
                    "role": "assistant",
                    "content": response_text,
                }

            )


            enforce_conversation_limits(
                conversation_id
            )


            return {

                "conversation_id":
                    conversation_id,

                "response":
                    response_text
            }


        # ==================================================
        # LLM FALLBACK
        # ==================================================
        #
        # Only genuinely conversational / unknown requests
        # reach Groq.
        # ==================================================

        date_context = get_date_context(
            message
        )


        llm_message = message


        if date_context:

            llm_message += date_context


        # Replace the just-added user message with the
        # internally enriched version for the LLM.

        conversation[-1] = {

            "role": "user",

            "content": llm_message,
        }


        try:

            response = ask_llm(
                conversation,
                tools=TOOLS
            )

        except Exception:

            logger.exception(
                "LLM request failed for conversation %s",
                conversation_id
            )


            response_text = (
                "I'm sorry, I couldn't process that "
                "right now. Please try again."
            )


            conversation.append(

                {
                    "role": "assistant",
                    "content": response_text,
                }

            )


            return {

                "conversation_id":
                    conversation_id,

                "response":
                    response_text
            }


        # ==================================================
        # LLM TOOL CALLS
        # ==================================================

        if response.tool_calls:

            conversation.append(

                {
                    "role": "assistant",

                    "content":
                        response.content or "",

                    "tool_calls": [

                        {
                            "id":
                                tool_call.id,

                            "type":
                                "function",

                            "function": {

                                "name":
                                    tool_call.function.name,

                                "arguments":
                                    tool_call.function.arguments,
                            }

                        }

                        for tool_call
                        in response.tool_calls
                    ]
                }
            )


            # --------------------------------------------------
            # EXECUTE TOOLS
            # --------------------------------------------------

            for tool_call in response.tool_calls:

                tool_name = (
                    tool_call.function.name
                )


                try:

                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):

                    logger.exception(
                        "Invalid tool arguments: %s",
                        tool_name
                    )


                    result = {

                        "success":
                            False,

                        "message":
                            "Invalid tool arguments."
                    }


                    conversation.append(

                        {
                            "role":
                                "tool",

                            "tool_call_id":
                                tool_call.id,

                            "content":
                                json.dumps(result),
                        }
                    )

                    continue


                try:

                    # ==================================================
                    # CHECK SPECIFIC APPOINTMENT
                    # ==================================================

                    if tool_name == (
                        "check_appointment_availability"
                    ):

                        result = (
                            check_appointment_availability(
                                date=arguments["date"],
                                preferred_time=arguments[
                                    "preferred_time"
                                ]
                            )
                        )


                        if result.get(
                            "available"
                        ):

                            appointment_id = (
                                result.get(
                                    "appointment_id"
                                )
                            )


                            if appointment_id is not None:

                                available_slots[
                                    conversation_id
                                ].add(
                                    appointment_id
                                )


                                state = get_booking_state(
                                    conversation_id
                                )


                                state["date"] = (
                                    result.get("date")
                                )

                                state["appointment_id"] = (
                                    appointment_id
                                )

                                state["time"] = (
                                    result.get("time")
                                )

                                state["confirmed"] = False


                    # ==================================================
                    # GET AVAILABLE APPOINTMENTS
                    # ==================================================

                    elif tool_name == (
                        "get_available_appointments"
                    ):

                        result = (
                            get_available_appointments(
                                date=arguments["date"]
                            )
                        )


                        logger.info(
                            "AVAILABILITY TOOL RESULT: %s",
                            result
                        )


                        if result.get(
                            "available"
                        ):

                            slots = result.get(
                                "slots",
                                []
                            )


                            for slot in slots:

                                appointment_id = (
                                    slot.get(
                                        "appointment_id"
                                    )
                                )


                                if appointment_id is not None:

                                    available_slots[
                                        conversation_id
                                    ].add(
                                        appointment_id
                                    )


                            state = get_booking_state(
                                conversation_id
                            )

                            state["date"] = (
                                arguments["date"]
                            )


                            enforce_slot_limits(
                                conversation_id
                            )


                    # ==================================================
                    # BOOK APPOINTMENT
                    # ==================================================

                    elif tool_name == (
                        "book_appointment"
                    ):

                        patient_name = (
                            arguments.get(
                                "patient_name"
                            )
                        )


                        state = get_booking_state(
                            conversation_id
                        )


                        appointment_id = (
                            state.get(
                                "appointment_id"
                            )
                        )


                        # Only use backend-verified IDs.

                        if appointment_id is None:

                            candidate_id = (
                                arguments.get(
                                    "appointment_id"
                                )
                            )


                            if candidate_id in (
                                available_slots.get(
                                    conversation_id,
                                    set()
                                )
                            ):

                                appointment_id = (
                                    candidate_id
                                )


                        if (
                            not isinstance(
                                patient_name,
                                str
                            )
                            or not patient_name.strip()
                        ):

                            result = {

                                "success":
                                    False,

                                "message":
                                    "A valid patient name "
                                    "must be provided."
                            }


                        elif (
                            appointment_id is None
                            or appointment_id
                            not in available_slots.get(
                                conversation_id,
                                set()
                            )
                        ):

                            result = {

                                "success":
                                    False,

                                "message":
                                    "This appointment slot "
                                    "has not been verified "
                                    "as available."
                            }


                        else:

                            result = book_appointment(

                                appointment_id=
                                    appointment_id,

                                patient_name=
                                    patient_name.strip()
                            )


                            if result.get(
                                "success"
                            ):

                                available_slots[
                                    conversation_id
                                ].discard(
                                    appointment_id
                                )


                                clear_booking_state(
                                    conversation_id
                                )


                    # ==================================================
                    # SEARCH CLINIC KNOWLEDGE
                    # ==================================================

                    elif tool_name == (
                        "search_clinic_knowledge"
                    ):

                        results = search_knowledge(
                            query=arguments["query"]
                        )


                        if results:

                            result = {

                                "found":
                                    True,

                                "results":
                                    results
                            }

                        else:

                            result = {

                                "found":
                                    False,

                                "results":
                                    [],

                                "message":
                                    (
                                        "No sufficiently relevant "
                                        "information was found in "
                                        "the clinic knowledge base."
                                    )
                            }


                    # ==================================================
                    # UNKNOWN TOOL
                    # ==================================================

                    else:

                        result = {

                            "success":
                                False,

                            "message":
                                "The requested action "
                                "is not available."
                        }


                except KeyError:

                    logger.exception(
                        "Missing tool argument: %s",
                        tool_name
                    )


                    result = {

                        "success":
                            False,

                        "message":
                            "The requested action "
                            "could not be completed."
                    }


                except Exception:

                    logger.exception(
                        "Tool execution failed: %s",
                        tool_name
                    )


                    result = {

                        "success":
                            False,

                        "message":
                            "The requested action "
                            "could not be completed right now."
                    }


                conversation.append(

                    {
                        "role":
                            "tool",

                        "tool_call_id":
                            tool_call.id,

                        "content":
                            json.dumps(result),
                    }
                )


            # ==================================================
            # FINAL LLM RESPONSE
            # ==================================================

            try:

                final_response = ask_llm(
                    conversation,
                    tools=TOOLS
                )


                response_text = (
                    final_response.content
                    or ""
                )


            except Exception:

                logger.exception(
                    "Final LLM request failed for conversation %s",
                    conversation_id
                )


                response_text = (
                    "I'm sorry, I couldn't complete "
                    "that request right now."
                )


        else:

            response_text = (
                response.content
                or ""
            )


        # ==================================================
        # EMPTY RESPONSE PROTECTION
        # ==================================================

        if not response_text.strip():

            response_text = (
                "I'm sorry, I couldn't generate a response "
                "right now. Please try again."
            )


        # ==================================================
        # SAVE ASSISTANT RESPONSE
        # ==================================================

        conversation.append(

            {
                "role":
                    "assistant",

                "content":
                    response_text
            }
        )


        enforce_conversation_limits(
            conversation_id
        )


        # ==================================================
        # RETURN
        # ==================================================

        return {

            "conversation_id":
                conversation_id,

            "response":
                response_text
        }


    except HTTPException:

        raise


    except Exception:

        logger.exception(
            "Unexpected chat endpoint failure"
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "An unexpected error occurred "
                "while processing the request."
            )
        )


# ==================================================
# ADMIN: BUILD RAG
# ==================================================

@app.post("/api/admin/build-rag")
def build_rag(
    x_admin_key: str | None = Header(
        default=None
    )
):

    verify_admin_key(
        x_admin_key
    )


    try:

        return build_index()

    except Exception:

        logger.exception(
            "RAG index build failed"
        )


        raise HTTPException(
            status_code=500,
            detail="RAG index build failed."
        )


# ==================================================
# ADMIN: SEARCH RAG
# ==================================================

@app.post("/api/admin/search-rag")
def search_rag(
    query: str,
    x_admin_key: str | None = Header(
        default=None
    )
):

    verify_admin_key(
        x_admin_key
    )


    try:

        return {
            "results":
                search_knowledge(query)
        }

    except Exception:

        logger.exception(
            "RAG search failed"
        )


        raise HTTPException(
            status_code=500,
            detail="RAG search failed."
        )


# ==================================================
# STATIC FRONTEND
# ==================================================

app.mount(

    "/",

    StaticFiles(

        directory="static",

        html=True
    ),

    name="static"
)