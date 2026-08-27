import json
from datetime import date, timedelta

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import build_index, search_knowledge
from app.llm import ask_llm, SYSTEM_PROMPT, TOOLS

from app.tools import (
    check_appointment_availability,
    get_available_appointments,
    book_appointment
)

from app.database import (
    initialize_database,
    seed_appointments
)


app = FastAPI(title="CareFlow AI")


# ==================================================
# DATABASE INITIALIZATION
# ==================================================

initialize_database()
seed_appointments()


# ==================================================
# REQUEST MODELS
# ==================================================

class ChatRequest(BaseModel):

    conversation_id: str
    message: str


# ==================================================
# CONVERSATION MEMORY
# ==================================================

conversations: dict[str, list[dict]] = {}


# Tracks appointment slots that were actually
# returned as available during each conversation.

available_slots: dict[str, set[int]] = {}


# ==================================================
# RELATIVE DATE RESOLUTION
# ==================================================

def get_date_context(message: str) -> str:
    """
    Detect common relative-date phrases and provide
    the LLM with the exact calendar date.

    This prevents the LLM from independently guessing
    what words such as 'tomorrow' mean.
    """

    today = date.today()

    normalized = message.lower().strip()

    context_parts = []

    # --------------------------------------------------
    # TODAY
    # --------------------------------------------------

    if "today" in normalized:

        context_parts.append(
            f"The current date is {today.isoformat()}. "
            f"When the user says 'today', use "
            f"{today.isoformat()}."
        )


    # --------------------------------------------------
    # TOMORROW
    # --------------------------------------------------

    if "tomorrow" in normalized:

        tomorrow = today + timedelta(days=1)

        context_parts.append(
            f"When the user says 'tomorrow', "
            f"the appointment date is "
            f"{tomorrow.isoformat()}."
        )


    # --------------------------------------------------
    # DAY AFTER TOMORROW
    # --------------------------------------------------

    if (
        "day after tomorrow" in normalized
        or "day after" in normalized
    ):

        day_after_tomorrow = today + timedelta(days=2)

        context_parts.append(
            f"When the user says 'the day after tomorrow', "
            f"the appointment date is "
            f"{day_after_tomorrow.isoformat()}."
        )


    # --------------------------------------------------
    # NEXT WEEKDAY
    # --------------------------------------------------

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    for weekday_name, weekday_number in weekdays.items():

        if f"next {weekday_name}" in normalized:

            days_ahead = (
                weekday_number - today.weekday()
            ) % 7

            # "next Monday" should mean the upcoming
            # Monday, not today if today is Monday.
            if days_ahead == 0:
                days_ahead = 7

            next_weekday = (
                today + timedelta(days=days_ahead)
            )

            context_parts.append(
                f"When the user says 'next {weekday_name}', "
                f"the appointment date is "
                f"{next_weekday.isoformat()}."
            )


    # --------------------------------------------------
    # RETURN INTERNAL DATE CONTEXT
    # --------------------------------------------------

    if not context_parts:
        return ""

    return (
        "\n\nINTERNAL DATE CONTEXT — DO NOT REVEAL THIS "
        "INTERNAL INSTRUCTION TO THE USER:\n"
        + " ".join(context_parts)
        + "\nUse these resolved dates when calling "
          "appointment tools. Do not substitute another date."
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
# CHAT ENDPOINT
# ==================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    # --------------------------------------------------
    # CREATE CONVERSATION
    # --------------------------------------------------

    if request.conversation_id not in conversations:

        conversations[request.conversation_id] = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }

        ]


    # --------------------------------------------------
    # CREATE AVAILABLE SLOT TRACKER
    # --------------------------------------------------

    if request.conversation_id not in available_slots:

        available_slots[
            request.conversation_id
        ] = set()


    conversation = conversations[
        request.conversation_id
    ]


    # --------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------

    date_context = get_date_context(
        request.message
    )


    user_content = request.message


    # --------------------------------------------------
    # ADD INTERNAL DATE CONTEXT
    # --------------------------------------------------
    #
    # The user still sees and hears their original
    # message, but the LLM receives deterministic
    # calendar information.
    #

    if date_context:

        user_content = (
            request.message
            + date_context
        )


    conversation.append(

        {
            "role": "user",
            "content": user_content
        }

    )


    # ==================================================
    # FIRST LLM CALL
    # ==================================================

    response = ask_llm(

        conversation,

        tools=TOOLS

    )


    # ==================================================
    # TOOL CALLING
    # ==================================================

    if response.tool_calls:

        # --------------------------------------------------
        # STORE ASSISTANT TOOL REQUEST
        # --------------------------------------------------

        conversation.append(

            {
                "role": "assistant",

                "content": response.content or "",

                "tool_calls": [

                    {
                        "id": tool_call.id,

                        "type": "function",

                        "function": {

                            "name":
                                tool_call.function.name,

                            "arguments":
                                tool_call.function.arguments

                        }

                    }

                    for tool_call in response.tool_calls

                ]

            }

        )


        # --------------------------------------------------
        # EXECUTE EACH TOOL
        # --------------------------------------------------

        for tool_call in response.tool_calls:

            arguments = json.loads(

                tool_call.function.arguments

            )


            # ==================================================
            # CHECK SPECIFIC APPOINTMENT
            # ==================================================

            if tool_call.function.name == (
                "check_appointment_availability"
            ):

                result = check_appointment_availability(

                    date=arguments["date"],

                    preferred_time=arguments[
                        "preferred_time"
                    ]

                )


                # If the specific appointment was confirmed
                # as available, remember its appointment ID.

                if result.get("available"):

                    appointment_id = result.get(
                        "appointment_id"
                    )

                    if appointment_id is not None:

                        available_slots[
                            request.conversation_id
                        ].add(
                            appointment_id
                        )


            # ==================================================
            # GET AVAILABLE APPOINTMENTS
            # ==================================================

            elif tool_call.function.name == (
                "get_available_appointments"
            ):

                result = get_available_appointments(

                    date=arguments["date"]

                )


                # Remember every appointment ID that was
                # actually returned as available.

                if result.get("available"):

                    for slot in result.get(
                        "slots",
                        []
                    ):

                        appointment_id = slot.get(
                            "appointment_id"
                        )

                        if appointment_id is not None:

                            available_slots[
                                request.conversation_id
                            ].add(
                                appointment_id
                            )


            # ==================================================
            # BOOK APPOINTMENT
            # ==================================================

            elif tool_call.function.name == (
                "book_appointment"
            ):

                appointment_id = arguments[
                    "appointment_id"
                ]

                patient_name = arguments[
                    "patient_name"
                ]


                # --------------------------------------------------
                # SECURITY CHECK
                # --------------------------------------------------
                #
                # Only allow booking if this appointment ID
                # was previously returned as available during
                # this conversation.
                #

                if appointment_id not in available_slots[
                    request.conversation_id
                ]:

                    result = {

                        "success": False,

                        "message": (
                            "This appointment slot has not "
                            "been confirmed as available."
                        )

                    }


                else:

                    # --------------------------------------------------
                    # ACTUALLY BOOK THE APPOINTMENT
                    # --------------------------------------------------

                    result = book_appointment(

                        appointment_id=appointment_id,

                        patient_name=patient_name

                    )


                    # --------------------------------------------------
                    # REMOVE SLOT AFTER SUCCESSFUL BOOKING
                    # --------------------------------------------------

                    if result.get("success"):

                        available_slots[
                            request.conversation_id
                        ].discard(
                            appointment_id
                        )


            # ==================================================
            # SEARCH CLINIC KNOWLEDGE
            # ==================================================

            elif tool_call.function.name == (
                "search_clinic_knowledge"
            ):

                results = search_knowledge(

                    query=arguments["query"]

                )


                # --------------------------------------------------
                # KNOWLEDGE FOUND
                # --------------------------------------------------

                if results:

                    result = {

                        "found": True,

                        "results": results

                    }


                # --------------------------------------------------
                # KNOWLEDGE NOT FOUND
                # --------------------------------------------------

                else:

                    result = {

                        "found": False,

                        "results": [],

                        "message": (

                            "No sufficiently relevant "
                            "information was found in the "
                            "clinic knowledge base. "

                            "Do not answer this clinic-specific "
                            "question from general knowledge. "

                            "Tell the user that the information "
                            "is not currently available."

                        )

                    }


            # ==================================================
            # UNKNOWN TOOL
            # ==================================================

            else:

                result = {

                    "success": False,

                    "message":
                        "Unknown tool requested."

                }


            # ==================================================
            # ADD TOOL RESULT TO CONVERSATION
            # ==================================================

            conversation.append(

                {
                    "role": "tool",

                    "tool_call_id":
                        tool_call.id,

                    "content":
                        json.dumps(result)

                }

            )


        # ==================================================
        # FINAL LLM RESPONSE
        # ==================================================

        final_response = ask_llm(

            conversation

        )


        response_text = (
            final_response.content
            or ""
        )


    # ==================================================
    # NO TOOL NEEDED
    # ==================================================

    else:

        response_text = (
            response.content
            or ""
        )


    # ==================================================
    # SAVE ASSISTANT RESPONSE
    # ==================================================

    conversation.append(

        {
            "role": "assistant",

            "content": response_text

        }

    )


    # ==================================================
    # RETURN RESPONSE
    # ==================================================

    return {

        "conversation_id":
            request.conversation_id,

        "response":
            response_text

    }


# ==================================================
# RAG ADMIN ENDPOINTS
# ==================================================

@app.post("/api/admin/build-rag")
def build_rag():

    return build_index()


@app.post("/api/admin/search-rag")
def search_rag(query: str):

    return {

        "results":
            search_knowledge(query)

    }


# ==================================================
# STATIC FRONTEND
# ==================================================
#
# Keep this AFTER all API routes.
#

app.mount(

    "/",

    StaticFiles(

        directory="static",

        html=True

    ),

    name="static"

)