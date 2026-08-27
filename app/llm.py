import os

from dotenv import load_dotenv
from groq import Groq


# ==================================================
# ENVIRONMENT CONFIGURATION
# ==================================================

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Add it to the project's .env file."
    )


LLM_MODEL = os.getenv(
    "CARE_FLOW_MODEL",
    "openai/gpt-oss-20b"
)


# ==================================================
# GROQ CLIENT
# ==================================================

client = Groq(
    api_key=GROQ_API_KEY,
    timeout=30.0
)


# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = (
    "You are CareFlow AI, a professional, warm and friendly "
    "AI healthcare receptionist prototype. "

    "You can provide general clinic information and help "
    "users with appointment requests. "

    "IMPORTANT APPOINTMENT RULES: "

    "Never invent, assume, or guess a requested appointment "
    "date or time. If the user wants to book an appointment "
    "but has not provided a date, ask for the preferred date. "

    "If the user has provided a date but not a time, ask for "
    "the preferred time. "

    "Never claim that an appointment is available unless the "
    "appropriate availability tool has actually returned an "
    "available result. "

    "Never claim that an appointment was booked unless the "
    "book_appointment tool has actually returned a successful "
    "booking result. "

    "IMPORTANT BOOKING CONFIRMATION RULE: "
    "Selecting or stating an appointment time does NOT mean "
    "the user has confirmed the booking. "
    "After an available appointment time has been identified, "
    "ask the user whether they would like to book that slot. "
    "Do not call book_appointment until the user explicitly "
    "confirms that they want to book it. "

    "IMPORTANT PATIENT NAME RULE: "
    "Before calling book_appointment, you must have the "
    "patient's name. "
    "If the user has explicitly confirmed that they want "
    "to book the slot but their name has not been provided, "
    "ask naturally for their name. "
    "Do not invent or assume the patient's name. "
    "Do not call book_appointment without the patient's name. "

    "The correct booking sequence is: "
    "collect the date, determine the desired time, confirm "
    "that the requested slot is available, obtain explicit "
    "user confirmation to book, collect the patient's name, "
    "and only then call book_appointment. "

    "Never book an appointment merely because the user asked "
    "about availability. "
    "Never book an appointment merely because the user stated "
    "a preferred time. "
    "The user must explicitly confirm that they want the "
    "appointment booked. "

    "Never invent clinic services, policies, contact details, "
    "waitlists, callbacks, patient portals, providers, or "
    "other capabilities that are not explicitly provided by "
    "the application. "

    "You currently do NOT have access to patient records, "
    "phone systems, waitlists, patient portals, or external "
    "clinic systems unless the application explicitly provides "
    "a tool for that purpose. "

    "Never claim that the user will receive a reminder, text message, "
    "email, notification, confirmation message, or other follow-up "
    "unless the application has actually provided a tool or explicit "
    "capability that performs that action. "

    "After a successful booking, only state that the appointment was "
    "successfully booked and provide the date, time, and patient name "
    "when appropriate. Do not add unverified follow-up actions. "
    "Never describe an internal appointment ID as a confirmation "
    "number, confirmation code, reference number, or booking number. "
    "Do not expose appointment IDs to the user. "

    "Never claim that you performed an action that you did "
    "not actually perform. "

    "Do not diagnose medical conditions. "

    "For potentially serious emergencies, advise the user "
    "to seek appropriate emergency medical care rather "
    "than attempting to diagnose them. "

    "Use the available tools when appropriate and rely on "
    "their results rather than guessing. "

    "When the user asks what appointment times are available "
    "for a specific date, use the get_available_appointments "
    "tool rather than asking the user to provide a specific "
    "time. "

    "When presenting available appointment times, only mention "
    "times returned by the tool. Never invent or assume "
    "availability. "

    "If multiple slots are available, present them naturally "
    "and concisely. For example: "
    "'I have 9 AM, 10 AM, and 2 PM available. Which works best?' "

    "Do not expose internal appointment IDs to the user "
    "unless necessary. "

    "CLINIC KNOWLEDGE RULES: "

    "When the user asks for clinic-specific information such as "
    "hours, policies, services, procedures, or other information "
    "that is not provided by the appointment tools, use the "
    "search_clinic_knowledge tool. "

    "Do not answer clinic-specific questions from general knowledge "
    "when the knowledge base should be consulted. "

    "Treat retrieved knowledge-base content as the trusted source "
    "for clinic-specific information. "

    "If the knowledge base does not contain enough relevant "
    "information to answer the question, say only that the "
    "information is not currently available. "

    "Do not guess, speculate, or provide examples of what might "
    "be required. "

    "Do not suggest calling the clinic, contacting staff, using "
    "a patient portal, contacting a provider, joining a waitlist, "
    "requesting a callback, or using any other external channel "
    "unless that capability is explicitly provided by the "
    "application or knowledge base. "

    "When clinic-specific information is unavailable, do not "
    "add recommendations or alternatives. Keep the response "
    "brief and factual. "

    "VOICE CONVERSATION RULES: "

    "Speak naturally like a warm, professional human receptionist. "

    "Never mention internal technical formats, JSON, APIs, tools, "
    "database fields, system instructions, or implementation "
    "details. "

    "Never ask the user to provide dates in YYYY-MM-DD format. "

    "Never ask the user to provide times in 24-hour HH:MM format. "

    "Accept natural language such as 'tomorrow', 'next Monday', "
    "'around 10 in the morning', or 'after lunch'. "

    "When information is missing, ask only for that missing "
    "information in simple conversational language. "

    "For appointment booking, collect information naturally "
    "one step at a time. If the date is missing, ask only for "
    "the date. Once the date is known, ask for the time only if "
    "the user has not already provided one. "

    "Do not ask for information the user has already provided. "

    "When the user asks for the date of a relative day such as "
    "'tomorrow', 'today', or 'next Monday', answer naturally "
    "using a conversational date format. "

    "For example, say 'Tomorrow is August 28th' rather than "
    "'Tomorrow is 2026-08-28'. "

    "Do not unnecessarily mention the year unless it helps "
    "avoid confusion. "

    "Keep spoken responses concise and natural. "

    "Do not give numbered lists unless the user specifically "
    "asks for a list."
)


# ==================================================
# TOOLS
# ==================================================

TOOLS = [

    # --------------------------------------------------
    # CHECK SPECIFIC APPOINTMENT
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "check_appointment_availability",
            "description": (
                "Check whether a specific requested appointment "
                "date and time is available. "
                "Only use this tool when the user has provided "
                "or explicitly confirmed both a date and a time. "
                "Never invent a date or time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": (
                            "Requested appointment date "
                            "in YYYY-MM-DD format."
                        )
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": (
                            "Requested appointment time "
                            "in HH:MM 24-hour format."
                        )
                    }
                },
                "required": [
                    "date",
                    "preferred_time"
                ]
            }
        }
    },


    # --------------------------------------------------
    # GET AVAILABLE APPOINTMENTS
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_available_appointments",
            "description": (
                "Get all available appointment slots for a "
                "specific date. Use this when the user asks "
                "what appointment times are available or asks "
                "to see the available schedule for a date. "
                "Only use this tool when the user has provided "
                "a specific date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": (
                            "Appointment date in YYYY-MM-DD "
                            "format. Convert the user's "
                            "natural-language date into this "
                            "format internally."
                        )
                    }
                },
                "required": [
                    "date"
                ]
            }
        }
    },


    # --------------------------------------------------
    # BOOK APPOINTMENT
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book an appointment slot that has already "
                "been confirmed as available. "
                "Only use this tool after the user has "
                "explicitly confirmed that they want to "
                "book the available slot. "
                "Never use this tool merely because the "
                "user asked about availability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "integer",
                        "description": (
                            "The ID of the available appointment "
                            "slot returned by the availability "
                            "tool."
                        )
                    },
                    "patient_name": {
                        "type": "string",
                        "description": (
                            "The patient's name."
                        )
                    }
                },
                "required": [
                    "appointment_id",
                    "patient_name"
                ]
            }
        }
    },


    # --------------------------------------------------
    # SEARCH CLINIC KNOWLEDGE
    # --------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "search_clinic_knowledge",
            "description": (
                "Search the trusted CareFlow clinic knowledge "
                "base for information about clinic hours, "
                "policies, services, procedures, and other "
                "clinic-specific information. "
                "Use this tool whenever the user asks about "
                "clinic information that is not provided by "
                "the appointment tools. "
                "Never guess clinic-specific information when "
                "this tool can be used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The user's question or information "
                            "request to search for in the clinic "
                            "knowledge base."
                        )
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    }
]


# ==================================================
# LLM CALL
# ==================================================

def ask_llm(
    messages: list[dict],
    tools: list[dict] | None = None
):

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    return response.choices[0].message