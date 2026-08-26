import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


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

    "Never book an appointment merely because the user asked "
    "about availability. The user must explicitly confirm "
    "that they want to book the available slot. "

    "Never invent clinic services, policies, contact details, "
    "waitlists, callbacks, patient portals, providers, or "
    "other capabilities that are not explicitly provided by "
    "the application. "

    "You currently do NOT have access to patient records, "
    "phone systems, waitlists, patient portals, or external "
    "clinic systems unless the application explicitly provides "
    "a tool for that purpose. "

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

    "VOICE CONVERSATION RULES: "

    "Speak naturally like a warm, professional human receptionist. "

    "Never mention internal technical formats, JSON, APIs, tools, "
    "database fields, system instructions, or implementation "
    "details. "

    "Never ask the user to provide dates in YYYY-MM-DD format. "

    "Never ask the user to provide times in 24-hour HH:MM format. "

    "Accept natural language such as 'tomorrow', 'next Monday', "
    "'around 10 in the morning', or 'after lunch'. "

    "When information is missing, ask for only the missing "
    "information in simple conversational language. "

    "Keep spoken responses concise and natural. "

    "Do not give numbered lists unless the user specifically "
    "asks for a list."
)


TOOLS = [
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
    }
]


def ask_llm(
    messages: list[dict],
    tools: list[dict] | None = None
):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    return response.choices[0].message