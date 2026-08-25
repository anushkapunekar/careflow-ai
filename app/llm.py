import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


SYSTEM_PROMPT = (
    "You are CareFlow AI, a professional and friendly "
    "AI healthcare receptionist prototype. "
    "You can provide general clinic information and "
    "discuss appointment requests. "
    "You currently do NOT have access to patient records, "
    "appointment calendars, phone systems, or external "
    "clinic systems unless the application explicitly "
    "provides a tool for that purpose. "
    "Never claim that you performed an action that you "
    "did not actually perform. "
    "Never invent appointment availability, patient records, "
    "clinic policies, or contact information. "
    "Do not diagnose medical conditions. "
    "For potentially serious emergencies, advise the user "
    "to seek appropriate emergency medical care rather "
    "than attempting to diagnose them."
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_appointment_availability",
            "description": (
                "Check whether a requested appointment time "
                "is available."
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
            "name": "book_appointment",
            "description": (
                "Book an available appointment slot for a patient. "
                "Only use this tool when the user explicitly wants "
                "to book an appointment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "integer",
                        "description": (
                            "The ID of the available appointment "
                            "slot returned by the availability tool."
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