import json

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
from app.database import initialize_database, seed_appointments


app = FastAPI(title="CareFlow AI")


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

initialize_database()
seed_appointments()


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    conversation_id: str
    message: str


# --------------------------------------------------
# CONVERSATION MEMORY
# --------------------------------------------------

conversations: dict[str, list[dict]] = {}


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/chat")
def chat(request: ChatRequest):

    # Create a new conversation if this ID does not exist.
    if request.conversation_id not in conversations:

        conversations[request.conversation_id] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    conversation = conversations[
        request.conversation_id
    ]


    # Add the user's message to conversation history.
    conversation.append(
        {
            "role": "user",
            "content": request.message
        }
    )


    # Ask the LLM whether it wants to answer directly
    # or use one of our available tools.
    response = ask_llm(
        conversation,
        tools=TOOLS
    )


    # --------------------------------------------------
    # TOOL CALLING
    # --------------------------------------------------

    if response.tool_calls:

        # Store the assistant's tool request.
        conversation.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in response.tool_calls
                ]
            }
        )


        # Execute each requested tool.
        for tool_call in response.tool_calls:

            arguments = json.loads(
                tool_call.function.arguments
            )


            # --------------------------------------------------
            # CHECK SPECIFIC APPOINTMENT
            # --------------------------------------------------

            if tool_call.function.name == (
                "check_appointment_availability"
            ):

                result = check_appointment_availability(
                    date=arguments["date"],
                    preferred_time=arguments["preferred_time"]
                )


            # --------------------------------------------------
            # GET AVAILABLE APPOINTMENTS
            # --------------------------------------------------

            elif tool_call.function.name == (
                "get_available_appointments"
            ):

                result = get_available_appointments(
                    date=arguments["date"]
                )


            # --------------------------------------------------
            # BOOK APPOINTMENT
            # --------------------------------------------------

            elif tool_call.function.name == (
                "book_appointment"
            ):

                result = book_appointment(
                    appointment_id=arguments["appointment_id"],
                    patient_name=arguments["patient_name"]
                )


            # --------------------------------------------------
            # SEARCH CLINIC KNOWLEDGE
            # --------------------------------------------------

            elif tool_call.function.name == (
                "search_clinic_knowledge"
            ):

                results = search_knowledge(
                    query=arguments["query"]
                )

                if results:

                    result = {
                        "found": True,
                        "results": results
                    }

                else:

                    result = {
                        "found": False,
                        "results": [],
                        "message": (
                            "No sufficiently relevant information "
                            "was found in the clinic knowledge base. "
                            "Do not answer this clinic-specific "
                            "question from general knowledge. "
                            "Tell the user that the information "
                            "is not currently available."
                        )
                    }


            # --------------------------------------------------
            # UNKNOWN TOOL
            # --------------------------------------------------

            else:

                result = {
                    "success": False,
                    "message": "Unknown tool requested."
                }


            # Add the tool result to conversation history.
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            )


        # --------------------------------------------------
        # FINAL LLM RESPONSE
        # --------------------------------------------------

        final_response = ask_llm(
            conversation
        )

        response_text = final_response.content


    else:

        # No tool was needed.
        response_text = response.content


    # --------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------

    conversation.append(
        {
            "role": "assistant",
            "content": response_text
        }
    )


    return {
        "conversation_id": request.conversation_id,
        "response": response_text
    }


# --------------------------------------------------
# RAG ADMIN ENDPOINTS
# --------------------------------------------------

@app.post("/api/admin/build-rag")
def build_rag():

    return build_index()


@app.post("/api/admin/search-rag")
def search_rag(query: str):

    return {
        "results": search_knowledge(query)
    }


# --------------------------------------------------
# STATIC FRONTEND
# --------------------------------------------------

# Keep this AFTER all API routes.
app.mount(
    "/",
    StaticFiles(
        directory="static",
        html=True
    ),
    name="static"
)