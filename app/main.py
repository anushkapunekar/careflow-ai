from fastapi import FastAPI
from pydantic import BaseModel

from app.llm import ask_llm, SYSTEM_PROMPT, TOOLS
from app.tools import check_appointment_availability


app = FastAPI(title="CareFlow AI")


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


conversations: dict[str, list[dict]] = {}


@app.get("/")
def root():
    return {
        "message": "CareFlow AI is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    # Create a new conversation if this ID does not exist
    if request.conversation_id not in conversations:
        conversations[request.conversation_id] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    conversation = conversations[request.conversation_id]

    # Add the user's message to conversation history
    conversation.append(
        {
            "role": "user",
            "content": request.message
        }
    )

    # Ask the LLM whether it wants to answer directly
    # or use one of our available tools
    response = ask_llm(
        conversation,
        tools=TOOLS
    )

    # If the LLM requested a tool
    if response.tool_calls:

        # Store the assistant's tool request in conversation history
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

        # Execute each requested tool
        for tool_call in response.tool_calls:

            if tool_call.function.name == "check_appointment_availability":

                import json

                arguments = json.loads(
                    tool_call.function.arguments
                )

                result = check_appointment_availability(
                    date=arguments["date"],
                    preferred_time=arguments["preferred_time"]
                )

                # Give the tool result back to the LLM
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    }
                )

        # Ask the LLM to produce the final natural-language response
        final_response = ask_llm(conversation)

        response_text = final_response.content

    else:
        # No tool was needed, so use the LLM's direct response
        response_text = response.content

    # Save the final assistant response to conversation history
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