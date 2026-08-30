Absolutely. Copy-paste this **entire thing** into `README.md`:

````markdown
# CareFlow AI

## AI Voice Receptionist Prototype

CareFlow AI is an AI-powered voice receptionist prototype for a fictional healthcare clinic.

The system combines conversational AI, retrieval-augmented generation (RAG), appointment scheduling tools, database integration, and browser-based voice interaction.

> **Note:** CareFlow Demo Clinic is fictional demonstration data. This project is not connected to a real healthcare facility or real patient records.

---

## Features

- Natural-language voice interaction
- Browser-based speech input and output
- Clinic FAQ retrieval using RAG
- Semantic search using embeddings
- FAISS vector index
- Appointment availability lookup
- Specific appointment-time verification
- Appointment booking with explicit confirmation
- Multi-turn conversation state
- Appointment session state
- Natural-language date and time handling
- Cancellation and rescheduling capability detection
- Input validation
- Protected administrative RAG endpoints
- Error and failure handling
- Application logging
- Automated tests
- Embedding caching for reduced latency

---

## Architecture

```text
                         ┌──────────────────────┐
                         │      Browser UI      │
                         │ Voice + Transcript   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │     app/main.py      │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐              ┌───────────────────┐
        │ Conversation      │              │ Local FAQ Router  │
        │ State              │              │ + RAG             │
        └─────────┬─────────┘              └─────────┬─────────┘
                  │                                   │
                  │                                   ▼
                  │                         ┌───────────────────┐
                  │                         │ Hugging Face      │
                  │                         │ Embeddings        │
                  │                         └─────────┬─────────┘
                  │                                   │
                  │                                   ▼
                  │                         ┌───────────────────┐
                  │                         │ FAISS Index       │
                  │                         │ + Clinic FAQ      │
                  │                         └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Groq LLM          │
        │ Qwen              │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Appointment Tools │
        │ app/tools.py      │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ SQLite Database   │
        │ app/database.py   │
        └───────────────────┘
````

---

## Project Structure

```text
voice agent/
│
├── app/
│   ├── database.py
│   ├── llm.py
│   ├── main.py
│   ├── rag.py
│   ├── tools.py
│   └── __init__.py
│
├── knowledge/
│   ├── clinic_faq.txt
│   ├── careflow.index
│   └── chunks.json
│
├── static/
│   └── index.html
│
├── tests/
│   ├── test_main.py
│   ├── test_rag.py
│   └── test_tools.py
│
├── .env
├── README.md
└── requirements.txt
```

---

## Technology Stack

### Backend

* Python
* FastAPI
* Uvicorn

### AI / LLM

* Groq API
* Qwen
* Hugging Face Inference API
* Sentence Transformers

### Retrieval

* FAISS
* Semantic similarity search
* Local clinic knowledge base
* Query normalization
* Embedding cache

### Database

* SQLite

### Frontend

* HTML
* CSS
* JavaScript
* Browser speech capabilities

### Testing

* pytest
* FastAPI TestClient

---

# RAG Pipeline

The clinic knowledge base is stored in:

```text
knowledge/clinic_faq.txt
```

The RAG pipeline works as follows:

```text
Clinic FAQ
    │
    ▼
Document Chunking
    │
    ▼
Sentence Embeddings
    │
    ▼
FAISS Vector Index
    │
    ▼
Semantic Query Search
    │
    ▼
Relevant Clinic Information
    │
    ▼
AI Response
```

The system also normalizes common conversational voice queries.

For example:

```text
"can I walk in"
"can I come without an appointment"
"do you accept insurance"
"what should I bring"
```

These are mapped toward the terminology used by the clinic knowledge base to improve retrieval quality.

---

# Appointment Workflow

Appointment availability is controlled by the application database.

The AI does not invent appointment availability.

The booking workflow is:

```text
User requests appointment
        │
        ▼
Determine date
        │
        ▼
Check available appointments
        │
        ▼
User selects time
        │
        ▼
Verify specific time
        │
        ▼
Ask for confirmation
        │
        ▼
Ask for patient name
        │
        ▼
Book verified appointment
        │
        ▼
Database confirms booking
        │
        ▼
Return confirmation
```

The database is the final authority for appointment availability and booking status.

---

# Example Conversation

```text
User:
Book an appointment for September 21st.

CareFlow AI:
What time would you prefer?

User:
2 PM.

CareFlow AI:
Just to confirm, would you like me to book an
appointment for September 21st at 2 PM?

User:
Yes, go ahead.

CareFlow AI:
Absolutely. What is your full name?

User:
Shawn.

CareFlow AI:
Your appointment for September 21st at 2 PM
has been booked, Shawn.
```

The system maintains the appointment context across multiple turns.

---

# Agentic Architecture

CareFlow AI uses an LLM-driven tool-calling workflow.

The model can determine when it needs to use application tools such as:

```text
check_appointment_availability
get_available_appointments
book_appointment
search_clinic_knowledge
```

The LLM is not treated as the source of truth for database state.

Instead:

```text
LLM reasoning
      │
      ▼
Tool selection
      │
      ▼
Application logic
      │
      ▼
Database / RAG
      │
      ▼
Verified result
      │
      ▼
Natural-language response
```

This separation reduces the risk of fabricated appointment availability or booking confirmations.

---

# Conversation State

The application maintains conversation state so users can interact naturally across multiple turns.

For example:

```text
User:
Book September 21st.

AI:
What time would you prefer?

User:
2 PM.

AI:
Just to confirm, would you like me to book September 21st at 2 PM?

User:
Yes.

AI:
What is your full name?

User:
Shawn.
```

The system can preserve the relevant appointment context while collecting the remaining required information.

---

# Safety

CareFlow AI is a demonstration healthcare receptionist.

It does not:

* Diagnose medical conditions
* Provide personalized treatment recommendations
* Prescribe medication
* Retrieve real medical records
* Modify real medical records
* Provide emergency medical services
* Invent clinic information
* Invent appointment availability
* Invent booking confirmations

For potentially life-threatening emergencies, the system advises users to seek appropriate emergency medical assistance immediately.

---

# Privacy

The application uses synthetic demonstration data.

It is not intended for storing or processing real sensitive medical information.

Users should not provide:

* Real medical records
* Passwords
* Financial information
* Other sensitive personal information

---

# Security

The prototype includes:

* Environment-based API key configuration
* Protected administrative endpoints
* Input validation
* Appointment ID validation
* Patient-name validation
* No hardcoded production API credentials

Administrative RAG operations require an admin API key.

---

# Error Handling

The application handles failures across multiple layers.

Examples include:

* LLM API failures
* Groq rate limits
* API status errors
* RAG failures
* Invalid tool arguments
* Missing tool arguments
* Tool execution failures
* Database booking failures
* Unexpected chat endpoint failures

The application logs failures rather than silently ignoring them.

---

# Latency Optimization

The project includes several latency-oriented optimizations.

### Embedding Cache

Repeated embedding requests can be served from an in-memory cache rather than repeatedly calling the Hugging Face inference service.

### Conversation Compaction

Older conversation messages are compacted before being sent to the LLM to control prompt size.

### Concise Voice Responses

The LLM is configured for short, natural responses suitable for voice interaction.

### Local Routing

Common clinic FAQ requests can be handled through local routing without unnecessarily invoking the LLM.

---

# Logging and Observability

The application provides logging for important operations.

Examples include:

```text
LOCAL FAQ ROUTER
LOCAL FAQ ANSWER
LOCAL AVAILABILITY RESULT
LOCAL DATE AVAILABILITY CHECK
LOCAL DATE CHANGE AVAILABILITY RESULT
LOCAL TIME CHECK RESULT
LOCAL BOOKING RESULT
LOCAL ROUTER handled request
Calling Groq model
```

Failures are also logged with exception information.

This makes it easier to diagnose:

* RAG failures
* LLM failures
* Tool failures
* Database failures
* Unexpected endpoint errors

---

# API Endpoints

The application exposes the following main endpoints:

```text
GET  /api/health
POST /api/chat
POST /api/admin/build-rag
POST /api/admin/search-rag
```

The administrative RAG endpoints are protected using the configured admin API key.

---

# Running the Application

## 1. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 2. Configure environment variables

Create a `.env` file in the project root.

Example:

```text
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
CARE_FLOW_ADMIN_KEY=your_admin_key
```

Do not commit real credentials to Git.

---

## 3. Start the server

```powershell
uvicorn app.main:app --reload
```

The application will run locally at:

```text
http://127.0.0.1:8000
```

---

# Building the RAG Index

If the clinic knowledge base is modified, rebuild the FAISS index:

```powershell
python -c "from app.rag import build_index; print(build_index())"
```

Example:

```text
{
    'success': True,
    'chunks_indexed': 21,
    'embedding_dimension': 384
}
```

This generates/updates:

```text
knowledge/careflow.index
knowledge/chunks.json
```

---

# Testing

The project uses pytest for automated testing.

Run the complete test suite:

```powershell
pytest -v
```

Current test coverage includes:

### RAG

* Document chunking
* Walk-in query normalization
* Insurance query normalization
* Appointment preparation normalization
* Unknown query handling

### Appointment Tools

* Available appointment lookup
* Specific appointment-time verification
* Nonexistent time handling
* Appointment ID validation
* Patient-name validation
* Placeholder-name rejection
* Invalid booking handling

### FastAPI

* Health endpoint
* Chat endpoint
* Walk-in questions
* Invalid chat input

Current test suite:

```text
16 passed
```

---

# Development Principles

The project follows several reliability principles.

### Database is authoritative

The LLM cannot decide whether an appointment is actually available.

### Tool results are authoritative

The AI only confirms an appointment after the application/database confirms the operation.

### Explicit confirmation

The user must explicitly confirm before an appointment is booked.

### Required information

Booking requires:

```text
Verified appointment ID
+
Explicit patient name
+
Explicit user confirmation
```

### No fabrication

The system should not invent:

* Appointment slots
* Clinic policies
* Addresses
* Contact information
* Patient records
* Booking results

---

# Demo Limitations

This is a fictional demonstration system.

The prototype currently does not integrate with:

* Real healthcare providers
* Real EHR systems
* Real patient records
* Real insurance systems
* Real clinic locations
* SMS notification providers
* Email notification providers
* Human call transfer systems
* Emergency medical services

Cancellation and rescheduling are intentionally not implemented in the current prototype.

---

# Future Improvements

Potential future improvements include:

* Production authentication
* Persistent conversation storage
* Redis-based state management
* Distributed deployment
* Production metrics
* OpenTelemetry tracing
* Streaming LLM responses
* Production speech-to-text
* Production text-to-speech
* Human handoff
* SMS/email reminders
* Real healthcare-system integrations
* Expanded integration testing
* CI/CD pipeline
* Containerized deployment

---

