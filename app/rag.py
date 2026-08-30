import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_FILE = (
    BASE_DIR / "knowledge" / "clinic_faq.txt"
)

FAISS_INDEX_FILE = (
    BASE_DIR / "knowledge" / "careflow.index"
)

CHUNKS_FILE = (
    BASE_DIR / "knowledge" / "chunks.json"
)


# --------------------------------------------------
# HUGGING FACE
# --------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not configured in the .env file."
    )


client = InferenceClient(
    token=HF_TOKEN
)


EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# LOAD KNOWLEDGE
# --------------------------------------------------

def load_knowledge() -> str:

    if not KNOWLEDGE_FILE.exists():

        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    return KNOWLEDGE_FILE.read_text(
        encoding="utf-8"
    )


# --------------------------------------------------
# CHUNK DOCUMENT
# --------------------------------------------------

def create_chunks(
    text: str
) -> list[str]:

    sections = []

    current_section = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Uppercase lines represent section headings.
        if line.isupper():

            if current_section:

                sections.append(
                    "\n".join(current_section)
                )

            current_section = [line]

        else:

            current_section.append(line)

    if current_section:

        sections.append(
            "\n".join(current_section)
        )

    return sections


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

def create_embeddings(
    texts: list[str]
) -> np.ndarray:

    embeddings = []

    for text in texts:

        result = client.feature_extraction(
            text,
            model=EMBEDDING_MODEL
        )

        vector = np.asarray(
            result,
            dtype="float32"
        )

        vector = vector.reshape(-1)

        embeddings.append(vector)

    return np.vstack(embeddings)


# --------------------------------------------------
# BUILD FAISS INDEX
# --------------------------------------------------

def build_index():

    text = load_knowledge()

    chunks = create_chunks(text)

    if not chunks:

        raise ValueError(
            "Knowledge document is empty."
        )

    embeddings = create_embeddings(
        chunks
    )

    dimension = embeddings.shape[1]

    # Normalize vectors so inner product
    # behaves like cosine similarity.
    faiss.normalize_L2(
        embeddings
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_FILE)
    )

    # Store chunks in JSON so the FAISS
    # index position maps reliably to a chunk.
    CHUNKS_FILE.write_text(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return {
        "success": True,
        "chunks_indexed": len(chunks),
        "embedding_dimension": dimension
    }


# --------------------------------------------------
# LOAD INDEX
# --------------------------------------------------

def load_index():

    if not FAISS_INDEX_FILE.exists():

        raise FileNotFoundError(
            "FAISS index does not exist. "
            "Run build_index() first."
        )

    return faiss.read_index(
        str(FAISS_INDEX_FILE)
    )


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

def load_chunks() -> list[str]:

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            "Chunks file does not exist. "
            "Run build_index() first."
        )

    return json.loads(
        CHUNKS_FILE.read_text(
            encoding="utf-8"
        )
    )



# --------------------------------------------------
# QUERY NORMALIZATION
# --------------------------------------------------

def normalize_query(query: str) -> str:
    """
    Normalize common conversational variations into
    terminology used by the clinic knowledge base.

    This improves retrieval for short voice queries.
    """

    normalized = query.lower().strip()

    walk_in_phrases = (
        "walk in",
        "walk-in",
        "without booking",
        "without an appointment",
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
        return (
            "walk-in appointments "
            "walk in without an appointment "
            "walk-in appointment policy"
        )

    insurance_phrases = (
        "insurance",
        "insurance plan",
        "insurance accepted",
        "accept insurance",
        "take insurance",
        "coverage",
    )

    if any(
        phrase in normalized
        for phrase in insurance_phrases
    ):
        return (
            "insurance accepted insurance plans "
            "insurance coverage insurance requirements"
        )

    preparation_phrases = (
        "what should i bring",
        "what do i bring",
        "bring to my appointment",
        "bring for my appointment",
        "documents for my appointment",
        "appointment preparation",
        "prepare for my appointment",
    )

    if any(
        phrase in normalized
        for phrase in preparation_phrases
    ):
        return (
            "what to bring appointment preparation "
            "photo ID medical records medication information"
        )

    return query

# --------------------------------------------------
# SEARCH KNOWLEDGE
# --------------------------------------------------

# --------------------------------------------------
# SEARCH KNOWLEDGE
# --------------------------------------------------

def search_knowledge(
    query: str,
    top_k: int = 3,
    relevance_threshold: float = 0.40
) -> list[dict]:
    """
    Search the clinic knowledge base using semantic similarity.

    A slightly lower threshold is used because short,
    conversational voice queries can have lower embedding
    similarity even when they clearly refer to a known
    clinic policy.

    The caller can still override the threshold when needed.
    """

    index = load_index()

    chunks = load_chunks()

    if not query or not query.strip():
        return []

    #normalize conversational voice query

    normalized_query = normalize_query(query)

    #create query embedding

    query_embedding = create_embeddings(
        [normalized_query]
    )

    faiss.normalize_L2(
        query_embedding
    )

    top_k = min(
        top_k,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        score = float(score)

        if score < relevance_threshold:
            continue

        results.append(
            {
                "score": score,
                "content": chunks[index_id]
            }
        )

    return results