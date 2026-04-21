from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import asyncio
import os

from logger import logger
from db import create_table, save_chat, get_chat_history

# -----------------------------------------------------
# Ensure FAISS index exists (auto-ingest for container)
# -----------------------------------------------------
if not os.path.exists("faiss_index/index.faiss"):
    logger.info("FAISS index missing → running ingestion pipeline")

    import ingest
    ingest.main()

from rag import retrieve_context
from llmservice import call_llm
from prompts import build_prompt
from safety import is_safe_input
from query_rewriter import rewrite_query


# ---------------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------------
logger.info("Starting Enterprise RAG API")

create_table()

app = FastAPI()


# ---------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    message: str | None = None
    query: str | None = None      # accepts older clients that still send "query"
    session_id: str | None = None
    domain: str | None = None   # metadata filter support


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str]


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------
@app.get("/")
def health():
    logger.info("Health check called")
    return {"status": "Enterprise Bedrock RAG running"}


# ---------------------------------------------------------
# MAIN CHAT ENDPOINT
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    # create or reuse session
    session_id = req.session_id or str(uuid.uuid4())
    user_message = req.message or req.query

    if not user_message or not user_message.strip():
        raise HTTPException(
            status_code=400,
            detail="Please send either 'message' or 'query' in the request body."
        )

    logger.info(f"New chat request | session={session_id}")

    # -----------------------------------------------------
    # Guardrails check
    # -----------------------------------------------------
    if not is_safe_input(user_message):
        return {
            "answer": "Request blocked by policy.",
            "session_id": session_id,
            "sources": []
        }

    # -----------------------------------------------------
    # Load conversation memory
    # -----------------------------------------------------
    history = get_chat_history(session_id)

    # -----------------------------------------------------
    # Rewrite vague/follow-up questions for better retrieval.
    # call_llm has a mode toggle in llmservice.py:
    # - mode="rewrite" uses the lightweight query-rewrite LLM
    # - mode="answer" uses the stronger final-answer LLM
    # -----------------------------------------------------
    rewritten_query = await asyncio.to_thread(
        rewrite_query,
        lambda prompt: call_llm(prompt, mode="rewrite"),
        history,
        user_message
    )

    # -----------------------------------------------------
    # Metadata filter (domain routing)
    # -----------------------------------------------------
    metadata_filter = None

    if req.domain:
        metadata_filter = {
            "doc_type": req.domain
        }

    # -----------------------------------------------------
    # Retrieve context from FAISS (non-blocking)
    # -----------------------------------------------------
    context, sources, _, _ = await asyncio.to_thread(
        retrieve_context,
        rewritten_query,
        metadata_filter
    )

    if not context:
        logger.warning("No retrieval context found")
        context = "No company knowledge found."

    # -----------------------------------------------------
    # Build final prompt
    # -----------------------------------------------------
    prompt = build_prompt(history, context, user_message)

    # -----------------------------------------------------
    # Generate the final user answer.
    # Bedrock calls are blocking, so run them in a worker thread
    # to keep FastAPI responsive when served by uvicorn.
    # -----------------------------------------------------
    answer = await asyncio.to_thread(call_llm, prompt, "answer")

    # -----------------------------------------------------
    # Save conversation memory
    # -----------------------------------------------------
    save_chat(session_id, user_message, answer)

    logger.info(f"Response completed | session={session_id}")

    return {
        "answer": answer,
        "session_id": session_id,
        "sources": sources
    }
