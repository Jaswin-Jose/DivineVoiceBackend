from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import json

from rag import generate_answer  # your RAG pipeline must support (history, query, docs)

# Load allowed document names (e.g. "Intro/Chapter 1.htm")

# OpenAI Client
COMPLETION_MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Request Schema ====
class ChatRequest(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]]

# ==== /chat ====
@app.post("/chat")
async def chat(request: ChatRequest):
    updated_history, answer = generate_answer(
        request.conversation_history,
        request.query
    )
    return {
        "response": answer,
        "conversation_history": updated_history
    }
