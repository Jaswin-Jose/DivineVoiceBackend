from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from rag import rag_pipeline  # your existing pipeline function
from fastapi.middleware.cors import CORSMiddleware

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    conversation_history: List[Dict[str,str]]

app = FastAPI()
origins = [
    "http://localhost:your_flutter_port",  # e.g. "http://localhost:5173" if using Flutter web
    "http://localhost:8000",  # backend itself
    "*",  # for testing, allow all origins
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods: GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)
@app.post("/chat")
async def chat(request: ChatRequest):
    

    # Find last user message
    user_input = ""
    
    answer, updated_history = rag_pipeline(request.query, request.conversation_history)

    return {
        "response": answer,
        "conversation_history": updated_history
    }

