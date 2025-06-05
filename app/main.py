from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from rag import rag_pipeline  # your existing pipeline function
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
COMPLETION_MODEL = "gpt-4o-mini"
client = OpenAI(openai_api_key = os.environ.get("OPENAI_API_KEY"))
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
class OnboardingRequest(BaseModel):
    conversation_history: List[Message]
@app.post("/onboarding")
async def onboarding_chat(request: OnboardingRequest):
    try:
        completion = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[msg.dict() for msg in request.conversation_history],
            temperature=0.7,
            max_tokens=300,
        )
        reply = completion.choices[0].message.content
        return {
            "response": reply,
            "conversation_history": request.conversation_history + [Message(role="assistant", content=reply)]
        }
    except Exception as e:
        return {"error": str(e)}
