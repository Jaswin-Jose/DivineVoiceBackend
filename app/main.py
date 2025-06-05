from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import firebase_admin
from firebase_admin import credentials, firestore

from rag import rag_pipeline  # your existing RAG pipeline

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("/etc/secrets/firebase.json")  # make sure this file exists
    firebase_admin.initialize_app(cred)
db = firestore.client()

# OpenAI setup
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

# ==== /chat ====
class ChatRequest(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]]

@app.post("/chat")
async def chat(request: ChatRequest):
    answer, updated_history = rag_pipeline(request.query, request.conversation_history)
    return {
        "response": answer,
        "conversation_history": updated_history
    }

# ==== /onboarding ====
class Message(BaseModel):
    role: str
    content: str

class OnboardingRequest(BaseModel):
    user_id: str
    conversation_history: List[Message]

@app.post("/onboarding")
async def onboarding_chat(request: OnboardingRequest):
    system_prompt = {
        "role": "system",
        "content": (
            "You are Lucy, a Catholic assistant helping new users set up their spiritual plan."
            " Ask exactly 7 personalized onboarding questions, one at a time."
            " The questions must help generate a personalized prayer schedule (times, intentions, church access, etc)."
            " At the end of the 7th question, summarize the user's answers and generate a weekly spiritual schedule."
            " Format your final reply as:\n\n"
            "---SCHEDULE_START---\n"
            "{json with the weekly schedule}\n"
            "---SCHEDULE_END---"
        )
    }

    messages = [system_prompt] + [msg.dict() for msg in request.conversation_history]

    try:
        completion = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        reply = completion.choices[0].message.content

        # Check for schedule block
        if "---SCHEDULE_START---" in reply and "---SCHEDULE_END---" in reply:
            schedule_json = reply.split("---SCHEDULE_START---")[1].split("---SCHEDULE_END---")[0].strip()
            schedule_data = eval(schedule_json)  # Or use json.loads() if you prefer JSON-safe
            db.collection("users").document(request.user_id).set({
                "schedule": schedule_data,
                "onboarding_completed": True,
            }, merge=True)

        return {
            "response": reply,
            "conversation_history": request.conversation_history + [Message(role="assistant", content=reply)]
        }

    except Exception as e:
        return {"error": str(e)}

# ==== /lucychats ====
class LucyChatRequest(BaseModel):
    user_id: str
    conversation_history: List[Message]

@app.post("/lucychats")
async def lucy_chat(request: LucyChatRequest):
    system_prompt = {
        "role": "system",
        "content": (
            "You are Lucy, a Catholic AI companion who chats with users after onboarding."
            " You should use the user’s history to follow up on events (e.g., exams, interviews),"
            " suggest prayers, and remind them of their spiritual goals."
            " If you want to update the user schedule or data, reply using a JSON block like:\n"
            "---UPDATE_START---\n"
            "{json}\n"
            "---UPDATE_END---"
            "\nThis block will be parsed and applied to the user's Firestore document."
        )
    }

    messages = [system_prompt] + [msg.dict() for msg in request.conversation_history]

    try:
        completion = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=600
        )
        reply = completion.choices[0].message.content

        # Firestore Update Logic
        if "---UPDATE_START---" in reply and "---UPDATE_END---" in reply:
            update_json = reply.split("---UPDATE_START---")[1].split("---UPDATE_END---")[0].strip()
            update_data = eval(update_json)  # Use json.loads() if preferred
            db.collection("users").document(request.user_id).set(update_data, merge=True)

        return {
            "response": reply,
            "conversation_history": request.conversation_history + [Message(role="assistant", content=reply)]
        }

    except Exception as e:
        return {"error": str(e)}
