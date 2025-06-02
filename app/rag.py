import json
import faiss
import numpy as np
import os
from openai import OpenAI

# Set your OpenAI API key as env var OPENAI_API_KEY
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Set your OPENAI_API_KEY environment variable")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)
TOP_K = 15
# Constants
EMBEDDING_MODEL = "text-embedding-3-small"  # or your preferred embedding model
COMPLETION_MODEL = "gpt-4o-mini"  # or another you want for rewriting + answering

# Load FAISS index & numpy embeddings
FAISS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.index")
index = faiss.read_index(FAISS_PATH)

# Load chunks & metadata
BASE_DIR = os.path.dirname(__file__)
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.json")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def embed_text(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding).astype("float32")

def rewrite_query(conversation_history):
    """Rewrite the last user query for clarity, given the conversation history."""
    messages = conversation_history + [
        {
            "role": "system",
            "content": """
You are a faithful Catholic theologian. Your task is to take a user's raw question and rewrite it into a clearer, more precise query that can be answered using only official Catholic Church sources such as the Code of Canon Law and writings of the Church Fathers.

When rewriting:
- Clarify vague or broad language.
- Add relevant canonical or theological context if needed.
- Keep the query short, formal, and accurate.
- Make sure it is shaped in the best way.
- You should not answer, simply rewrite the question.
- When query is like mathew 3:1-5, rewrite it as mathew 3:1,3:2,3:3,3:4,3:5.
"""
        }
    ]
    completion = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    rewritten = completion.choices[0].message.content.strip()
    return rewritten

def search_index(query_embedding, top_k=TOP_K):
    """Search FAISS index for top_k closest chunks."""
    query_vector = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return indices[0]

def build_context(retrieved_indices):
    """Build context text by concatenating retrieved chunks."""
    context_pieces = []
    for idx in retrieved_indices:
        chunk_text = chunks[idx]["content"]
        meta = metadata[idx] if idx < len(metadata) else {}
        context_pieces.append(chunk_text)
    return "\n\n".join(context_pieces)

def generate_answer(conversation_history, context_text, rewritten_query):
    # Compose messages
    final_prompt = conversation_history + [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer the question ONLY using the following context."
"Do NOT use any information outside the context."
"Always provide citation. When you are writing answer using retrieved content, whenever you reference it, provide numbers. First reference is 1, second is 2 and so on. In the end, provide all citations with their number."
            "sometimes you can use words close othe retrieved content. for example, if said \"that novelist is good\", it can be rewritten as \"that writer is good\"."
"Here is the context:"
f"{context_text}"

        )
    },
    {
        "role": "user",
        "content": rewritten_query
    }
]

    completion = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=final_prompt,
        max_tokens=512,
        temperature=0.7
    )
    answer = completion.choices[0].message.content.strip()
    return answer

TOP_K = 15  # increase to top 15 chunks

def rag_pipeline(user_input, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})

    # 1. Rewrite query
    rewritten_query = rewrite_query(conversation_history)
    print(rewritten_query)
    # 2. Embed rewritten query
    query_embedding = embed_text(rewritten_query)

    # 3. Retrieve top 15 chunks
    retrieved_indices = search_index(query_embedding, TOP_K)

    # 4. Print retrieved chunks for debugging
    print("\n=== Top 15 Retrieved Chunks ===")
    for i, idx in enumerate(retrieved_indices):
        snippet = chunks[idx]["content"][:300].replace("\n", " ")  # first 300 chars no newline
        print(f"{i+1}. Chunk idx={idx}: {snippet}...")

    # 5. Build context from retrieved chunks only
    context_text = build_context(retrieved_indices)

    # 6. Generate answer with ONLY these chunks as context
    answer = generate_answer(conversation_history, context_text, rewritten_query)

    conversation_history.append({"role": "assistant", "content": answer})

    return answer, conversation_history


if __name__ == "__main__":
    print("RAG Chat ready! Type 'exit' or 'quit' to stop.")
    conv_history = []
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in ["exit", "quit"]:
            break
        try:
            response, conv_history = rag_pipeline(user_text, conv_history)
            print("\nAssistant:", response)
        except Exception as e:
            print(f"Error: {e}")
