import json
import os
import numpy as np
from openai import OpenAI

# Set your OpenAI API key as env var OPENAI_API_KEY
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Set your OPENAI_API_KEY environment variable")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Define the model you want to use
COMPLETION_MODEL = "gpt-4o-mini"

# Load document names
with open("output.json", "r") as f:
    docs = json.load(f)

docs_text = "Available documents:\n" + "\n".join(docs)

def generate_answer(conversation_history, rewritten_query):
    # Prepare the prompt
    final_prompt = conversation_history + [
        {
            "role": "system",
            "content": (
                "You are an assistant that can only answer questions using the document names provided. "
                "You should provide answers with proper citations from those documents. "
                "In the end, you must cite the name of book as Document/Chapter and wrap them in angled brackets like <<Document/Chapter.htm>>. "
                "Do NOT use any other knowledge or information. "
                "Do not mention that you are limited to a list. "
                "If the user is talking casually, engage with them in the same manner."

                "\n\n"
                f"{docs_text}"
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
        max_tokens=1024,
        temperature=0.7,
        stream=True
    )

    answer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            part = chunk.choices[0].delta.content
            answer += part
            print(part, end="", flush=True)

    return answer, conversation_history + [{"role": "user", "content": rewritten_query},
                                           {"role": "assistant", "content": answer}]

if __name__ == "__main__":
    print("🕊️ DivineVoice RAG Chat ready! Type 'exit' or 'quit' to stop.\n")
    conv_history = []
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in ["exit", "quit"]:
            print("Grace and peace be with you. ✨")
            break
        try:
            response, conv_history = generate_answer(conv_history, user_text)
            print("\n")
        except Exception as e:
            print(f"\n😓 Error: {e}")
