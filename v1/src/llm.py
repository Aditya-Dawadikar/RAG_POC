import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY. Add it to your .env file.")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict RAG assistant. "
                    "Answer only using the provided context. "
                    "If the answer is not present in the context, say: "
                    "'I don't know based on the provided context.'"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content