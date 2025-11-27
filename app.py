from fastapi import FastAPI, Request
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from groq import Groq

# Rate Limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

# CORS
from fastapi.middleware.cors import CORSMiddleware

# Load env
load_dotenv()

app = FastAPI(title="AI Search Agent 💖")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Slow down queen 👑 Too many requests"}
    )

# API Clients
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Input Schema
class Query(BaseModel):
    question: str

    @field_validator("question")
    def check_length(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Query too short")
        return v


@app.get("/")
def home():
    return {"message": "AI Search Agent is running 🚀"}


# Web Search
def web_search(query):
    results = tavily.search(query, max_results=5)
    return results["results"]


# ✅ CLEAN ANSWER PROMPT
def ai_answer(question, context):
    prompt = f"""
Answer clearly in numbered points.

Rules:
- No markdown
- No symbols like *, +, or #
- Each point max 1 line
- Only 5 points
- Do not mention sources
- Simple human language

Question:
{question}

Data:
{context}

Output format:
1. ...
2. ...
3. ...
4. ...
5. ...
"""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


@app.post("/ask")
@limiter.limit("5/minute")
def ask_ai(request: Request, query: Query):
    try:
        results = web_search(query.question)
        answer = ai_answer(query.question, results)

        sources = [{"title": r["title"], "url": r["url"]} for r in results]

        return {
            "question": query.question,
            "answer": answer,
            "sources": sources
        }

    except Exception:
        return {
            "question": query.question,
            "answer": "AI brain crashed 😔 Try again in a moment",
            "sources": []
        }
