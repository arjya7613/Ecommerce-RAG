from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import load_pipeline, ask_question

# Initialize app
app = FastAPI(title="Ecommerce RAG API")

# Load RAG pipeline once at startup
chain = load_pipeline()


# Request schema
class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Chat endpoint using RAG pipeline (no database)
    """
    result = ask_question(chain, request.query)
    return {
        "query": request.query,
        "answer": result
    }
