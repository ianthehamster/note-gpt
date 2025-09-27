from fastapi import FastAPI
from pydantic import BaseModel

# Import the new modules
from app.memory_search import search_memory
from app.llm_engine import get_llm_response

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/chat")
def chat(q: Query):

    # 1. Search the memory index for relevant chunks
    memories = search_memory(q.prompt, k=3)

    # 2. Pass memory + user input into the LLM engine
    reply = get_llm_response(memories, q.prompt)


    return {
        "input": q.prompt,
        "memories": memories,
        "lucy_reply": reply
    }