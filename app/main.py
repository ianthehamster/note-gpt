from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/chat")
def chat(q: Query):
    return {"response": f"Echo: {q.prompt}"}