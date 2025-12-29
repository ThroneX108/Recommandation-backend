from fastapi import FastAPI, HTTPException
from src.model import mental_health_rag
from pydantic import BaseModel
from typing import List


app = FastAPI(title="Mental Health RAG API")

@app.get("/")
def health_check():
    return {"status": "OK", "message": "Mental Health RAG API running"}


class MentalHealthAnswer(BaseModel):
    cause: List[str]
    recommended_actions: List[str]

class QueryResponse(BaseModel):
    answer: MentalHealthAnswer

class QueryRequest(BaseModel): 
    question: str


@app.post("/query", response_model=QueryResponse)
def query_rag(data: QueryRequest):
    try:
        result = mental_health_rag(data.question) 

        return QueryResponse(answer=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

