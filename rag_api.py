from fastapi import FastAPI
from pydantic import BaseModel
from rag import retrieve_context

app = FastAPI()

class QueryRequest(BaseModel):
    query : str

@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    context, sources = retrieve_context(request.query)

    return {
        "context":context,
        "sources":sources
    }