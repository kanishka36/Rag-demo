from fastapi import FastAPI, HTTPException
from rag_retriever import initialize_retriever, query_retriever

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    initialize_retriever()

@app.get("/query/")
async def read_query(question: str):
    try:
        answer = query_retriever(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
