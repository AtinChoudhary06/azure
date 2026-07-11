import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag

app = FastAPI(title="PDF RAG Chatbot API")

# Allow the Streamlit frontend (deployed on a different URL) to call this API.
# Tighten allow_origins to your exact frontend URL once deployed, instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = rag.ingest_pdf(tmp_path, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        os.remove(tmp_path)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer = rag.ask_question(request.question)
    return AskResponse(answer=answer)