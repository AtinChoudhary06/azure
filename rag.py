"""
Shared RAG pipeline: extract -> chunk -> embed -> index -> retrieve -> answer.

This is the ONLY place this logic lives. Both the FastAPI app and any CLI
script import from here, so there's no more drift between three copies of
the same function (which is what happened in the old main.py / query.py /
ingest_v2.py).
"""
import time
import uuid

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from openai import AzureOpenAI

from config import settings

# ---- Clients (created once, reused) ----
blob_service_client = BlobServiceClient.from_connection_string(settings.STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(settings.STORAGE_CONTAINER)

doc_intel_client = DocumentIntelligenceClient(
    endpoint=settings.DOCINTEL_ENDPOINT,
    credential=AzureKeyCredential(settings.DOCINTEL_KEY),
)

embed_client = AzureOpenAI(
    api_key=settings.EMBEDDING_KEY,
    api_version="2024-02-01",
    azure_endpoint=settings.EMBEDDING_ENDPOINT,
)

chat_client = ChatCompletionsClient(
    endpoint=settings.CHAT_ENDPOINT,
    credential=AzureKeyCredential(settings.CHAT_KEY),
    api_version="2024-05-01-preview",
)

search_client = SearchClient(
    endpoint=settings.SEARCH_ENDPOINT,
    index_name=settings.SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(settings.SEARCH_KEY),
)


# =====================================================
# Ingestion
# =====================================================

def upload_to_blob(file_path: str, filename: str) -> str:
    blob_client = container_client.get_blob_client(filename)
    with open(file_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    return blob_client.url


def extract_text(file_path: str) -> str:
    with open(file_path, "rb") as f:
        poller = doc_intel_client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=f.read()),
        )
    result = poller.result()
    return result.content


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    chunk_size = chunk_size or settings.CHUNK_SIZE_WORDS
    overlap = overlap or settings.CHUNK_OVERLAP_WORDS
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size (or you get an infinite loop)")

    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
    response = embed_client.embeddings.create(input=text, model=settings.EMBEDDING_DEPLOYMENT)
    return response.data[0].embedding


def delete_existing_chunks_for_source(filename: str) -> int:
    """
    Delete any chunks already indexed for this filename before re-indexing it.
    Without this, re-uploading the same PDF just piles up duplicate chunks
    forever and search quality degrades.
    """
    results = search_client.search(
        search_text="*",
        filter=f"source eq '{filename}'",
        select=["id"],
    )
    ids_to_delete = [{"id": r["id"]} for r in results]
    if ids_to_delete:
        search_client.delete_documents(documents=ids_to_delete)
    return len(ids_to_delete)


def ingest_pdf(file_path: str, filename: str) -> dict:
    """Full ingestion pipeline for a single PDF. Returns a summary dict."""
    blob_url = upload_to_blob(file_path, filename)
    text = extract_text(file_path)
    chunks = chunk_text(text)

    deleted = delete_existing_chunks_for_source(filename)

    documents_to_upload = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        vector = get_embedding(chunk)
        documents_to_upload.append({
            # UUIDs instead of a counted integer - avoids the string-sort
            # collision bug where "10" < "2" alphabetically could cause
            # duplicate/overwritten IDs once you passed 10 chunks.
            "id": str(uuid.uuid4()),
            "content": chunk,
            "content_vector": vector,
            "source": filename,
        })

    if documents_to_upload:
        search_client.upload_documents(documents=documents_to_upload)

    return {
        "filename": filename,
        "blob_url": blob_url,
        "chunks_indexed": len(documents_to_upload),
        "old_chunks_replaced": deleted,
    }


# =====================================================
# Retrieval + answering
# =====================================================

def search_chunks(question: str, top_k: int = None) -> list[tuple[str, str]]:
    top_k = top_k or settings.TOP_K
    vector = get_embedding(question)
    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content", "source"],
        top=top_k,
    )
    return [(r["content"], r["source"]) for r in results]


def ask_question(question: str, max_retries: int = 5, wait_seconds: int = 15) -> str:
    chunks = search_chunks(question)
    if not chunks:
        return "No relevant information found in the documents."

    context = "\n\n".join(f"[Source: {src}]\n{content}" for content, src in chunks)
    prompt = f"""Answer the question based only on the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    for attempt in range(1, max_retries + 1):
        try:
            completion = chat_client.complete(
                model=settings.CHAT_DEPLOYMENT,
                messages=[UserMessage(content=prompt)],
                max_tokens=300,
            )
            return completion.choices[0].message.content
        except HttpResponseError as e:
            # Azure AI Inference raises HttpResponseError for rate limits
            # (429) as well as auth/deployment errors - so we check the
            # status code to decide whether retrying makes sense.
            is_rate_limited = getattr(e, "status_code", None) == 429
            if attempt == max_retries:
                if is_rate_limited:
                    return "Sorry, the model is rate-limited and didn't respond after several retries."
                return f"Sorry, the chat model request failed: {getattr(e, 'message', str(e))}"
            time.sleep(wait_seconds)

    return "Sorry, something went wrong answering your question."