"""
FastAPI backend for the Adaptive RAG pipeline.

Endpoints:
    POST /query      — main RAG query endpoint
    GET  /health     — liveness + readiness check
    GET  /           — redirect to Swagger UI

Run with:
    uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging, time, os
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.config import settings
from backend.models import HealthResponse, QueryRequest, QueryResponse
from backend.rag.ingestion import get_doc_count
from backend.rag.pipeline import AdaptiveRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

_pipeline : AdaptiveRAGPipeline | None = None

def get_pipeline()->AdaptiveRAGPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail='Pipeline not initialised. Check server logs.'
        )
    return _pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup before any requests are served.
    Initialises the pipeline (loads ChromaDB, builds BM25, loads LLM chains).
    Any error here aborts startup — intentional, fail fast is better than
    silently serving broken responses.
    """
    global _pipeline

    logger.info(f"DEBUG: System Env OLLAMA_MODEL = {os.environ.get('OLLAMA_MODEL')}")

    logger.info("Starting Adaptive RAG API...")
    logger.info(f"  Ollama model  : {settings.ollama_model}")
    logger.info(f"  Embedding     : {settings.embedding_model}")
    logger.info(f"  ChromaDB path : {settings.chroma_persist_dir}")

    doc_count = get_doc_count()
    if doc_count == 0:
        logger.error(
            "No documents found in ChromaDB. "
            "Run 'uv run python -m scripts.setup' first."
        )
        raise RuntimeError("Vector store is empty.")

    logger.info(f"  Docs indexed  : {doc_count} chunks")
    logger.info("Initialising pipeline (this takes ~10s on first run)...")

    _pipeline = AdaptiveRAGPipeline()

    logger.info("✅ API ready — visit http://localhost:8000/docs")

    yield  # Server runs here

    logger.info("Shutting down...")
    _pipeline = None


# --- INITIALIZE APP ---
app = FastAPI(
    title="Adaptive RAG API",
    description=(
        "A self-correcting Retrieval-Augmented Generation pipeline that routes "
        "queries through semantic, BM25, or hybrid retrieval strategies, "
        "grades its own context quality, and rewrites failed queries automatically."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allows the Streamlit frontend (running on :8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit dev
        "http://127.0.0.1:8501",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware('http')
async def app_timing_header(request: Request, call_next):
    """
    Adds X-Response-Time header to every response.
    Useful for profiling which queries trigger the rewrite loop
    (those take noticeably longer — 2–3× the baseline latency).
    """
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    return response

@app.get("/", include_in_schema=False)
async def root():
    """Redirect bare root to the Swagger UI — nicer for demos."""
    return RedirectResponse(url="/docs") # Added 'return' here!

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
)
async def health():
    """
    Liveness + readiness check.
    Returns 200 if the pipeline is initialised and ChromaDB has documents.
    Returns 503 if the pipeline failed to start.
    """
    # Run get_doc_count and get_pipeline concurrently
    # Note: Wrap get_doc_count in run_in_executor if it becomes a heavy sync call
    doc_count_task = asyncio.to_thread(get_doc_count)
    pipeline_task = asyncio.to_thread(get_pipeline)
    
    doc_count, pipeline = await asyncio.gather(doc_count_task, pipeline_task)

    return HealthResponse(
        status="ok",
        ollama_model=settings.ollama_model,
        embedding_model=settings.embedding_model,
        docs_indexed=doc_count,
    )

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG pipeline",
    tags=["RAG"],
    responses={
        200: {"description": "Successful RAG response with sources and metadata"},
        503: {"description": "Pipeline not ready — run ingest.py first"},
        422: {"description": "Invalid request body"},
    },
)
async def query_endpoint(request:QueryRequest):
    """
    Submit a natural language question about the indexed technical documentation.
    """
    pipeline = get_pipeline()

    try:
        response = pipeline.query(request=request)
    except Exception as e:
        logger.exception(f"Pipeline error for query - '{request.query}'")
        raise HTTPException(
            status_code=500,
            detail=f'Pipeline error: {str(e)}'
        )
    
    return response