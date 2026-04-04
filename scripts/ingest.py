"""
One-time ingestion script. Run this first before starting the API server.

Usage:
    uv run python -m scripts.ingest

What it does:
    1. Loads FastAPI documentation pages (or your local .md files)
    2. Chunks them with code-aware splitting
    3. Embeds each chunk with all-MiniLM-L6-v2 (runs locally)
    4. Persists to ChromaDB at ./chroma_db/
"""

import logging
import sys
from pathlib import Path
from backend.config import settings
from backend.rag.ingestion import (
    load_from_directory,
    load_from_urls,
    build_vector_store,
    chunk_documents,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

FASTAPI_DOCS_URLS = [
    "https://fastapi.tiangolo.com/",
    "https://fastapi.tiangolo.com/tutorial/",
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
    "https://fastapi.tiangolo.com/tutorial/body/",
    "https://fastapi.tiangolo.com/tutorial/body-fields/",
    "https://fastapi.tiangolo.com/tutorial/request-files/",
    "https://fastapi.tiangolo.com/tutorial/middleware/",
    "https://fastapi.tiangolo.com/tutorial/cors/",
    "https://fastapi.tiangolo.com/tutorial/security/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/",
    "https://fastapi.tiangolo.com/tutorial/background-tasks/",
    "https://fastapi.tiangolo.com/tutorial/handling-errors/",
    "https://fastapi.tiangolo.com/tutorial/response-model/",
    "https://fastapi.tiangolo.com/advanced/",
    "https://fastapi.tiangolo.com/advanced/response-model/",
    "https://fastapi.tiangolo.com/advanced/custom-response/",
    "https://fastapi.tiangolo.com/deployment/",
    "https://fastapi.tiangolo.com/deployment/docker/",
]

def main()->None:
    docs_dir = Path(settings.docs_dir)
    has_local_docs = docs_dir.exists() and any(docs_dir.glob("**/*.md"))
    if has_local_docs:
        logger.info(f'Found local docs at {docs_dir} - loading from disc')
        documents = load_from_directory(docs_dir=str(docs_dir))

    else:
        logger.info("No local docs found — scraping FastAPI docs from web")
        documents = load_from_urls(urls=FASTAPI_DOCS_URLS)

    if not documents:
        logger.error('No documents loaded. Check your internet connection or docs_dir path')
        sys.exit(1)

    logger.info('Chunking documents...')
    chunks = chunk_documents(documents=documents)

    logger.info('Embedding and Indexing chunks...')
    build_vector_store(chunks=chunks)

    logger.info("✅ Ingestion complete.")
    logger.info(f"   Chunks indexed : {len(chunks)}")
    logger.info(f"   Vector store   : {settings.chroma_persist_dir}")
    logger.info(f"   Collection     : {settings.chroma_collection_name}")
    logger.info("\n   Next step: run 'uv run python -m scripts.verify_ingest' to test a query")

if __name__== '__main__':
    main()