import logging
from typing import List
from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import settings

logger = logging.getLogger(__name__)

def load_from_directory(docs_dir:str)->List[Document]:
    """Load .md files from a local directory. Use this if you have docs locally."""
    loader = DirectoryLoader(
        docs_dir,
        glob='**/*.md',
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    logger.info(f'Loaded {len(docs)} documents from {docs_dir}')
    return docs

def load_from_urls(urls = List[str])->List[Document]:
    """
    Scrape documentation pages directly from the web.
    WebBaseLoader uses BeautifulSoup under the hood.
    This is the default for Phase 1 — no files to manage.
    """
    loader = WebBaseLoader(web_paths=urls, requests_per_second=2)
    docs = loader.load()

    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())

    logger.info(f'Loaded {len(docs)} documents from {len(urls)} URLs')
    return docs

def chunk_documents(documents: List[Document])->List[Document]:
    """
    Split documents into retrieval-optimised chunks.

    Key decisions made here:
    - chunk_size=1000: Large enough to contain full code examples + context
    - chunk_overlap=200: Ensures context isn't lost at chunk boundaries
    - Custom separators: Prioritise semantic boundaries (paragraphs, code
      blocks) over arbitrary character counts. This is especially important
      for technical docs which mix prose and code.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators=[
            '\n\n',
            '\n',
            '```',
            '. ',
            ' ',
            '',
        ],
        length_function = len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents=documents)
    chunks = [c for c in chunks if len(c.page_content.strip())>50]

    logger.info(f'Produced {len(chunks)} chunks from {len(documents)} documents')
    return chunks

def get_embeddings()->HuggingFaceEmbeddings:
    """
    Initialize the local HuggingFace embedding model.

    all-MiniLM-L6-v2 is a strong baseline — fast on CPU, 384-dim vectors,
    good semantic understanding. In Phase 4 (fine-tuning), we'll swap this
    out for a domain-adapted version trained on our own corpus.
    """
    return HuggingFaceEmbeddings(
        model_name = settings.embedding_model,
        model_kwargs = {'device': settings.embedding_device},
        encode_kwargs = {
            'normalize_embeddings': True,
            'batch_size': 32
        }
    )

def build_vector_store(chunks = List[Document])->Chroma:
    """
    Embed all chunks and persist to ChromaDB.
    This is the expensive one-time operation — takes 2–5 min on CPU.
    """
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    count = vector_store._collection.count()
    logger.info(f'Indexed {count} chunks -> {settings.chroma_persist_dir}')
    return vector_store

def load_vector_store()->Chroma:
    """
    Load an already-built ChromaDB store from disk.
    Used at server startup — fast, no re-embedding needed.
    """
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
    )

def get_doc_count()->int:
    """Quick health check — how many chunks are indexed?"""
    try:
        store = load_vector_store()
        return store._collection.count()
    except Exception:
        return 0