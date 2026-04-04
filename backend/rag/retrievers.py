"""
Three retrieval strategies + Reciprocal Rank Fusion merger.

Why three strategies?
- Semantic search (FAISS/Chroma) wins on conceptual queries:
    "how does authentication work" — finds relevant chunks even if exact
    words don't match, because the embedding captures meaning.
- BM25 keyword search wins on precise queries:
    "OAuth2PasswordBearer class" — exact term matching outperforms embeddings
    when the user knows the specific name/function they want.
- Hybrid (RRF) wins when you're unsure — combines both ranked lists
    into one, promoting chunks that rank well in both systems.

Reciprocal Rank Fusion formula:
    RRF_score(doc) = Σ  1 / (k + rank_in_list_i)
                   lists
    where k=60 is an empirically-established constant that dampens the
    impact of very high ranks without ignoring low-ranked results.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from langchain_classic.schema import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from backend.config import settings

logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    """A single retrieved document with its provenance and score."""
    document: Document
    score: float                       # Normalised 0–1 (higher = more relevant)
    strategy: str                      # "semantic" | "bm25" | "hybrid"
    rank: int                          # Position in the result list (1-indexed)

@dataclass
class RetrievedResult:
    """The full output of one retrieval call, passed downstream to the grader."""
    chunks: List[RetrievedChunk]
    strategy_used: str
    query_used: str

class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi with a LangChain-compatible interface.

    BM25Okapi uses probabilistic term frequency scoring — the same family
    of algorithms powering Elasticsearch's default scorer. It's significantly
    better than raw TF-IDF for retrieval tasks.
    """

    def __init__(self, documents: List[Document])->None:
        self._documents = documents
        # Tokenise by whitespace — good enough for English technical docs.
        # For production, swap with a proper tokeniser (e.g. spaCy, NLTK).
        tokenised = [doc.page_content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenised)
        logger.info(f'BM25 index built over {len(documents)} documents')

    def search(self, query: str, top_k: int)-> List[Tuple[Document, float]]:
        """Returns (document, normalised_score) pairs sorted by relevance."""
        tokens = query.lower().split()
        raw_scores = self._bm25.get_scores(tokens)
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        normalized = raw_scores/max_score

        scored_docs = sorted(
            zip(self._documents, normalized),
            key=lambda x: x[1],
            reverse=True
        )
        return scored_docs[:top_k]
    
    @classmethod
    def from_chroma(cls, vector_store: Chroma)->"BM25Index":
        """
        Bootstrap the BM25 index directly from the existing ChromaDB store
        so we have one source of truth for all documents.
        """
        collection = vector_store._collection
        result = collection.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata = meta)
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        return cls(docs)
    
def reciprocal_rank_fusion(ranked_lists: List[Tuple[Document, float]], k: int = 60, top_k: int = None)->List[RetrievedChunk]:
    """
    Merge multiple ranked retrieval results into one unified ranking.

    RRF is rank-based, not score-based — this is its key advantage. It
    doesn't matter that BM25 scores and cosine similarities live on
    different scales. We only care about each document's position in each
    list, not its raw score value.

    Args:
        ranked_lists: Each list is [(Document, score), ...] sorted by relevance.
        k:            RRF constant. k=60 is the standard from the original paper
                      (Cormack et al., 2009). Higher k = less reward for top ranks.
        top_k:        Return only the top N fused results. Defaults to settings.top_k.

    Returns:
        List of RetrievedChunk objects, sorted by fused RRF score (descending).
    """

    if top_k is None:
        top_k = settings.top_k
    
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list, start=1):
            # Use content hash as a unique key (avoids deduplication issues
            # when the same chunk appears in both semantic and BM25 results)
            doc_id = str(hash(doc.page_content))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc
    
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    max_rrf = rrf_scores[sorted_ids[0]] if sorted_ids else 1.0

    result = []
    for rank, doc_id in enumerate(sorted_ids[:top_k], start=1):
        result.append(
            RetrievedChunk(
                document=doc_map[doc_id],
                rank=rank,
                strategy="hybrid",
                score= round(rrf_scores[doc_id]/max_rrf, 4)
            )
        )
    return result

class AdaptiveRetriever:
    """
    The main retriever class. Holds references to both the vector store
    and the BM25 index, and exposes a single `retrieve()` method that
    dispatches to the right strategy based on the router's decision.
    """

    def __init__(self, vector_store: Chroma)->None:
        self._vector_store = vector_store
        self._bm25 = BM25Index.from_chroma(vector_store)
        logger.info("Adaptive retriever initialized (semantic + BM25 + hybrid)")

    def semantic_search(self, query:str, top_k:int = None)->List[RetrievedChunk]:
        """Cosine similarity search over embedded chunks."""
        if top_k is None:
            top_k = settings.top_k

        results = self._vector_store.similarity_search_with_relevance_scores(query, k=top_k)

        return[
            RetrievedChunk(
                document=doc,
                rank=rank,
                score=score,
                strategy='semantic',
            )
            for rank, (doc, score) in enumerate(results, start=1)
        ]
    
    def bm25_search(self, query:str, top_k:int = None)->List[RetrievedChunk]:
        """BM25Okapi term-frequency search over all document chunks."""
        if top_k is None:
            top_k = settings.top_k
        
        results = self._bm25.search(query=query, top_k=top_k)

        return[
            RetrievedChunk(
                document=doc,
                rank=rank,
                score=score,
                strategy='bm25',
            )
            for rank, (doc, score) in enumerate(results, start=1)
        ]
    
    def hybrid_search(self, query:str, top_k:int = None)->List[RetrievedChunk]:
        """
        Run both semantic and BM25, then fuse with RRF.
        Fetches 2×top_k from each strategy so RRF has enough candidates
        to produce a high-quality top_k final list.
        """

        if top_k is None:
            top_k = settings.top_k
        
        fetch_k = 2*top_k
        
        semantic_results = self._vector_store.similarity_search_with_relevance_scores(query, k=fetch_k)
        bm25_results = self._bm25.search(query=query, top_k=fetch_k)

        semantic_list = [(doc, score) for doc, score in semantic_results]
        bm25_list = list(bm25_results)

        return reciprocal_rank_fusion(ranked_lists=[semantic_results, bm25_results], top_k=top_k)
    
    def retrieve(self, query:str, strategy:str, top_k:int = None)->RetrievedResult:
        """
        Main entry point. Called by the pipeline with the router's chosen strategy.
        """

        dispatch = {
            "semantic" : self.semantic_search,
            "bm25": self.bm25_search,
            "hybrid": self.hybrid_search,
        }

        if strategy not in dispatch:
            logger.info(f"Unknown strategy {strategy}, falling back to hybrid")
            strategy = "hybrid"

        chunks = dispatch[strategy](query=query, top_k=top_k)

        logger.info(
            f'Retrieved {len(chunks)} chunks | strategy = {strategy} | '
            f'top_score = {chunks[0].score if chunks else 0:.4f}'
        )

        return RetrievedResult(
            chunks=chunks,
            strategy_used=strategy,
            query_used=query,
        )