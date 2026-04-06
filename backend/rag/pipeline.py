"""
The full adaptive RAG pipeline.

Execution flow:
    1. Router classifies query → picks retrieval strategy
    2. AdaptiveRetriever fetches chunks using chosen strategy
    3. RelevanceGrader scores context quality (LLM-as-judge)
    4. If grade=sufficient → Generator produces answer
       If grade=insufficient → QueryRewriter rewrites query
                             → Go back to step 2 (max N retries)
    5. Return structured QueryResponse with source citations

The retry loop is what separates this from a naive RAG demo.
"""

import logging
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.config import settings
from backend.models import (
    ConfidenceLevel,
    RetrievalStrategy,
    GradeResult,
    QueryRequest,
    QueryResponse,
    SourceDocument
)
from backend.rag.ingestion import load_vector_store
from backend.rag.retrievers import AdaptiveRetriever, RetrievedResult
from backend.rag.router import QueryRouter

logger = logging.getLogger(__name__)

# LLM-as-judge: the model grades its own context before generating an answer.
# Structured output makes parsing reliable — we look for a single word.
RELEVANCE_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are an extremely strict, uncompromising relevance grader for a Retrieval-Augmented Generation (RAG) pipeline.
     Your ONLY responsibility is to determine if the provided context contains the exact, explicit information required to fully answer the user's question.

     CRITICAL RULES:
     1. NO PARTIAL CREDIT: If the context only provides partial information or hints at the answer, it is INSUFFICIENT.
     2. NO OUTSIDE KNOWLEDGE: If you have to guess, infer, or rely on your own internal knowledge to piece together the answer, it is INSUFFICIENT.
     3. BE RUTHLESS: Do not try to be helpful. If the explicit answer is missing from the text, you must reject it.

     OUTPUT FORMAT:
     You must respond with EXACTLY one word. Choose only from:
     SUFFICIENT
     INSUFFICIENT
     
     Do not add punctuation, explanations, or introductory text."""
    ),
    ("human",
     """Question: {question}

     Retrieved Context:
     {context}

     Grade:"""
    ),
])
# Query rewriter: produces a better query when retrieval failed.
# We tell the model exactly why we're asking — this improves output quality.
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a query optimisation assistant. A user's question was sent to a
     retrieval system, but the retrieved documents were not relevant enough.
     Your job is to rewrite the question to improve retrieval quality.\n\n
     Rules:\n
       - Preserve the original intent completely\n
       - Make it more specific: add technical synonyms, expand acronyms\n
       - Remove ambiguous pronouns ('it', 'this', 'that')\n
       - Output ONLY the rewritten question — no explanation, no prefix"""
    ),
    ("human", "Original question: {question}\n\nRewritten question:"),
])

# Main answer generation prompt
RAG_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a strict, factual technical documentation assistant. Your ONLY job is to synthesize an answer from the provided context.

     CRITICAL RULES:
     1. GROUNDING: You must answer the user's question using ONLY the provided context. Do NOT use outside knowledge, general programming advice, or your own internal memory.
     2. THE REFUSAL PROTOCOL: If the context does not contain the explicit answer to the user's question, you MUST output exactly: "I cannot answer this based on the provided context." Do NOT add any extra commentary, guesses, or apologies.
     3. NO HALLUCINATION: Do not infer or make assumptions. If a detail or parameter is not explicitly written in the text, do not include it in your answer.
     4. FORMATTING: Be concise and precise. Include exact code snippets from the context if they are relevant to the answer."""
    ),
    ("human",
     """Context:
     {context}

     Question: {question}

     Answer:"""
    ),
])

def _format_context(result: RetrievedResult)->str:
    """
    Concatenate retrieved chunks into a single context string.
    Each chunk is prefixed with its source URL so the generator can
    cite it, and its rank/score for transparency in logs.
    """
    sections = []
    for chunk in result.chunks:
        source = chunk.document.metadata.get("source", "unknown")
        sections.append(
            f"""Source: {source}\n 
            Page content: {chunk.document.page_content}"""
        )
    return "\n\n---\n\n".join(sections)

def _extract_sources(result:RetrievedResult)->list[SourceDocument]:
    return [
        SourceDocument(
            content=chunk.document.page_content[:200] + "...",
            source=chunk.document.metadata.get("source", "unknown"),
            relevance_score=chunk.score,
        )
        for chunk in result.chunks
    ]

def _score_confidence(grade: GradeResult, retry_count: int, top_score: float)->ConfidenceLevel:
    """Simple heuristic — can be replaced with a calibrated model later."""
    if grade == GradeResult.INSUFFICIENT:
        return ConfidenceLevel.LOW
    if retry_count>0 and top_score < 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.HIGH

class AdaptiveRAGPipeline:
    """
    Wires together: router → retriever → grader → (rewriter) → generator.

    This class is instantiated once at server startup and shared across
    all requests (it's stateless per-request — safe to share).
    """

    def __init__(self)->None:
        logger.info("Initializing AdaptiveRAGPipeline...")

        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,   # Low temp for factual Q&A
        )

        parser = StrOutputParser()

        # Build chains by compositing prompt | llm | parser
        # This is standard LangChain Expression Language (LCEL)
        self._grader_chain    =   RELEVANCE_GRADE_PROMPT   | llm | parser
        self._rewriter_chain  =   QUERY_REWRITE_PROMPT     | llm | parser
        self._generator_chain =   RAG_GENERATION_PROMPT    | llm | parser

        # Load the pre-built ChromaDB index
        vector_store = load_vector_store()
        self._retriever = AdaptiveRetriever(vector_store=vector_store)
        self._router = QueryRouter()

        logger.info("AdaptiveRAGPipeline ready...")

    def _grade_context(self, question:str, context: str)->GradeResult:
        """Ask the LLM to judge whether retrieved context is good enough."""
        raw: str = self._grader_chain.invoke({
            "question": question,
            "context": context,
        }).strip().upper()

        # Defensive parsing — LLMs sometimes add punctuation or extra words
        if "SUFFICIENT" in raw and "INSUFFICIENT" not in raw:
            return GradeResult.SUFFICIENT
        return GradeResult.INSUFFICIENT
    
    def _rewrite_query(self, original_query:str)->str:
        """Generate an improved query when retrieval failed."""
        rewritten: str = self._rewriter_chain.invoke({"question": original_query}).strip()
        logger.info(f"Query rewritten: {original_query} -> {rewritten}")
        return rewritten
    
    def query(self, request: QueryRequest)->QueryResponse:
        """
        Execute the full adaptive RAG loop.

        The retry loop is bounded by settings.max_retry_loops (default=2).
        On each failed attempt, the query is rewritten and re-routed.
        After exhausting retries, we generate a best-effort answer
        from whatever context we have — never silently fail.
        """

        current_query      = request.query
        retry_count        = 0
        grade              = GradeResult.INSUFFICIENT
        strategy_used      = RetrievalStrategy.HYBRID
        retrieval_result : Optional[RetrievedResult] = None

        while retry_count<=settings.max_retry_loops:
            # --- Step 1: Route ---
            strategy = self._router.route(query=current_query)
            strategy_used = strategy.value

            # --- Step 2: Retrieve ---
            retrieval_result = self._retriever.retrieve(current_query, strategy=strategy.value)

            if not retrieval_result.chunks:
                logger.info("Empty retrieval result - skipping grade, retrying")
                current_query = self._rewrite_query(current_query)
                retry_count += 1
                continue

            # --- Step 3: Grade ---
            context = _format_context(retrieval_result)
            grade = self._grade_context(question=current_query, context=context)

            if grade == GradeResult.SUFFICIENT:
                logger.info(f"Grade = sufficient after {retry_count} retries")
                break

            if retry_count< settings.max_retry_loops:
                logger.info(
                    f"Grade=INSUFFICIENT (attempt {retry_count + 1}) — rewriting query"
                )
                current_query = self._rewrite_query(current_query)
            retry_count+=1
        
         # --- Step 4: Generate (always runs, even after exhausted retries) ---
        
        context = _format_context(retrieval_result)
        answer: str = self._generator_chain.invoke({
            "context": context,
            "question": current_query
        })

        top_score = retrieval_result.chunks[0].score if retrieval_result.chunks else 0.0

        return QueryResponse(
            query=current_query,
            answer=answer.strip(),
            retrieval_strategy=strategy_used,
            relevance_grade=grade,
            retry_count=retry_count,
            sources= _extract_sources(retrieval_result),
            confidence=_score_confidence(grade=grade, retry_count=retry_count, top_score=top_score),
        )