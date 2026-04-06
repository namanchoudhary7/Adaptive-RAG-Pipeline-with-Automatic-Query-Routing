"""
RAGAS evaluation runner.

RAGAS (Retrieval Augmented Generation Assessment) evaluates RAG pipelines
across four dimensions. We run it by:
  1. Taking each test question and running it through our full pipeline
  2. Collecting the answer, retrieved contexts, and ground truth
  3. Passing everything to RAGAS which uses an LLM judge internally

Note on the LLM judge: RAGAS uses an LLM to compute metrics like
faithfulness. We configure it to use our local Ollama model so the
entire evaluation runs offline with no API costs.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging, math
from dataclasses import dataclass, field
from typing import List

from datasets import Dataset
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from backend.config import settings
from backend.models import QueryRequest
from backend.rag.pipeline import AdaptiveRAGPipeline
from backend.rag.retrievers import AdaptiveRetriever
from backend.rag.ingestion import load_vector_store
from evaluation.testset import TestCase

logger = logging.getLogger(__name__)

@dataclass
class EvalResult:
    question: str
    answer: str
    ground_truth: str
    contexts: List[str]
    retrieval_strategy: str
    retry_count: int
    # Populated after RAGAS scoring
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

@dataclass
class EvalSummary:
    results: List[EvalResult] = field(default_factory=list)
    mean_faithfulness: float = 0.0
    mean_answer_relevancy: float = 0.0
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0
    n_retried: int = 0
    n_total: int = 0

    def compute_means(self) -> None:
        if not self.results:
            return
        
        # Filter out NaN values for each metric
        valid_faith = [r.faithfulness for r in self.results if not math.isnan(r.faithfulness)]
        valid_rel = [r.answer_relevancy for r in self.results if not math.isnan(r.answer_relevancy)]
        valid_prec = [r.context_precision for r in self.results if not math.isnan(r.context_precision)]
        valid_recall = [r.context_recall for r in self.results if not math.isnan(r.context_recall)]

        # Calculate means safely (defaults to 0.0 if the filtered list is empty)
        self.mean_faithfulness = round(sum(valid_faith) / len(valid_faith) if valid_faith else 0.0, 4)
        self.mean_answer_relevancy = round(sum(valid_rel) / len(valid_rel) if valid_rel else 0.0, 4)
        self.mean_context_precision = round(sum(valid_prec) / len(valid_prec) if valid_prec else 0.0, 4)
        self.mean_context_recall = round(sum(valid_recall) / len(valid_recall) if valid_recall else 0.0, 4)
        
        self.n_retried = sum(1 for r in self.results if r.retry_count > 0)
        self.n_total = len(self.results)

class RAGASEvaluator:

    def __init__(self) -> None:
        self._pipeline = AdaptiveRAGPipeline()

        # Point RAGAS at our local Ollama model so eval is fully offline
        ollama_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,   # Deterministic grading
        )
        local_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._ragas_llm        = LangchainLLMWrapper(ollama_llm)
        self._ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    def _run_pipeline(self, test_case: TestCase) -> EvalResult:
        """Run one test case through the full RAG pipeline."""
        response = self._pipeline.query(QueryRequest(query=test_case.question))

        return EvalResult(
            question=test_case.question,
            answer=response.answer,
            ground_truth=test_case.ground_truth,
            # RAGAS expects a list of context strings — one per retrieved chunk
            contexts=[s.content for s in response.sources],
            retrieval_strategy=response.retrieval_strategy.value,
            retry_count=response.retry_count,
        )

    def evaluate(self, test_cases: List[TestCase]) -> EvalSummary:
        """
        Run the full pipeline on all test cases, then score with RAGAS.

        Two-phase approach:
          Phase A — collect pipeline outputs (slow, LLM inference per question)
          Phase B — batch score with RAGAS (also slow, LLM-as-judge)
        """
       
        # Phase A: Parallel pipeline inference
        eval_results: List[EvalResult] = []
        
        # Adjust max_workers based on your provider:
        # - Groq Free Tier: 2 or 3 (avoid 429s)
        # - Local Ollama (CPU): 1 or 2
        # - Local Ollama (GPU): 4 to 8
        MAX_WORKERS = 3

        logger.info(f"🚀 Starting parallel inference with {MAX_WORKERS} workers...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map test cases to the worker function
            future_to_case = {executor.submit(self._run_pipeline, tc): tc for tc in test_cases}
            
            for i, future in enumerate(as_completed(future_to_case), 1):
                tc = future_to_case[future]
                try:
                    result = future.result()
                    eval_results.append(result)
                    logger.info(f"✅ [{i}/{len(test_cases)}] Finished: {tc.question[:50]}...")

                except Exception as e:
                    logger.error(f"❌ PIPELINE CRASHED ON: {tc.question[:50]}")
                    logger.error(f"Error: {str(e)}")
                    # Shut down the executor and abort
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f"Evaluation aborted due to pipeline failure: {e}")

        logger.info(f"Pipeline complete. Scoring {len(eval_results)} results with RAGAS...")

        # Phase B: RAGAS scoring
        # RAGAS expects a HuggingFace Dataset with specific column names
        dataset = Dataset.from_dict({
            "question":     [r.question     for r in eval_results],
            "answer":       [r.answer       for r in eval_results],
            "contexts":     [r.contexts     for r in eval_results],
            "ground_truth": [r.ground_truth for r in eval_results],
        })

        scores = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
            raise_exceptions=False,   # Log failures, don't abort the whole eval
            run_config=RunConfig(
                max_workers=4,       # Run one at a time to prevent CPU overload
                timeout=None,         
            )
        )

        scores_df = scores.to_pandas()

        # Merge RAGAS scores back into our result objects
        for i, result in enumerate(eval_results):
            result.faithfulness      = round(float(scores_df.iloc[i].get("faithfulness",      0.0)), 4)
            result.answer_relevancy  = round(float(scores_df.iloc[i].get("answer_relevancy",  0.0)), 4)
            result.context_precision = round(float(scores_df.iloc[i].get("context_precision", 0.0)), 4)
            result.context_recall    = round(float(scores_df.iloc[i].get("context_recall",    0.0)), 4)

        summary = EvalSummary(results=eval_results)
        summary.compute_means()

        logger.info(
            f"Evaluation complete.\n"
            f"  Faithfulness      : {summary.mean_faithfulness}\n"
            f"  Answer relevancy  : {summary.mean_answer_relevancy}\n"
            f"  Context precision : {summary.mean_context_precision}\n"
            f"  Context recall    : {summary.mean_context_recall}\n"
            f"  Queries retried   : {summary.n_retried}/{summary.n_total}"
        )

        return summary