"""
End-to-end pipeline smoke test.
Run: uv run python -m scripts.test_pipeline
"""

import sys, logging
from pathlib import Path
from backend.rag.pipeline import AdaptiveRAGPipeline
from backend.models import QueryRequest

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# These three queries are crafted to exercise each routing path:
#  - Q1: Conceptual → should route to SEMANTIC
#  - Q2: Specific API name → should route to KEYWORD (or HYBRID)
#  - Q3: Mixed → should route to HYBRID
TEST_QUERIES = [
    "How does dependency injection work in FastAPI?",
    "What parameters does HTTPException accept?",
    "How do I use OAuth2PasswordBearer to secure a specific route endpoint?",
]

def main():
    print("\n" + "="*60)
    print("  Adaptive RAG Pipeline — End-to-End Test")
    print("="*60)

    pipeline = AdaptiveRAGPipeline()

    for i, q in enumerate(TEST_QUERIES, start=1):
        print(f"\n[Query {i}]: {q}")
        print("-" * 60)

        response = pipeline.query(QueryRequest(query=q))

        print(f"  Strategy  : {response.retrieval_strategy.value}")
        print(f"  Grade     : {response.relevance_grade.value}")
        print(f"  Confidence: {response.confidence.value}")
        print(f"  Retries   : {response.retry_count}")
        print(f"  Sources   : {len(response.sources)} chunks")
        print(f"\n  Answer preview:")
        print(f"  {response.answer[:300]}...")
        print(f"\n  Top source: {response.sources[0].source if response.sources else 'none'}")
        print(f"  Top score : {response.sources[0].relevance_score if response.sources else 0:.4f}")

    print("\n" + "="*60)
    print("  Test complete. If all 3 queries returned answers,")
    print("  you're ready to build the FastAPI layer (Phase 5).")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()