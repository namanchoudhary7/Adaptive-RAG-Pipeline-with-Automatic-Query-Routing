"""Quick smoke test — run after ingest.py to verify retrieval works."""
import sys
from pathlib import Path
from backend.rag.ingestion import load_vector_store, get_doc_count

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    count = get_doc_count()
    if count == 0:
        print("❌ No documents found. Run 'uv run python -m scripts.ingest' first.")
        sys.exit(1)

    print(f"✅ Vector store healthy — {count} chunks indexed")

    store = load_vector_store()
    test_query = "How do I add a middleware in FastAPI?"
    results = store.similarity_search_with_score(test_query, k=3)

    print(f"\nTest query: '{test_query}'")
    print("-" * 60)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"    Source: {doc.metadata.get('source', 'unknown')}")
        print(f"    Content: {doc.page_content[:150]}...")

if __name__ == "__main__":
    main()