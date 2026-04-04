"""
API integration smoke test.
Run: uv run python -m scripts.test_api
"""

import sys
import httpx

BASE = "http://localhost:8000"

def check(label: str, condition: bool, detail: str = ""):
    icon = "✅" if condition else "❌"
    print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))
    return condition

def main():
    print("\n" + "="*55)
    print("  Adaptive RAG API — Integration Test")
    print("="*55 + "\n")

    passed = True

    # 1. Health
    print("[1] Health check")
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
        passed &= check("Status 200", r.status_code == 200)
        data = r.json()
        passed &= check("Status field is 'ok'", data.get("status") == "ok")
        passed &= check("Docs indexed > 0", data.get("docs_indexed", 0) > 0,
                        f"{data.get('docs_indexed')} chunks")
    except Exception as e:
        check("API reachable", False, str(e))
        print("\n  Start the backend first:")
        print("  uv run uvicorn backend.main:app --reload\n")
        sys.exit(1)

    # 2. Valid query
    print("\n[2] Valid query")
    r = httpx.post(
        f"{BASE}/query",
        json={"query": "How do I add middleware in FastAPI?"},
        timeout=500,
    )
    passed &= check("Status 200", r.status_code == 200)
    data = r.json()
    passed &= check("Answer present", bool(data.get("answer")))
    passed &= check("Sources present", len(data.get("sources", [])) > 0,
                    f"{len(data.get('sources', []))} chunks")
    passed &= check("Strategy field present", "retrieval_strategy" in data,
                    data.get("retrieval_strategy"))
    passed &= check("Grade field present", "relevance_grade" in data,
                    data.get("relevance_grade"))
    passed &= check("Retry count >= 0", data.get("retry_count", -1) >= 0,
                    f"{data.get('retry_count')} retries")
    passed &= check("Timing header present",
                    "x-response-time" in r.headers,
                    r.headers.get("x-response-time"))

    # 3. Input validation
    print("\n[3] Input validation")
    r = httpx.post(f"{BASE}/query", json={"query": "hi"}, timeout=10)
    passed &= check("Short query rejected (422)", r.status_code == 422)
    r = httpx.post(f"{BASE}/query", json={}, timeout=10)
    passed &= check("Missing field rejected (422)", r.status_code == 422)

    # Summary
    print("\n" + "="*55)
    if passed:
        print("  ✅ All tests passed.")
        print("  Next: uv run streamlit run frontend/app.py")
    else:
        print("  ❌ Some tests failed — check logs above.")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()