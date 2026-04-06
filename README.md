## Evaluation results (RAGAS, 5-question synthetic testset)

| Metric | Score | Bar | Grade |
|--------|-------|-----|-------|
| Faithfulness      | 0.526 | `███████████░░░░░░░░░` | 🔴 Needs improvement |
| Answer relevancy  | 0.270 | `█████░░░░░░░░░░░░░░░` | 🔴 Needs improvement |
| Context precision | 0.000 | `░░░░░░░░░░░░░░░░░░░░` | 🔴 Needs improvement |
| Context recall    | 0.200 | `████░░░░░░░░░░░░░░░░` | 🔴 Needs improvement |

Evaluation runs fully locally — no OpenAI API required.
See [`evaluation/EVAL_REPORT.md`](evaluation/EVAL_REPORT.md) for per-question breakdown.

**Notable:** 4 of 5 queries triggered the automatic query rewriter,
demonstrating the self-correction loop working as designed.

## Architecture

(architecture.png)

adaptive-rag/
├── .env
├── .gitignore
├── pyproject.toml
│
├── backend/
│   ├── config.py
│   ├── main.py           ← FastAPI server
│   ├── models.py
│   └── rag/
│       ├── ingestion.py  ← document loading + ChromaDB
│       ├── retrievers.py ← semantic + BM25 + RRF
│       ├── router.py     ← zero-shot intent classifier
│       └── pipeline.py   ← full RAG loop with self-correction
│
├── evaluation/
│   ├── testset.py        ← synthetic Q&A generation
│   ├── evaluator.py      ← RAGAS runner
│   ├── report.py         ← markdown report generator
│   ├── testset.json      ← generated, commit this
│   └── EVAL_REPORT.md    ← generated, commit this ← the key artifact
│
├── frontend/
│   └── app.py            ← Streamlit UI
│
└── scripts/
    ├── ingest.py
    ├── verify_ingest.py
    ├── test_pipeline.py
    ├── test_api.py
    └── run_eval.py

## Running the full stack
```bash
# 1. setup (once)
uv run python -m scripts.setup

# 2. Start API
uv run uvicorn backend.main:app --reload

# 3. Start UI (separate terminal)
uv run streamlit run frontend/app.py

# 4. Run evaluation
uv run python -m scripts.run_eval

# Full run — generates testset + evaluates + writes report (~15–20 min on CPU)
uv run python -m scripts.run_eval

# Faster smoke test with fewer questions
uv run python -m scripts.run_eval --n 5

# If you've already generated the testset and just want to re-evaluate
uv run python -m scripts.run_eval --skip-gen
```