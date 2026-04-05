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

## Running the full stack
```bash
# 1. Ingest docs (once)
uv run python -m scripts.ingest

# 2. Start API
uv run uvicorn backend.main:app --reload

# 3. Start UI (separate terminal)
uv run streamlit run frontend/app.py

# 4. Run evaluation
uv run python -m scripts.run_eval
```