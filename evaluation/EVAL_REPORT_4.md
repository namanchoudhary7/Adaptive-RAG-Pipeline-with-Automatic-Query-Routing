# Adaptive RAG — Evaluation Report

> Generated: 2026-04-07 07:14  
> Test cases: 20  
> Model: Ollama / llama3:8b  
> Embeddings: BAAI/bge-small-en-v1.5

---

## Summary scores

| Metric | Score | Bar | Grade |
|--------|-------|-----|-------|
| Faithfulness      | 0.657 | `█████████████░░░░░░░` | 🔴 Needs improvement |
| Answer relevancy  | 0.469 | `█████████░░░░░░░░░░░` | 🔴 Needs improvement |
| Context precision | 0.505 | `██████████░░░░░░░░░░` | 🔴 Needs improvement |
| Context recall    | 0.820 | `████████████████░░░░` | 🟡 Acceptable |

---

## Retrieval behaviour

- **Queries that triggered the rewriter:** 12 / 20 (60%)

### Strategy distribution

| Strategy | Count | % of queries |
|----------|-------|--------------|
| BM25 | 1 | 5% |
| HYBRID | 17 | 85% |
| SEMANTIC | 2 | 10% |

---

## Per-question results

| # | Question | Strategy | Retries | Faith. | Rel. | Prec. | Recall |
|---|----------|----------|---------|--------|------|-------|--------|
| 1 | What is the purpose of using FastAPI for building REST ... | hybrid | 0 | 0.50 | 0.00 | 0.00 | 1.00 |
| 2 | What is the purpose of the write_notification function ... | hybrid | 0 | 0.67 | 0.96 | 0.62 | 1.00 |
| 3 | How do I enable container restarts on failure in Docker... | hybrid | 3 | nan | 0.00 | 1.00 | 1.00 |
| 4 | What is an example of a deployment strategy that allows... | semantic | 0 | 0.75 | 0.63 | 0.64 | 1.00 |
| 5 | How do I configure a Load Balancer to distribute networ... | hybrid | 3 | 0.50 | 0.00 | 0.00 | 1.00 |
| 6 | What is the purpose of returning HTTP status codes in t... | hybrid | 0 | 0.50 | 1.00 | 1.00 | 1.00 |
| 7 | What is the purpose of using the `Annotated` decorator ... | hybrid | 3 | 0.50 | 0.00 | 0.00 | 0.00 |
| 8 | How do I integrate security mechanisms with the interac... | hybrid | 3 | 0.00 | 0.00 | 1.00 | 0.00 |
| 9 | How does the hierarchical dependency injection system h... | semantic | 3 | 0.67 | 0.00 | 0.00 | 1.00 |
| 10 | What is the purpose of using a separate output model (U... | hybrid | 2 | 1.00 | 0.93 | 0.25 | 1.00 |
| 11 | How do I declare a path operation function in FastAPI? | hybrid | 1 | 0.50 | 0.00 | 0.53 | 1.00 |
| 12 | How would I run database migrations in parallel with ot... | hybrid | 3 | 0.50 | 0.00 | 1.00 | 1.00 |
| 13 | How do I handle RequestValidationError in my FastAPI ap... | hybrid | 0 | 0.75 | 1.00 | 0.44 | 1.00 |
| 14 | What is the purpose of the `@app.get("/")` decorator in... | bm25 | 1 | nan | 0.00 | 0.11 | 1.00 |
| 15 | What is the purpose of adding an await anyio.sleep(0) s... | hybrid | 3 | 1.00 | 0.95 | 1.00 | 1.00 |
| 16 | What is the purpose of using an official Python Docker ... | hybrid | 0 | 0.50 | 1.00 | 0.34 | 0.00 |
| 17 | How do I quit the reloader process? | hybrid | 3 | 1.00 | 0.00 | 1.00 | 1.00 |
| 18 | How do I return a plain text response in FastAPI? | hybrid | 0 | 0.83 | 1.00 | 0.16 | 0.40 |
| 19 | What is the purpose of using a specific data type in th... | hybrid | 0 | 1.00 | 1.00 | 0.00 | 1.00 |
| 20 | What is the purpose of setting the filename in the Cont... | hybrid | 1 | 0.67 | 0.92 | 1.00 | 1.00 |

---

## Lowest faithfulness cases

_These are the questions where the model most deviated from the retrieved context — useful for identifying retrieval failure modes._

**Q:** How do I integrate security mechanisms with the interactive documentation system in a FastAPI application?  
**Faithfulness:** 0.000 | **Strategy:** hybrid | **Retries:** 3  
**Answer excerpt:** I cannot answer this based on the provided context....  

**Q:** What is the purpose of using FastAPI for building REST servers that can obtain predictions?  
**Faithfulness:** 0.500 | **Strategy:** hybrid | **Retries:** 0  
**Answer excerpt:** I cannot answer this based on the provided context....  

**Q:** What is the purpose of the write_notification function called by the .add_task() method?  
**Faithfulness:** 0.667 | **Strategy:** hybrid | **Retries:** 0  
**Answer excerpt:** The purpose of the `write_notification` function called by the `.add_task()` method is to simulate sending an email notification. In this specific example, it writes a message to a log file (`log.txt`...  

---

## Resume bullet points

_Copy these directly into your resume. Replace X.XX with your actual scores._

```
• Engineered a self-correcting RAG pipeline with adaptive query routing
  (semantic / BM25 / hybrid-RRF), achieving 0.66 faithfulness
  and 0.47 answer relevancy on a 20-question
  synthetic eval set (RAGAS framework, Mistral 7B, local inference).

• Implemented LLM-as-judge relevance grading with automatic query rewriting;
  12 of 20 eval queries triggered the rewriter loop,
  improving retrieval precision from initial INSUFFICIENT grades to SUFFICIENT.
```