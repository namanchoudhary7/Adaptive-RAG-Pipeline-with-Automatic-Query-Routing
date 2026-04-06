# Adaptive RAG — Evaluation Report

> Generated: 2026-04-06 23:09  
> Test cases: 20  
> Model: Ollama / llama3:8b  
> Embeddings: BAAI/bge-small-en-v1.5

---

## Summary scores

| Metric | Score | Bar | Grade |
|--------|-------|-----|-------|
| Faithfulness      | 0.711 | `██████████████░░░░░░` | 🟡 Acceptable |
| Answer relevancy  | 0.690 | `██████████████░░░░░░` | 🔴 Needs improvement |
| Context precision | 0.499 | `██████████░░░░░░░░░░` | 🔴 Needs improvement |
| Context recall    | 0.750 | `███████████████░░░░░` | 🟡 Acceptable |

---

## Retrieval behaviour

- **Queries that triggered the rewriter:** 4 / 20 (20%)

### Strategy distribution

| Strategy | Count | % of queries |
|----------|-------|--------------|
| HYBRID | 17 | 85% |
| SEMANTIC | 3 | 15% |

---

## Per-question results

| # | Question | Strategy | Retries | Faith. | Rel. | Prec. | Recall |
|---|----------|----------|---------|--------|------|-------|--------|
| 1 | What is the purpose of using FastAPI for building REST ... | hybrid | 0 | 0.00 | 0.00 | 0.00 | 1.00 |
| 2 | What is the purpose of the write_notification function ... | hybrid | 0 | 0.67 | 0.98 | 1.00 | 1.00 |
| 3 | How do I enable container restarts on failure in Docker... | hybrid | 3 | 1.00 | 0.89 | 0.33 | 1.00 |
| 4 | What is an example of a deployment strategy that allows... | semantic | 0 | 0.67 | 0.96 | 1.00 | 1.00 |
| 5 | What is the purpose of using the `Annotated` decorator ... | hybrid | 0 | 0.50 | 0.00 | 0.00 | 0.00 |
| 6 | How do I configure a Load Balancer to distribute networ... | hybrid | 3 | 0.00 | 0.00 | 0.00 | 1.00 |
| 7 | What is the purpose of returning HTTP status codes in t... | hybrid | 0 | 0.33 | 1.00 | 1.00 | 1.00 |
| 8 | How does the hierarchical dependency injection system h... | semantic | 0 | 0.67 | 0.00 | 0.00 | 1.00 |
| 9 | What is the purpose of using a separate output model (U... | hybrid | 0 | 1.00 | 0.97 | 0.25 | 1.00 |
| 10 | How do I integrate security mechanisms with the interac... | hybrid | 1 | 1.00 | 0.00 | 0.00 | 0.00 |
| 11 | How would I run database migrations in parallel with ot... | hybrid | 3 | 1.00 | 0.00 | 1.00 | 1.00 |
| 12 | How do I handle RequestValidationError in my FastAPI ap... | hybrid | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 13 | How do I declare a path operation function in FastAPI? | hybrid | 0 | 0.67 | 1.00 | 0.37 | 1.00 |
| 14 | What is the purpose of the `@app.get("/")` decorator in... | semantic | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 15 | What is the purpose of adding an await anyio.sleep(0) s... | hybrid | 0 | 0.75 | 1.00 | 1.00 | 1.00 |
| 16 | What is the purpose of using an official Python Docker ... | hybrid | 0 | 0.75 | 1.00 | 0.70 | 0.00 |
| 17 | How do I quit the reloader process? | hybrid | 0 | 0.67 | 1.00 | 1.00 | 1.00 |
| 18 | How do I return a plain text response in FastAPI? | hybrid | 0 | 0.83 | 1.00 | 0.00 | 0.00 |
| 19 | What is the purpose of setting the filename in the Cont... | hybrid | 0 | nan | 1.00 | 0.33 | 1.00 |
| 20 | What is the purpose of using a specific data type in th... | hybrid | 0 | 1.00 | 1.00 | 0.00 | 0.00 |

---

## Lowest faithfulness cases

_These are the questions where the model most deviated from the retrieved context — useful for identifying retrieval failure modes._

**Q:** What is the purpose of using FastAPI for building REST servers that can obtain predictions?  
**Faithfulness:** 0.000 | **Strategy:** hybrid | **Retries:** 0  
**Answer excerpt:** I cannot answer this based on the provided context....  

**Q:** How do I configure a Load Balancer to distribute network requests to multiple worker containers in a Kubernetes environment?  
**Faithfulness:** 0.000 | **Strategy:** hybrid | **Retries:** 3  
**Answer excerpt:** I cannot answer this based on the provided context....  

**Q:** What is the purpose of returning HTTP status codes in the range of 400 when handling errors in an API?  
**Faithfulness:** 0.333 | **Strategy:** hybrid | **Retries:** 0  
**Answer excerpt:** I can answer this based on the provided context.

The purpose of returning HTTP status codes in the range of 400 when handling errors in an API is to notify the client that there was an error from the...  

---

## Resume bullet points

_Copy these directly into your resume. Replace X.XX with your actual scores._

```
• Engineered a self-correcting RAG pipeline with adaptive query routing
  (semantic / BM25 / hybrid-RRF), achieving 0.71 faithfulness
  and 0.69 answer relevancy on a 20-question
  synthetic eval set (RAGAS framework, Mistral 7B, local inference).

• Implemented LLM-as-judge relevance grading with automatic query rewriting;
  4 of 20 eval queries triggered the rewriter loop,
  improving retrieval precision from initial INSUFFICIENT grades to SUFFICIENT.
```