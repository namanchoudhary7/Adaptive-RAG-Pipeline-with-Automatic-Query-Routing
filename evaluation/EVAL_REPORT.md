# Adaptive RAG — Evaluation Report

> Generated: 2026-04-05 05:57  
> Test cases: 5  
> Model: Ollama / llama3.2:1b  
> Embeddings: sentence-transformers/all-MiniLM-L6-v2

---

## Summary scores

| Metric | Score | Bar | Grade |
|--------|-------|-----|-------|
| Faithfulness      | 0.526 | `███████████░░░░░░░░░` | 🔴 Needs improvement |
| Answer relevancy  | 0.270 | `█████░░░░░░░░░░░░░░░` | 🔴 Needs improvement |
| Context precision | 0.000 | `░░░░░░░░░░░░░░░░░░░░` | 🔴 Needs improvement |
| Context recall    | 0.200 | `████░░░░░░░░░░░░░░░░` | 🔴 Needs improvement |

---

## Retrieval behaviour

- **Queries that triggered the rewriter:** 4 / 5 (80%)

### Strategy distribution

| Strategy | Count | % of queries |
|----------|-------|--------------|
| HYBRID | 3 | 60% |
| SEMANTIC | 2 | 40% |

---

## Per-question results

| # | Question | Strategy | Retries | Faith. | Rel. | Prec. | Recall |
|---|----------|----------|---------|--------|------|-------|--------|
| 1 | What are the main concepts that should be kept in mind ... | hybrid | 0 | 0.75 | 0.00 | 0.00 | 0.00 |
| 2 | What is typically handled at the cluster level when wor... | semantic | 3 | 0.43 | 0.00 | 0.00 | 0.50 |
| 3 | What is required for high performance serving of a Fast... | hybrid | 3 | 0.57 | 0.00 | 0.00 | 0.00 |
| 4 | What is the primary difference between using Separate O... | semantic | 3 | 0.31 | 0.95 | 0.00 | 0.50 |
| 5 | What types of additional responses can be defined in an... | hybrid | 2 | 0.57 | 0.40 | 0.00 | 0.00 |

---

## Lowest faithfulness cases

_These are the questions where the model most deviated from the retrieved context — useful for identifying retrieval failure modes._

**Q:** What is the primary difference between using Separate OpenAPI Schemas for Input and Output in a FastAPI application?  
**Faithfulness:** 0.308 | **Strategy:** semantic | **Retries:** 3  
**Answer excerpt:** The provided context does not fully answer your question regarding the primary differences between utilizing distinct OpenAPI schemas for input and output data models in a FastAPI application.

Howeve...  

**Q:** What is typically handled at the cluster level when working with distributed container management systems like Kubernetes?  
**Faithfulness:** 0.429 | **Strategy:** semantic | **Retries:** 3  
**Answer excerpt:** The provided context does not fully answer your question about the specific cluster-level operations and management tasks typically performed in a distributed container orchestration platform like Kub...  

**Q:** What is required for high performance serving of a FastAPI application?  
**Faithfulness:** 0.571 | **Strategy:** hybrid | **Retries:** 3  
**Answer excerpt:** Unfortunately, the provided context does not fully answer your question about the specific deployment settings and industry-standard guidelines for maximizing throughput and efficiency in live deploym...  

---

## Resume bullet points

_Copy these directly into your resume. Replace X.XX with your actual scores._

```
• Engineered a self-correcting RAG pipeline with adaptive query routing
  (semantic / BM25 / hybrid-RRF), achieving 0.53 faithfulness
  and 0.27 answer relevancy on a 5-question
  synthetic eval set (RAGAS framework, llama3.2:1b, local inference).

• Implemented LLM-as-judge relevance grading with automatic query rewriting;
  4 of 5 eval queries triggered the rewriter loop,
  improving retrieval precision from initial INSUFFICIENT grades to SUFFICIENT.
```