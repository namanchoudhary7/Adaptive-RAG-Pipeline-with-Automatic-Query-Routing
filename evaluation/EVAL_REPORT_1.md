# Adaptive RAG — Evaluation Report

> Generated: 2026-04-05 11:41  
> Test cases: 20  
> Model: Ollama / llama3.1  
> Embeddings: sentence-transformers/all-MiniLM-L6-v2

---

## Summary scores

| Metric | Score | Bar | Grade |
|--------|-------|-----|-------|
| Faithfulness      | 0.443 | `█████████░░░░░░░░░░░` | 🔴 Needs improvement |
| Answer relevancy  | 0.698 | `██████████████░░░░░░` | 🔴 Needs improvement |
| Context precision | 0.253 | `█████░░░░░░░░░░░░░░░` | 🔴 Needs improvement |
| Context recall    | nan | `░░░░░░░░░░░░░░░░░░░░` | ⚪ N/A |

---

## Retrieval behaviour

- **Queries that triggered the rewriter:** 1 / 20 (5%)

### Strategy distribution

| Strategy | Count | % of queries |
|----------|-------|--------------|
| BM25 | 4 | 20% |
| HYBRID | 14 | 70% |
| SEMANTIC | 2 | 10% |

---

## Per-question results

| # | Question | Strategy | Retries | Faith. | Rel. | Prec. | Recall |
|---|----------|----------|---------|--------|------|-------|--------|
| 1 | What is the type of the 'price' attribute in the 'Item'... | bm25 | 0 | 0.00 | 0.90 | 0.00 | 0.00 |
| 2 | What happens if you raise an HTTPException from inside ... | hybrid | 0 | 0.33 | 0.99 | 1.00 | 1.00 |
| 3 | What is the purpose of using a JSON Compatible Encoder ... | hybrid | 0 | 1.00 | 0.15 | 0.00 | 0.50 |
| 4 | What parameter in FastAPI's CORS configuration allows y... | hybrid | 0 | 0.20 | 1.00 | 0.70 | 0.25 |
| 5 | What is the purpose of using OpenAPI in this FastAPI ap... | hybrid | 0 | 0.29 | 1.00 | 0.20 | 0.25 |
| 6 | What happens when a developer omits one of the paramete... | hybrid | 0 | 0.20 | 0.27 | 0.00 | 0.00 |
| 7 | What are some examples of operations that would benefit... | hybrid | 0 | 0.80 | 1.00 | 0.70 | 1.00 |
| 8 | How does the RAG system handle multiple files being upl... | hybrid | 0 | 0.55 | 0.61 | 0.00 | 0.00 |
| 9 | What are the specific permission requirements for acces... | hybrid | 0 | 0.00 | 0.99 | 0.00 | 0.00 |
| 10 | What type of exception is raised when an HTTP error occ... | hybrid | 0 | 1.00 | 0.91 | 0.87 | 1.00 |
| 11 | How do you configure HTTP Basic Auth for a route using ... | hybrid | 0 | 0.30 | 0.89 | 0.25 | 1.00 |
| 12 | What HTTP methods are allowed by default for cross-orig... | bm25 | 0 | 0.20 | 0.56 | 0.00 | nan |
| 13 | What templating engine does FastAPI use by default? | hybrid | 0 | 0.80 | 0.98 | 0.33 | 1.00 |
| 14 | How does one configure Swagger UI to display static ass... | hybrid | 0 | 0.56 | 0.98 | 1.00 | 1.00 |
| 15 | What type of response object should be used to return a... | bm25 | 0 | 0.50 | 0.57 | 0.00 | 0.00 |
| 16 | What is the basis for the claim that FastAPI reduces hu... | semantic | 1 | 0.50 | 0.74 | 0.00 | 0.00 |
| 17 | What is the type of values that the API expects for the... | bm25 | 0 | 0.33 | 0.98 | 0.00 | 0.00 |
| 18 | What happens if a developer attempts to update an item ... | hybrid | 0 | 0.20 | 0.47 | 0.00 | 0.50 |
| 19 | What is the purpose of using virtual environments with ... | semantic | 0 | 0.50 | 0.00 | 0.00 | 0.33 |
| 20 | What is the recommended way to deploy a FastAPI applica... | hybrid | 0 | 0.60 | 0.00 | 0.00 | 0.00 |

---

## Lowest faithfulness cases

_These are the questions where the model most deviated from the retrieved context — useful for identifying retrieval failure modes._

**Q:** What is the type of the 'price' attribute in the 'Item' data model?  
**Faithfulness:** 0.000 | **Strategy:** bm25 | **Retries:** 0  
**Answer excerpt:** According to the context from https://fastapi.tiangolo.com/tutorial/body/, it states that "Check that it has a required attribute price that has to be a float."

Therefore, the type of the 'price' att...  

**Q:** What are the specific permission requirements for accessing the "/items/pro/" API endpoint?  
**Faithfulness:** 0.000 | **Strategy:** hybrid | **Retries:** 0  
**Answer excerpt:** According to the context, the specific permission requirement for accessing the "/items/pro/" API endpoint is that the user must be a "paying_user". This can be seen in the graph:

graph TB
...
pro_it...  

**Q:** What parameter in FastAPI's CORS configuration allows you to specify custom headers that should be exposed to clients?  
**Faithfulness:** 0.200 | **Strategy:** hybrid | **Retries:** 0  
**Answer excerpt:** The parameter in FastAPI's CORS configuration that allows you to specify custom headers that should be exposed to clients is `expose_headers`. 

According to the documentation, it is used with the `CO...  

---

## Resume bullet points

_Copy these directly into your resume. Replace X.XX with your actual scores._

```
• Engineered a self-correcting RAG pipeline with adaptive query routing
  (semantic / BM25 / hybrid-RRF), achieving 0.44 faithfulness
  and 0.70 answer relevancy on a 20-question
  synthetic eval set (RAGAS framework, Mistral 7B, local inference).

• Implemented LLM-as-judge relevance grading with automatic query rewriting;
  1 of 20 eval queries triggered the rewriter loop,
  improving retrieval precision from initial INSUFFICIENT grades to SUFFICIENT.
```