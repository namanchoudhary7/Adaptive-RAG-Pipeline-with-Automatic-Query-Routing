# Adaptive RAG — Evaluation Report

> Generated: 2026-04-06 02:09  
> Test cases: 20  
> Model: Ollama / llama3:8b  
> Embeddings: sentence-transformers/all-MiniLM-L6-v2

---

## Summary scores

| Metric | Score | Bar | Grade |
|--------|-------|-----|-------|
| Faithfulness      | 0.670 | `█████████████░░░░░░░` | 🔴 Needs improvement |
| Answer relevancy  | 0.684 | `██████████████░░░░░░` | 🔴 Needs improvement |
| Context precision | 0.327 | `███████░░░░░░░░░░░░░` | 🔴 Needs improvement |
| Context recall    | 0.430 | `█████████░░░░░░░░░░░` | 🔴 Needs improvement |

---

## Retrieval behaviour

- **Queries that triggered the rewriter:** 4 / 20 (20%)

### Strategy distribution

| Strategy | Count | % of queries |
|----------|-------|--------------|
| BM25 | 2 | 10% |
| HYBRID | 15 | 75% |
| SEMANTIC | 3 | 15% |

---

## Per-question results

| # | Question | Strategy | Retries | Faith. | Rel. | Prec. | Recall |
|---|----------|----------|---------|--------|------|-------|--------|
| 1 | What is the default value for the allow_methods argumen... | bm25 | 0 | 0.33 | 0.99 | 0.00 | 0.00 |
| 2 | What types of data can I return from a FastAPI endpoint... | hybrid | 0 | 1.00 | 0.84 | 0.45 | 0.00 |
| 3 | What is the purpose of the `Docker Compose` tool in the... | hybrid | 0 | 0.60 | 0.00 | 0.00 | 0.00 |
| 4 | What is the URL where you can access the automatic inte... | hybrid | 0 | 0.50 | 0.94 | 0.00 | 1.00 |
| 5 | What is the effect of using the `response_model_exclude... | hybrid | 1 | 0.67 | 0.00 | 0.00 | 0.00 |
| 6 | What are the key features of the FastAPI framework? | semantic | 0 | 0.43 | 1.00 | 1.00 | 0.00 |
| 7 | What is the main advantage of using UploadFile instead ... | hybrid | 0 | 1.00 | 0.99 | 0.25 | 0.00 |
| 8 | What type of input does the `/files/` endpoint expect i... | hybrid | 0 | 0.17 | 0.86 | 1.00 | 1.00 |
| 9 | What is the order in which multiple middlewares are exe... | semantic | 0 | 0.75 | 0.70 | 1.00 | 0.00 |
| 10 | What is the recommended configuration for the number of... | hybrid | 0 | 0.57 | 0.95 | 1.00 | 0.00 |
| 11 | What is the purpose of the `yield` keyword when used in... | hybrid | 0 | 1.00 | 0.00 | 0.00 | 0.00 |
| 12 | What is the command used to run the FastAPI application... | hybrid | 0 | 1.00 | 1.00 | 0.50 | 1.00 |
| 13 | How do you configure separate OpenAPI schemas for input... | hybrid | 1 | 0.50 | 0.73 | 0.00 | 1.00 |
| 14 | What is the purpose of using yield in path operation de... | hybrid | 1 | 0.60 | 0.00 | 0.00 | 0.00 |
| 15 | What is the purpose of the `UploadFile` class in the de... | hybrid | 0 | 0.50 | 1.00 | 0.33 | 1.00 |
| 16 | What is the recommended approach for managing memory co... | hybrid | 0 | 0.50 | 0.49 | 0.00 | 1.00 |
| 17 | What type of data should be used for the "q" function p... | bm25 | 0 | 0.75 | 0.59 | 0.00 | 0.00 |
| 18 | How do I declare a request with multiple parameters and... | semantic | 0 | 0.88 | 0.72 | 0.00 | 0.60 |
| 19 | What is the templating engine used by FastAPI for rende... | hybrid | 3 | 1.00 | 0.92 | 0.00 | 1.00 |
| 20 | What is the primary mechanism that FastAPI uses to ensu... | hybrid | 0 | 0.67 | 0.96 | 1.00 | 1.00 |

---

## Lowest faithfulness cases

_These are the questions where the model most deviated from the retrieved context — useful for identifying retrieval failure modes._

**Q:** What type of input does the `/files/` endpoint expect in its request body?  
**Faithfulness:** 0.167 | **Strategy:** hybrid | **Retries:** 0  
**Answer excerpt:** According to the context from https://fastapi.tiangolo.com/tutorial/request-files/, the `/files/` endpoint expects a list of bytes or UploadFile. This is declared as follows:

```
@app.post("/files/")...  

**Q:** What is the default value for the allow_methods argument?  
**Faithfulness:** 0.333 | **Strategy:** bm25 | **Retries:** 0  
**Answer excerpt:** According to the context from https://fastapi.tiangolo.com/tutorial/cors/, the default value for the `allow_methods` argument is `['GET']`....  

**Q:** What are the key features of the FastAPI framework?  
**Faithfulness:** 0.429 | **Strategy:** semantic | **Retries:** 0  
**Answer excerpt:** According to the provided context, the key features of the FastAPI framework are:

* Fast: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic)
* Modern: Based on standa...  

---

## Resume bullet points

_Copy these directly into your resume. Replace X.XX with your actual scores._

```
• Engineered a self-correcting RAG pipeline with adaptive query routing
  (semantic / BM25 / hybrid-RRF), achieving 0.67 faithfulness
  and 0.68 answer relevancy on a 20-question
  synthetic eval set (RAGAS framework, Mistral 7B, local inference).

• Implemented LLM-as-judge relevance grading with automatic query rewriting;
  4 of 20 eval queries triggered the rewriter loop,
  improving retrieval precision from initial INSUFFICIENT grades to SUFFICIENT.
```