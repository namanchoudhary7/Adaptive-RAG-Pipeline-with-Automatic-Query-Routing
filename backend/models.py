from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class GradeResult(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The user's question about the technical documentation",
        examples=["How do I add middleware in FastAPI?"],
    )


class SourceDocument(BaseModel):
    content: str = Field(description="The retrieved chunk text")
    source: str = Field(description="URL or file path of the source document")
    relevance_score: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument]
    retrieval_strategy: RetrievalStrategy
    relevance_grade: GradeResult
    retry_count: int = Field(ge=0, description="Number of self-correction loops taken")
    confidence: ConfidenceLevel


class HealthResponse(BaseModel):
    status: str
    ollama_model: str
    embedding_model: str
    docs_indexed: int