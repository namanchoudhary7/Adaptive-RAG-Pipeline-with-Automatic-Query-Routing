"""
Synthetic test set generation.

Approach: feed random chunks to the LLM and ask it to generate a
realistic question that can be answered from that chunk. Then store
both the question and the ground-truth answer derived from the chunk.

This gives us an unbiased eval set — questions are grounded in the
actual corpus, not in what we assume the corpus contains.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from backend.config import settings
from backend.rag.ingestion import load_vector_store

logger = logging.getLogger(__name__)

TESTSET_PATH = Path("evaluation/testset.json")

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are building an evaluation dataset for a RAG system. "
     "Given a passage of technical documentation, generate ONE specific, "
     "self-contained question that:\n"
     "  - Can be answered using ONLY the information in the passage\n"
     "  - Is the kind of question a developer would actually ask\n"
     "  - Is not trivially answered by a single word\n\n"
     "Output ONLY the question — no preamble, no numbering, no explanation."),
    ("human", "Passage:\n{passage}\n\nQuestion:"),
])

ANSWER_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical documentation assistant. Answer the question "
     "using ONLY the information in the provided passage. Be concise and precise. "
     "Output ONLY the answer — no preamble."),
    ("human", "Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer:"),
])

@dataclass
class TestCase:
    question: str
    ground_truth: str       # LLM-generated reference answer from the source chunk
    source_chunk: str       # The chunk used to generate this Q&A pair
    source_url: str

class TestSetGenerator:

    def __init__(self) -> None:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.4,    # Slightly higher temp for question diversity
        )
        self._question_chain = QUESTION_GEN_PROMPT | llm | StrOutputParser()
        self._answer_chain   = ANSWER_GEN_PROMPT   | llm | StrOutputParser()

    def generate(self, n: int = 20, min_chunk_length: int = 200) -> List[TestCase]:
        """
        Sample n chunks from ChromaDB and generate one Q&A pair per chunk.

        Args:
            n: Number of test cases to generate.
            min_chunk_length: Skip chunks shorter than this — they're usually
                              nav elements or headings with no answerable content.
        """
        
        logger.info(f"Generating {n} synthetic test cases...")

        store = load_vector_store()
        collection = store._collection
        result = collection.get(include=["documents", "metadatas"])

        all_chunks = [
            (text, meta)
            for text, meta in zip(result["documents"], result["metadatas"])
            if len(text.strip()) >= min_chunk_length
        ]

        if len(all_chunks) < n:
            logger.warning(
                f"Only {len(all_chunks)} usable chunks — "
                f"generating {len(all_chunks)} test cases instead of {n}"
            )
            n = len(all_chunks)

        sampled = random.sample(all_chunks, n)
        test_cases: List[TestCase] = []

        for i, (chunk_text, meta) in enumerate(sampled, 1):
            logger.info(f"Generating test case {i}/{n}...")
            try:
                question = self._question_chain.invoke(
                    {"passage": chunk_text}
                ).strip()

                ground_truth = self._answer_chain.invoke(
                    {"passage": chunk_text, "question": question}
                ).strip()

                test_cases.append(TestCase(
                    question=question,
                    ground_truth=ground_truth,
                    source_chunk=chunk_text,
                    source_url=meta.get("source", "unknown"),
                ))

            except Exception as e:
                logger.warning(f"Failed to generate test case {i}: {e}")
                continue

        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases
    
def save_testset(test_cases: List[TestCase], path: Path = TESTSET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(tc) for tc in test_cases], f, indent=2)
    logger.info(f"Saved {len(test_cases)} test cases → {path}")


def load_testset(path: Path = TESTSET_PATH) -> List[TestCase]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TestCase(**item) for item in data]