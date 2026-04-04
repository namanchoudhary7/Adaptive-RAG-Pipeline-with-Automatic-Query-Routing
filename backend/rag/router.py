"""
Query intent classifier + routing logic.

The router answers one question: given this query, which retrieval
strategy will most likely return useful context?

Classification approach:
We use a zero-shot HuggingFace classifier (facebook/bart-large-mnli)
with three candidate labels. Zero-shot means no fine-tuning needed —
the model uses natural language inference to score each label.

Why not just always use hybrid?
Hybrid + RRF is the safest default but has two real costs:
  1. Latency: runs two full retrieval passes instead of one
  2. BM25 degrades gracefully but adds noise on purely conceptual queries
The classifier adds ~50ms of overhead but saves ~200ms on average by
routing simple queries to the faster single-strategy path.
"""

import logging
from functools import lru_cache
from typing import Dict
from transformers import pipeline  
from backend.models import RetrievalStrategy

logger = logging.getLogger(__name__)

_STRATEGY_LABELS: Dict[str, RetrievalStrategy] = {
    "an explanation of a concept or system architecture": RetrievalStrategy.SEMANTIC,
    "documentation for a specific code class, function, or parameter": RetrievalStrategy.KEYWORD,
    "instructions for implementing a specific technical task": RetrievalStrategy.HYBRID,
}

@lru_cache
def _load_classifier():
    """
    Load the zero-shot classifier once and cache it.
    lru_cache on a module-level function acts as a singleton.
    First call takes ~3s to download/load; subsequent calls are instant.
    """

    logger.info("Loading zero-shot clasifier...")   
    clf = pipeline(
        task='zero-shot-classification',
        model='cross-encoder/nli-deberta-v3-small',
        device=-1              # CPU; change to 0 if you have a GPU
    )
    logger.info("Zero-shot classifier loaded")
    return clf

class QueryRouter:
    """
    Classifies an incoming query into one of three retrieval strategies.

    Design decision: we keep a confidence threshold. If the classifier
    isn't confident about the top prediction (score < threshold), we
    fall back to hybrid — the safest strategy. This prevents a borderline
    "conceptual" query from getting routed to pure semantic when hybrid
    would serve it better.
    """

    CONFIDENCE_THRESHOLD = 0.3
    def __init__(self)->None:
        # Eagerly load the classifier at construction time so the first
        # user query doesn't pay the startup penalty
        self._classifier = _load_classifier()
        self._label_to_strategy = _STRATEGY_LABELS

    def classify(self, query:str)-> tuple[RetrievalStrategy, float]:
        """
        Returns (chosen_strategy, confidence_score).

        The confidence score is useful for logging and debugging — if you
        see lots of low-confidence classifications, your label wording
        may need adjustment.
        """

        candidate_labels = list(self._label_to_strategy.keys())
        result = self._classifier(
            query,
            candidate_labels = candidate_labels,
            multi_label = False,   # Mutually exclusive strategies
            hypothesis_template="This question is asking for {}.",
        )

        top_label = result['labels'][0]
        top_score = result['scores'][0]

        if top_score < self.CONFIDENCE_THRESHOLD:
            logger.info(
                f"Low confidence ({top_score:.2f}) for query='{query[:60]}...' "
                f"— routing to hybrid as safe default"
            )
            return RetrievalStrategy.HYBRID, top_score
        
        chosen = self._label_to_strategy[top_label]

        logger.info(
            f"Router decision: strategy={chosen.value} | "
            f"confidence={top_score:.2f} | query='{query[:60]}'"
        )

        return chosen, top_score
    
    def route(self, query:str)->RetrievalStrategy:
        """Convenience method — returns just the strategy."""
        strategy, _ = self.classify(query=query)
        return strategy