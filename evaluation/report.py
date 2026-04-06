"""
Generates a markdown evaluation report from EvalSummary.
"""
from datetime import datetime
from pathlib import Path
import math
from evaluation.evaluator import EvalSummary
from backend.config import settings

REPORT_PATH = Path("evaluation/EVAL_REPORT_3.md")


def _score_bar(score: float, width: int = 20) -> str:
    # Check if the score is NaN or None
    if score is None or math.isnan(score):
        return "░" * width
        
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)

def _grade(score: float) -> str:
    # Check if the score is NaN or None
    if score is None or math.isnan(score):
        return "⚪ N/A"
        
    if score >= 0.85:
        return "🟢 Strong"
    if score >= 0.70:
        return "🟡 Acceptable"
    return "🔴 Needs improvement"

def generate_report(summary: EvalSummary, path: Path = REPORT_PATH) -> str:
    lines = [
        "# Adaptive RAG — Evaluation Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> Test cases: {summary.n_total}  ",
        f"> Model: Ollama / {settings.ollama_model}  ",
        f"> Embeddings: {settings.embedding_model}",
        "",
        "---",
        "",
        "## Summary scores",
        "",
        "| Metric | Score | Bar | Grade |",
        "|--------|-------|-----|-------|",
        f"| Faithfulness      | {summary.mean_faithfulness:.3f} | `{_score_bar(summary.mean_faithfulness)}` | {_grade(summary.mean_faithfulness)} |",
        f"| Answer relevancy  | {summary.mean_answer_relevancy:.3f} | `{_score_bar(summary.mean_answer_relevancy)}` | {_grade(summary.mean_answer_relevancy)} |",
        f"| Context precision | {summary.mean_context_precision:.3f} | `{_score_bar(summary.mean_context_precision)}` | {_grade(summary.mean_context_precision)} |",
        f"| Context recall    | {summary.mean_context_recall:.3f} | `{_score_bar(summary.mean_context_recall)}` | {_grade(summary.mean_context_recall)} |",
        "",
        "---",
        "",
        "## Retrieval behaviour",
        "",
        f"- **Queries that triggered the rewriter:** {summary.n_retried} / {summary.n_total} "
        f"({round(summary.n_retried / summary.n_total * 100) if summary.n_total else 0}%)",
        "",
        "### Strategy distribution",
        "",
    ]

    # Strategy breakdown
    strategies = {}
    for r in summary.results:
        strategies[r.retrieval_strategy] = strategies.get(r.retrieval_strategy, 0) + 1

    lines.append("| Strategy | Count | % of queries |")
    lines.append("|----------|-------|--------------|")
    for strategy, count in sorted(strategies.items()):
        pct = round(count / summary.n_total * 100) if summary.n_total else 0
        lines.append(f"| {strategy.upper()} | {count} | {pct}% |")

    # Per-question results table
    lines += [
        "",
        "---",
        "",
        "## Per-question results",
        "",
        "| # | Question | Strategy | Retries | Faith. | Rel. | Prec. | Recall |",
        "|---|----------|----------|---------|--------|------|-------|--------|",
    ]

    for i, r in enumerate(summary.results, 1):
        q = r.question[:55] + "..." if len(r.question) > 55 else r.question
        lines.append(
            f"| {i} | {q} | {r.retrieval_strategy} | {r.retry_count} "
            f"| {r.faithfulness:.2f} | {r.answer_relevancy:.2f} "
            f"| {r.context_precision:.2f} | {r.context_recall:.2f} |"
        )

    # Worst performers — most useful for debugging
    lines += [
        "",
        "---",
        "",
        "## Lowest faithfulness cases",
        "",
        "_These are the questions where the model most deviated from the "
        "retrieved context — useful for identifying retrieval failure modes._",
        "",
    ]

    worst = sorted(summary.results, key=lambda r: r.faithfulness)[:3]
    for r in worst:
        lines += [
            f"**Q:** {r.question}  ",
            f"**Faithfulness:** {r.faithfulness:.3f} | "
            f"**Strategy:** {r.retrieval_strategy} | "
            f"**Retries:** {r.retry_count}  ",
            f"**Answer excerpt:** {r.answer[:200]}...  ",
            "",
        ]

    # Resume bullet points — the payoff
    lines += [
        "---",
        "",
        "## Resume bullet points",
        "",
        "_Copy these directly into your resume. Replace X.XX with your actual scores._",
        "",
        "```",
        f"• Engineered a self-correcting RAG pipeline with adaptive query routing",
        f"  (semantic / BM25 / hybrid-RRF), achieving {summary.mean_faithfulness:.2f} faithfulness",
        f"  and {summary.mean_answer_relevancy:.2f} answer relevancy on a {summary.n_total}-question",
        f"  synthetic eval set (RAGAS framework, Mistral 7B, local inference).",
        "",
        f"• Implemented LLM-as-judge relevance grading with automatic query rewriting;",
        f"  {summary.n_retried} of {summary.n_total} eval queries triggered the rewriter loop,",
        f"  improving retrieval precision from initial INSUFFICIENT grades to SUFFICIENT.",
        "```",
    ]

    report = "\n".join(lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"\n✅ Report saved → {path}")

    return report