"""
Full evaluation pipeline.

Usage:
    uv run python -m scripts.run_eval                # generate 20 test cases + run eval
    uv run python -m scripts.run_eval --n 10         # faster, 10 test cases
    uv run python -m scripts.run_eval --skip-gen     # reuse existing testset.json

Flags:
    --n INT          Number of synthetic test cases to generate (default: 20)
    --skip-gen       Skip generation, load testset.json from disk
    --report-only    Just regenerate the report from an existing eval (no pipeline)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

from evaluation.testset import (
    TestSetGenerator,
    load_testset,
    save_testset,
    TESTSET_PATH,
)
from evaluation.evaluator import RAGASEvaluator
from evaluation.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--n", type=int, default=20, help="Number of test cases")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip testset generation, load from disk")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Adaptive RAG — RAGAS Evaluation")
    print("="*60)

    # Step 1: Testset
    if args.skip_gen and TESTSET_PATH.exists():
        print(f"\n[1/3] Loading existing testset from {TESTSET_PATH}...")
        test_cases = load_testset()
        print(f"      Loaded {len(test_cases)} test cases")
    else:
        print(f"\n[1/3] Generating {args.n} synthetic test cases...")
        print("      This takes ~2 min (one LLM call per test case)")
        generator = TestSetGenerator()
        test_cases = generator.generate(n=args.n)
        save_testset(test_cases)
        print(f"      Saved to {TESTSET_PATH}")

    # Step 2: Evaluation
    print(f"\n[2/3] Running RAGAS evaluation on {len(test_cases)} questions...")
    print("      Expected time: 5–15 min on CPU (LLM-as-judge is slow)")
    print("      Progress is logged above each question\n")

    evaluator = RAGASEvaluator()
    # In scripts/run_eval.py
    print(f"Number of test cases loaded: {len(test_cases)}")

    if len(test_cases) == 0:
        raise ValueError("No test cases found! Check your data loading logic.")

    summary = evaluator.evaluate(test_cases)

    # Step 3: Report
    print("\n[3/3] Generating report...")
    report = generate_report(summary)

    # Print summary to terminal
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"  Faithfulness      : {summary.mean_faithfulness:.3f}")
    print(f"  Answer relevancy  : {summary.mean_answer_relevancy:.3f}")
    print(f"  Context precision : {summary.mean_context_precision:.3f}")
    print(f"  Context recall    : {summary.mean_context_recall:.3f}")
    print(f"  Queries retried   : {summary.n_retried}/{summary.n_total}")
    print("="*60)
    print(f"\n  Full report → evaluation/EVAL_REPORT.md")
    print("  Commit this file to GitHub — it's your proof of quality.\n")


if __name__ == "__main__":
    main()