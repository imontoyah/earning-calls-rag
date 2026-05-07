#!/usr/bin/env python
"""Run the eval suite against the current ChromaDB collection.

Two layers:
  1. Retrieval — does the retriever surface the right quarter/speaker/keywords?
  2. Answer    — LLM-as-judge scores faithfulness + relevance on a 1-5 scale.

Usage:
    uv run python scripts/evaluate.py                  # full eval (retrieval + LLM judge)
    uv run python scripts/evaluate.py --retrieval-only # skip the LLM judge (fast, no API cost)
    uv run python scripts/evaluate.py --output PATH    # also dump full results as JSON
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.embeddings import get_client, get_collection
from src.rag import ask, make_llm
from src.retrieval import semantic_search

CASES_PATH = Path(__file__).resolve().parent.parent / "evals" / "cases.json"


JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """Evaluate this RAG system answer on two dimensions:

1. Faithfulness (1-5): Does the answer only use information from the context?
2. Relevance (1-5): Does the answer address the question?

Context:
{context}

Question: {question}
Answer: {answer}
Expected keywords: {expected_keywords}

Respond exactly as:
Faithfulness: [1-5]
Relevance: [1-5]
Brief explanation: [one sentence]
"""
)


def _evaluate_retrieval(case: dict, collection) -> dict:
    """Score one case on retrieval metrics."""
    chunks = semantic_search(collection, case["question"])
    quarters = [c["quarter"] for c in chunks]
    speakers = [c["speaker"] for c in chunks]
    context = "\n\n".join(c["text"] for c in chunks).lower()

    expected_q = case["expected_quarter"]
    expected_s = case["expected_speaker"]
    keywords = case["expected_keywords"]

    quarter_hit = any(q in quarters for q in expected_q) if expected_q else True
    speaker_hit = (expected_s in speakers) if expected_s else True
    keyword_misses = [kw for kw in keywords if kw.lower() not in context]
    keyword_score = (
        (len(keywords) - len(keyword_misses)) / len(keywords) if keywords else 1.0
    )

    return {
        "chunks": chunks,
        "quarter_hit": quarter_hit,
        "speaker_hit": speaker_hit,
        "keyword_score": keyword_score,
        "keyword_misses": keyword_misses,
    }


def _judge_answer(case: dict, answer: str, context: str, llm) -> tuple[int, int, str]:
    """Ask the LLM to score one answer on faithfulness and relevance."""
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    judgment = chain.invoke({
        "context": context,
        "question": case["question"],
        "answer": answer,
        "expected_keywords": ", ".join(case["expected_keywords"]),
    })
    f_match = re.search(r"Faithfulness:\s*(\d)", judgment)
    r_match = re.search(r"Relevance:\s*(\d)", judgment)
    return (
        int(f_match.group(1)) if f_match else 0,
        int(r_match.group(1)) if r_match else 0,
        judgment,
    )


def _print_retrieval_summary(results: list[dict]) -> None:
    print("\nRETRIEVAL")
    print("=" * 60)
    for r in results:
        passed = r["quarter_hit"] and r["speaker_hit"] and r["keyword_score"] >= 0.5
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] [{r['category']:>16s}] {r['question'][:50]}")
        if r["keyword_misses"]:
            print(f"            missing keywords: {r['keyword_misses']}")

    n = len(results)
    print(f"\nQuarter accuracy : {sum(r['quarter_hit'] for r in results)}/{n}")
    print(f"Speaker accuracy : {sum(r['speaker_hit'] for r in results)}/{n}")
    print(f"Keyword coverage : {sum(r['keyword_score'] for r in results) / n:.0%}")


def _print_answer_summary(results: list[dict]) -> None:
    if not any("faithfulness" in r for r in results):
        return
    print("\nANSWER QUALITY (LLM judge)")
    print("=" * 60)
    for r in results:
        print(
            f"[{r['category']:>16s}] F:{r['faithfulness']}/5 R:{r['relevance']}/5 — {r['question'][:45]}"
        )

    n = len(results)
    avg_f = sum(r["faithfulness"] for r in results) / n
    avg_r = sum(r["relevance"] for r in results) / n
    print(f"\nAvg faithfulness : {avg_f:.1f} / 5")
    print(f"Avg relevance    : {avg_r:.1f} / 5")

    print("\nPer category:")
    for cat in sorted({r["category"] for r in results}):
        cat_results = [r for r in results if r["category"] == cat]
        cf = sum(r["faithfulness"] for r in cat_results) / len(cat_results)
        cr = sum(r["relevance"] for r in cat_results) / len(cat_results)
        print(f"  {cat:>16s}: F={cf:.1f} R={cr:.1f}  (n={len(cat_results)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval suite.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip the LLM-judge stage (no Groq calls, much faster).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to dump full per-case results as JSON.",
    )
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text())
    collection = get_collection(get_client())
    print(f"Loaded {len(cases)} cases | collection: {collection.count()} documents")

    results: list[dict[str, Any]] = []

    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['question'][:55]}...", end=" ", flush=True)
        retr = _evaluate_retrieval(case, collection)

        record: dict[str, Any] = {
            "question": case["question"],
            "category": case["category"],
            "quarter_hit": retr["quarter_hit"],
            "speaker_hit": retr["speaker_hit"],
            "keyword_score": retr["keyword_score"],
            "keyword_misses": retr["keyword_misses"],
        }

        if not args.retrieval_only:
            llm = make_llm()
            answer = ask(collection, case["question"], llm=llm)
            context = "\n\n".join(c["text"] for c in retr["chunks"])
            faithfulness, relevance, judgment = _judge_answer(case, answer, context, llm)
            record.update(
                answer=answer,
                faithfulness=faithfulness,
                relevance=relevance,
                judgment=judgment,
            )
            print(f"F:{faithfulness}/5 R:{relevance}/5")
        else:
            print("ok")

        results.append(record)

    _print_retrieval_summary(results)
    _print_answer_summary(results)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nWrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
