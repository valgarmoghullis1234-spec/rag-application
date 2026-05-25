"""
RAGAS Evaluation — Industry-standard RAG metrics

Requires: pip install -r evals/requirements_ragas.txt

Metrics:
  Faithfulness       — answer only uses retrieved context (detects hallucination)
  Response Relevancy — answer actually addresses the question
  Context Precision  — retrieved chunks are relevant (retrieval quality)
  LLM Context Recall — context contains enough info to answer (needs ground truth)

How it works:
  1. Load test_cases.csv (same file your custom eval uses)
  2. Call /query for each question — collect answer + chunk texts
  3. Run RAGAS metrics using OpenAI as the judge LLM
  4. Print scores + save per-question CSV
"""

import csv
import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

_here = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(_here, "..", ".env"), override=True)

API_URL        = os.getenv("BACKEND_URL", "http://localhost:8000")
TEST_CASES_CSV = os.path.join(_here, "test_cases.csv")
RESULTS_DIR    = os.path.join(_here, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Import RAGAS (friendly error if not installed) ───────────────────────────

def _import_ragas():
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextRecall,
            ContextPrecisionWithReference,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        return evaluate, EvaluationDataset, SingleTurnSample, Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecisionWithReference, LangchainLLMWrapper, ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Run:  pip install -r evals/requirements_ragas.txt\n")
        sys.exit(1)


# ── Data collection ──────────────────────────────────────────────────────────

def load_test_cases() -> list[dict]:
    with open(TEST_CASES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ask_rag(question: str) -> dict:
    resp = requests.post(
        f"{API_URL}/query",
        json={"question": question},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def collect_rag_outputs(test_cases: list[dict]) -> list[dict]:
    """Call the RAG API for every test case and collect raw outputs."""
    rows = []
    for i, tc in enumerate(test_cases, 1):
        question = tc["Question"]
        print(f"  [{i:02}/{len(test_cases)}] {question[:65]}...")

        try:
            result = ask_rag(question)
        except Exception as e:
            print(f"          ERROR: {e}")
            result = {"answer": "", "sources": [], "off_topic": False}

        if result.get("off_topic"):
            answer      = "I can only answer questions about the uploaded documents."
            chunk_texts = []
        else:
            answer      = result.get("answer", "")
            chunk_texts = [s["text"] for s in result.get("sources", []) if s.get("text")]

        rows.append({
            "question"    : question,
            "answer"      : answer,
            "contexts"    : chunk_texts if chunk_texts else ["No context retrieved."],
            "ground_truth": tc.get("Expected Answer", ""),
            "type"        : tc.get("Type", ""),
            "off_topic"   : result.get("off_topic", False),
        })
    return rows


# ── RAGAS evaluation ─────────────────────────────────────────────────────────

METRIC_DESCRIPTIONS = {
    "faithfulness"                  : "answer only claims things in retrieved context (↑ = less hallucination)",
    "answer_relevancy"              : "answer addresses the question (↑ = more on-point)",
    "context_precision_with_reference": "retrieved chunks are relevant to the question (↑ = better retrieval)",
    "context_recall"                : "context contained enough info to answer correctly (↑ = retrieval coverage)",
}


def run_ragas_metrics(rows: list[dict]) -> dict:
    (evaluate, EvaluationDataset, SingleTurnSample,
     Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecisionWithReference,
     LangchainLLMWrapper, ChatOpenAI, OpenAIEmbeddings) = _import_ragas()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY not set in .env — RAGAS uses OpenAI as its judge LLM.")
        sys.exit(1)

    llm        = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=openai_key))
    embeddings = OpenAIEmbeddings(api_key=openai_key)

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecisionWithReference(llm=llm),
        ContextRecall(llm=llm),
    ]

    samples = [
        SingleTurnSample(
            user_input        = row["question"],
            retrieved_contexts= row["contexts"],
            response          = row["answer"],
            reference         = row["ground_truth"],
        )
        for row in rows
    ]

    dataset = EvaluationDataset(samples=samples)
    result  = evaluate(dataset=dataset, metrics=metrics)
    return result


# ── Output ───────────────────────────────────────────────────────────────────

def print_summary(result) -> None:
    df = result.to_pandas()

    print("\n" + "=" * 70)
    print("  RAGAS RESULTS  (scale: 0.0 – 1.0, higher is better)")
    print("=" * 70)

    metric_keys = ["faithfulness", "answer_relevancy", "context_precision_with_reference", "context_recall"]
    for key in metric_keys:
        if key in df.columns:
            score = df[key].mean()
            desc  = METRIC_DESCRIPTIONS.get(key, "")
            bar   = "█" * int(score * 20)
            print(f"  {key:<25} {score:.3f}  {bar:<20}  {desc}")

    print("=" * 70)

    # Flag low-scoring questions per metric
    for key in metric_keys:
        if key not in df.columns:
            continue
        low = df[df[key] < 0.5]
        if not low.empty:
            print(f"\n  Low {key} questions (score < 0.5):")
            for _, row in low.iterrows():
                q = str(row.get("user_input", ""))[:60]
                print(f"    {row[key]:.2f}  {q}")

    print()


def save_results(result, rows: list[dict], timestamp: str) -> str:
    df = result.to_pandas()

    # Attach original question text if not already there
    if "user_input" not in df.columns:
        df.insert(0, "user_input", [r["question"] for r in rows])

    path = os.path.join(RESULTS_DIR, f"ragas_{timestamp}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check backend
    try:
        requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
    except Exception:
        print(f"\nCannot reach backend at {API_URL}")
        print("Start the backend first (start_backend.bat or uvicorn).\n")
        sys.exit(1)

    print("\nLoading test cases...")
    test_cases = load_test_cases()
    print(f"  {len(test_cases)} test cases found\n")

    print("Querying RAG backend...")
    rows = collect_rag_outputs(test_cases)

    print(f"\nRunning RAGAS metrics on {len(rows)} samples")
    print("(Uses OpenAI API as judge — takes ~2-3 minutes)\n")

    result = run_ragas_metrics(rows)

    print_summary(result)

    path = save_results(result, rows, timestamp)
    print(f"Per-question scores saved to:\n  {path}\n")


if __name__ == "__main__":
    main()
