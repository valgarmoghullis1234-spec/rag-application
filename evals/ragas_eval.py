"""
RAGAS Evaluation — Industry-standard RAG metrics

Requires: python -m pip install ragas --no-deps
          python -m pip install -r evals/requirements_ragas.txt

Metrics (all on a 0.0 – 1.0 scale):
  faithfulness       — answer only uses retrieved context (detects hallucination)
  answer_relevancy   — answer actually addresses the question
  context_precision  — retrieved chunks are relevant to the question (retrieval quality)
  context_recall     — context contains enough info to answer correctly (retrieval coverage)

How it works:
  1. Load test_cases.csv (same file your custom eval uses)
  2. Call /query for each question — collect answer + chunk texts
  3. Run RAGAS metrics using OpenAI gpt-4o-mini as the judge LLM
  4. Print scores + save per-question CSV to evals/results/
"""

import csv
import os
import sys
import warnings
from datetime import datetime

import requests
from dotenv import load_dotenv

# Suppress ragas deprecation warnings — the old-style metric API still works fine
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import llm_factory
        from langchain_openai import OpenAIEmbeddings
        from openai import OpenAI as OpenAIClient
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        return (
            evaluate, EvaluationDataset, SingleTurnSample,
            faithfulness, answer_relevancy, context_precision, context_recall,
            llm_factory, OpenAIEmbeddings, OpenAIClient,
        )
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Run:")
        print("  python -m pip install ragas --no-deps")
        print("  python -m pip install -r evals/requirements_ragas.txt\n")
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
    "faithfulness"     : "answer only claims things in retrieved context (↑ = less hallucination)",
    "answer_relevancy" : "answer addresses the question (↑ = more on-point)",
    "context_precision": "retrieved chunks are relevant to the question (↑ = better retrieval)",
    "context_recall"   : "context contained enough info to answer correctly (↑ = retrieval coverage)",
}


def run_ragas_metrics(rows: list[dict]):
    (evaluate, EvaluationDataset, SingleTurnSample,
     faithfulness, answer_relevancy, context_precision, context_recall,
     llm_factory, OpenAIEmbeddings, OpenAIClient) = _import_ragas()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY not set in .env — RAGAS uses OpenAI as its judge LLM.")
        sys.exit(1)

    # Wire up the LLM and embeddings to each metric
    llm = llm_factory("gpt-4o-mini", client=OpenAIClient(api_key=openai_key))
    lc_embeddings = OpenAIEmbeddings(api_key=openai_key)  # LangChain for embed_query interface

    faithfulness.llm            = llm
    context_precision.llm       = llm
    context_recall.llm          = llm
    answer_relevancy.llm        = llm
    answer_relevancy.embeddings = lc_embeddings

    # Build dataset
    samples = [
        SingleTurnSample(
            user_input         = row["question"],
            retrieved_contexts = row["contexts"],
            response           = row["answer"],
            reference          = row["ground_truth"],
        )
        for row in rows
    ]

    dataset = EvaluationDataset(samples=samples)
    return evaluate(
        dataset = dataset,
        metrics = [faithfulness, context_precision, context_recall, answer_relevancy],
    )


# ── Output ───────────────────────────────────────────────────────────────────

def print_summary(result) -> None:
    df = result.to_pandas()

    print("\n" + "=" * 72)
    print("  RAGAS RESULTS  (scale: 0.0 – 1.0, higher is better)")
    print("=" * 72)

    metric_keys = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    for key in metric_keys:
        if key in df.columns:
            score = df[key].mean()
            desc  = METRIC_DESCRIPTIONS.get(key, "")
            bar   = "█" * int(score * 20)
            print(f"  {key:<22} {score:.3f}  {bar:<20}  {desc}")

    print("=" * 72)

    # Flag low-scoring questions per metric (< 0.5)
    for key in metric_keys:
        if key not in df.columns:
            continue
        low = df[df[key] < 0.5]
        if not low.empty:
            print(f"\n  Low {key} (score < 0.5):")
            for _, row in low.iterrows():
                q = str(row.get("user_input", ""))[:65]
                print(f"    {row[key]:.2f}  {q}")

    print()


def save_results(result, rows: list[dict], timestamp: str) -> str:
    df = result.to_pandas()

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
    print("(Uses OpenAI gpt-4o-mini as judge — takes ~3-5 minutes)\n")

    result = run_ragas_metrics(rows)

    print_summary(result)

    path = save_results(result, rows, timestamp)
    print(f"Per-question scores saved to:\n  {path}\n")


if __name__ == "__main__":
    main()
