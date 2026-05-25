"""
Eval Runner — Automated RAG Evaluation
Reads test_cases.csv → asks each question to the RAG app →
scores answers using Claude → saves results.csv + prints summary
"""

import csv
import json
import os
import sys
from datetime import datetime
import requests
import anthropic
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

# Resolve .env regardless of where the script is run from
_here = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(_here, "..", ".env"), override=True)

# ── Config ───────────────────────────────────────────────────────────────────

API_URL        = "http://localhost:8000"
TEST_CASES_CSV = os.path.join(os.path.dirname(__file__), "test_cases.csv")
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Step 1: Load test cases ──────────────────────────────────────────────────

def load_test_cases(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


# ── Step 2: Ask the RAG app ──────────────────────────────────────────────────

def ask_rag(question: str) -> tuple[str, list[dict]]:
    """Send question to RAG API, return (answer, sources)."""
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            return data.get("answer", ""), data.get("sources", [])
        else:
            return f"API error: {resp.status_code} {resp.text}", []
    except requests.exceptions.ConnectionError:
        print("\n❌  Cannot reach the RAG backend at http://localhost:8000")
        print("    Make sure start_backend.bat is running before running evals.\n")
        sys.exit(1)
    except Exception as e:
        return f"Request failed: {e}", []


# ── Step 3: Score with Claude ────────────────────────────────────────────────

SCORER_SYSTEM = """You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.
Your job is to score the quality of an AI-generated answer against an expected answer.

Scoring rubric:
3 — Perfect: Correct, concise, directly answers the question, matches expected answer
2 — Good: Correct but too verbose, or missing minor detail, or slightly off phrasing
1 — Poor: Partially correct but missing key information or contains inaccuracies
0 — Fail: Wrong answer, hallucinated information, or said "not found" when answer exists in expected

For Negative type questions (expected answer says "Not available"):
3 — Correctly says the information is not available/not found
0 — Makes up an answer instead of saying not found

Always respond with valid JSON only. No explanation outside JSON."""

def score_answer(question: str, expected: str, app_answer: str, q_type: str, pass_criteria: str) -> dict:
    """Ask Claude to score the app's answer. Returns {score, reasoning}."""
    prompt = f"""Evaluate this RAG system answer:

Question: {question}
Question Type: {q_type}
Pass Criteria: {pass_criteria}
Expected Answer: {expected}
App's Answer: {app_answer}

Respond with JSON only:
{{
  "score": <0, 1, 2, or 3>,
  "reasoning": "<one sentence explaining the score>",
  "passed": <true or false>
}}"""

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system=SCORER_SYSTEM,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},  # force JSON start
        ],
    )

    try:
        raw = "{" + response.content[0].text
        # Extract just the JSON object if there's surrounding text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract score manually
        text = response.content[0].text
        score = 0
        for line in text.splitlines():
            if "score" in line.lower():
                for ch in line:
                    if ch.isdigit():
                        score = int(ch)
                        break
        return {"score": score, "reasoning": text[:120], "passed": score >= 2}


# ── Step 4: Save results ─────────────────────────────────────────────────────

def save_results(results: list[dict], timestamp: str):
    path = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")
    fieldnames = [
        "#", "Question", "Type", "Expected Answer",
        "App Answer", "Score (0-3)", "Passed", "Reasoning", "Sources"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return path


# ── Step 5: Print summary ────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    total       = len(results)
    passed      = sum(1 for r in results if r["Passed"] == "True")
    total_score = sum(int(r["Score (0-3)"]) for r in results)
    avg_score   = total_score / total if total else 0

    # By type
    types = {}
    for r in results:
        t = r["Type"]
        if t not in types:
            types[t] = {"total": 0, "passed": 0}
        types[t]["total"] += 1
        if r["Passed"] == "True":
            types[t]["passed"] += 1

    print("\n" + "=" * 60)
    print("  EVAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total Questions : {total}")
    print(f"  Passed          : {passed}/{total}  ({round(passed/total*100)}%)")
    print(f"  Average Score   : {avg_score:.2f} / 3.00")
    print("-" * 60)
    print("  Results by Type:")
    for q_type, stats in types.items():
        pct = round(stats["passed"] / stats["total"] * 100)
        print(f"    {q_type:<15} {stats['passed']}/{stats['total']} passed  ({pct}%)")
    print("-" * 60)

    # Show failed questions
    failed = [r for r in results if r["Passed"] == "False"]
    if failed:
        print(f"\n  ❌  Failed Questions ({len(failed)}):")
        for r in failed:
            print(f"    Q{r['#']:>2}: {r['Question'][:55]}")
            print(f"          Score: {r['Score (0-3)']} | {r['Reasoning']}")
    else:
        print("\n  ✅  All questions passed!")

    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_evals():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n📋  Loading test cases...")
    test_cases = load_test_cases(TEST_CASES_CSV)
    print(f"    Found {len(test_cases)} test cases\n")

    # Check backend is alive
    try:
        requests.get(f"{API_URL}/health", timeout=5)
    except:
        print("❌  Cannot reach the RAG backend at http://localhost:8000")
        print("    Make sure start_backend.bat is running first.\n")
        sys.exit(1)

    results = []

    for i, tc in enumerate(test_cases, 1):
        question      = tc["Question"]
        expected      = tc["Expected Answer"]
        q_type        = tc["Type"]
        pass_criteria = tc["Pass Criteria"]
        num           = tc["#"]

        print(f"[{i:02}/{len(test_cases)}] Q{num}: {question[:60]}...")

        # Ask RAG app
        app_answer, sources = ask_rag(question)
        source_str = " | ".join(
            f"{s['source']} (chunk {s['chunk_index']}, sim {s['similarity']})"
            for s in sources
        )

        # Score with Claude
        scored = score_answer(question, expected, app_answer, q_type, pass_criteria)
        score     = scored.get("score", 0)
        reasoning = scored.get("reasoning", "")
        passed    = scored.get("passed", False)

        icon = "✅" if passed else "❌"
        print(f"         {icon}  Score: {score}/3 — {reasoning}\n")

        results.append({
            "#"            : num,
            "Question"     : question,
            "Type"         : q_type,
            "Expected Answer": expected,
            "App Answer"   : app_answer,
            "Score (0-3)"  : score,
            "Passed"       : str(passed),
            "Reasoning"    : reasoning,
            "Sources"      : source_str,
        })

    # Save & summarise
    result_path = save_results(results, timestamp)
    print_summary(results)
    print(f"\n  📁  Full results saved to:\n      {result_path}\n")


if __name__ == "__main__":
    run_evals()
