"""
Agent Evaluation — A/B comparison: Single-pass RAG vs Agentic RAG

Runs a hard question set through BOTH pipelines, scores each answer with
Claude-as-judge, and prints a side-by-side comparison table.

These questions are designed to stress the single-pass pipeline — comparisons,
multi-hop reasoning, aggregation, and terminology mismatches.

Usage:
    # From the project root:
    cd backend && python ../evals/agent_eval.py

Requirements:
    Backend does NOT need to be running — this calls the Python functions directly.
    ANTHROPIC_API_KEY must be set in .env
"""

import json
import os
import sys
import time
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
_here    = os.path.abspath(os.path.dirname(__file__))
_backend = os.path.join(_here, "..", "backend")
sys.path.insert(0, _backend)

from dotenv import load_dotenv
load_dotenv(os.path.join(_here, "..", ".env"), override=True)

import anthropic
from query import answer_question
from agent import answer_agent

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

RESULTS_DIR = os.path.join(_here, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Hard question test set ────────────────────────────────────────────────────
#
# These question types are deliberately chosen because they expose the
# weaknesses of single-pass (one query → top-K chunks → answer):
#
#   multi_doc_comparison  — needs balanced coverage from multiple documents
#   multi_hop             — needs info from two different parts of a doc
#   aggregation           — needs a sweep across all documents, not just top chunks
#   cross_doc_extraction  — needs the same attribute extracted per-document
#   terminology_mismatch  — user's words may differ from doc's exact phrasing
#
# NOTE: these questions are intentionally generic so they work with ANY
# documents you have uploaded. Tailor them to your actual docs for sharper results.

HARD_QUESTIONS = [
    {
        "id"         : "comp_1",
        "question"   : "Compare the key skills, experience, and qualifications across all uploaded documents. What are the main differences and similarities?",
        "type"       : "multi_doc_comparison",
        "why_hard"   : "Single-pass retrieval tends to be dominated by whichever doc scores highest; the agent searches each doc separately.",
    },
    {
        "id"         : "multihop_1",
        "question"   : "What are the main professional achievements mentioned, and how do they connect to the goals or objectives stated elsewhere in the documents?",
        "type"       : "multi_hop",
        "why_hard"   : "Achievements and goals likely live in different sections; one embedding query rarely surfaces both.",
    },
    {
        "id"         : "aggregation_1",
        "question"   : "Provide a structured summary of the most important information from each uploaded document separately.",
        "type"       : "aggregation",
        "why_hard"   : "A single-pass query retrieves highest-scoring chunks globally, often missing important content from lower-scoring documents.",
    },
    {
        "id"         : "extraction_1",
        "question"   : "What technical skills or tools are mentioned in the documents, and which document mentions each one?",
        "type"       : "cross_doc_extraction",
        "why_hard"   : "Requires deliberate per-document searching; single-pass mixes everything and loses the per-doc attribution.",
    },
    {
        "id"         : "timeline_1",
        "question"   : "Reconstruct the chronological timeline of roles, positions, or events described across all documents.",
        "type"       : "multi_hop",
        "why_hard"   : "Timeline reconstruction requires gathering date-tagged facts from multiple sections/docs.",
    },
]


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating answers produced by a document Q&A system.
Score the answer on two dimensions (each 1–5):

COMPLETENESS (1–5): Does the answer address ALL parts of the question?
  1 = Misses most of the question
  2 = Partial — significant gaps
  3 = Covers the main point but misses details
  4 = Mostly complete, minor gaps
  5 = Fully addresses every aspect of the question

FAITHFULNESS (1–5): Is the answer grounded in actual document content?
  1 = Mostly speculation or general knowledge
  2 = Mix of document content and guessing
  3 = Mostly from documents but some guessing
  4 = Clearly from documents with good citations
  5 = Fully grounded, specific citations throughout

Question: {question}

Answer: {answer}

Reply with JSON only — no extra text:
{{"completeness": <1-5>, "faithfulness": <1-5>, "reasoning": "<one concise sentence>"}}"""


def judge_answer(question: str, answer: str) -> dict:
    """Score an answer using Claude Haiku as judge (cheap + fast)."""
    try:
        response = anthropic_client.messages.create(
            model      = "claude-haiku-4-5",
            max_tokens = 200,
            messages   = [{"role": "user",
                           "content": JUDGE_PROMPT.format(question=question, answer=answer)}],
        )
        return json.loads(response.content[0].text)
    except Exception as e:
        return {"completeness": 0, "faithfulness": 0, "reasoning": f"judge error: {e}"}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_eval():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print()
    print("=" * 72)
    print("  AGENTIC RAG vs SINGLE-PASS RAG — EVALUATION")
    print(f"  {len(HARD_QUESTIONS)} hard questions  |  judge: claude-haiku-4-5")
    print("=" * 72)

    all_results = []

    for q in HARD_QUESTIONS:
        print(f"\n[{q['type'].upper()}]")
        print(f"  Q: {q['question'][:75]}...")
        print(f"  Why hard: {q['why_hard']}")

        # ── Single-pass baseline ──────────────────────────────────────────────
        print("  Running baseline (single-pass)...", end=" ", flush=True)
        t0       = time.time()
        baseline = answer_question(q["question"])
        b_time   = round(time.time() - t0, 1)
        b_score  = judge_answer(q["question"], baseline["answer"])
        print(f"done ({b_time}s)")

        # ── Agentic pipeline ──────────────────────────────────────────────────
        print("  Running agent...", end=" ", flush=True)
        t0           = time.time()
        agent_result = answer_agent(q["question"])
        a_time       = round(time.time() - t0, 1)
        a_score      = judge_answer(q["question"], agent_result["answer"])
        print(f"done ({a_time}s, {agent_result['iterations']} iterations, "
              f"{len(agent_result['tool_calls'])} tool calls)")

        # ── Print comparison ──────────────────────────────────────────────────
        b_total = b_score["completeness"] + b_score["faithfulness"]
        a_total = a_score["completeness"] + a_score["faithfulness"]
        delta   = a_total - b_total
        winner  = "AGENT ▲" if delta > 0 else ("TIE  =" if delta == 0 else "BASELINE ▼")

        print(f"\n  {'':30}  Complete  Faithful  Total")
        print(f"  {'Baseline (single-pass)':<30}  {b_score['completeness']}/5       "
              f"{b_score['faithfulness']}/5       {b_total}/10")
        print(f"  {'Agent (agentic RAG)':<30}  {a_score['completeness']}/5       "
              f"{a_score['faithfulness']}/5       {a_total}/10")
        print(f"  Result: {winner}  |  Judge: \"{a_score['reasoning']}\"")

        # ── Tool call trace ───────────────────────────────────────────────────
        if agent_result["tool_calls"]:
            print("  Tool trace:")
            for i, tc in enumerate(agent_result["tool_calls"], 1):
                inp_str = json.dumps(tc["input"])[:60]
                print(f"    {i}. {tc['tool']}({inp_str}) → {tc['summary']}")

        all_results.append({
            "question_id"  : q["id"],
            "type"         : q["type"],
            "question"     : q["question"],
            "baseline": {
                "answer_preview" : baseline["answer"][:200],
                "scores"         : b_score,
                "time_s"         : b_time,
            },
            "agent": {
                "answer_preview" : agent_result["answer"][:200],
                "scores"         : a_score,
                "time_s"         : a_time,
                "iterations"     : agent_result["iterations"],
                "tool_calls"     : agent_result["tool_calls"],
                "sources"        : agent_result["sources"],
            },
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    def avg(key, pipeline):
        return sum(r[pipeline]["scores"][key] for r in all_results) / len(all_results)

    b_comp   = avg("completeness", "baseline")
    b_faith  = avg("faithfulness", "baseline")
    a_comp   = avg("completeness", "agent")
    a_faith  = avg("faithfulness", "agent")
    avg_iter = sum(r["agent"]["iterations"] for r in all_results) / len(all_results)
    avg_tc   = sum(len(r["agent"]["tool_calls"]) for r in all_results) / len(all_results)

    def bar(score, max_score=5):
        filled = round(score / max_score * 15)
        return "█" * filled + "░" * (15 - filled)

    print(f"\n  Metric             Baseline          Agent")
    print(f"  {'Completeness':<18} {b_comp:.2f}/5  {bar(b_comp)}  "
          f"{a_comp:.2f}/5  {bar(a_comp)}")
    print(f"  {'Faithfulness':<18} {b_faith:.2f}/5  {bar(b_faith)}  "
          f"{a_faith:.2f}/5  {bar(a_faith)}")

    comp_delta  = ((a_comp  - b_comp)  / b_comp  * 100) if b_comp  else 0
    faith_delta = ((a_faith - b_faith) / b_faith * 100) if b_faith else 0

    print(f"\n  Completeness improvement : {comp_delta:+.1f}%")
    print(f"  Faithfulness improvement : {faith_delta:+.1f}%")
    print(f"  Avg agent iterations     : {avg_iter:.1f}")
    print(f"  Avg tool calls per query : {avg_tc:.1f}")

    # Per-type breakdown
    types = sorted({r["type"] for r in all_results})
    if len(types) > 1:
        print("\n  Per-type agent completeness:")
        for t in types:
            t_results = [r for r in all_results if r["type"] == t]
            t_avg = sum(r["agent"]["scores"]["completeness"] for r in t_results) / len(t_results)
            b_avg = sum(r["baseline"]["scores"]["completeness"] for r in t_results) / len(t_results)
            print(f"    {t:<28}  baseline {b_avg:.1f}  →  agent {t_avg:.1f}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, f"agent_eval_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Full results saved to:\n    {out_path}")
    print()


if __name__ == "__main__":
    run_eval()
