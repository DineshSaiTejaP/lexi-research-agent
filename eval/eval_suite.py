"""
eval/eval_suite.py — Automated evaluation framework for the Lexi Research Agent.

Measures 4 dimensions:
1. Precision      — Of retrieved precedents, what % are actually relevant?
2. Recall         — Of relevant precedents, what % did the agent find?
3. Reasoning      — Does the agent's explanation hold up? (LLM-as-judge)
4. Adverse ID     — Does the agent honestly surface unfavorable precedents?

Run with:
    python eval/eval_suite.py

Results are saved to eval/results_report.md
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent import run_agent

# ─── Load Gold Set ────────────────────────────────────────────────────────────
GOLD_SET_PATH = Path(__file__).parent / "gold_set.json"
REPORT_PATH = Path(__file__).parent / "results_report.md"

with open(GOLD_SET_PATH) as f:
    GOLD_SET = json.load(f)

EVAL_QUERIES = GOLD_SET["eval_queries"]


# ─── Dimension 1: Precision ───────────────────────────────────────────────────
def eval_precision(answer: str, steps: list[dict], query_type: str, expected_themes: list[str]) -> dict:
    """
    Measures: Of the document citations in the answer, what % align with expected themes?
    
    Method: 
    - Extract DOC_XXX references from the answer
    - Check if the surrounding context mentions any expected theme
    - Score = (contextually relevant citations) / (total citations)
    """
    # Extract all DOC citations from answer
    cited_docs = re.findall(r"DOC_\d+", answer)
    if not cited_docs:
        return {
            "score": 0.0,
            "cited_docs": [],
            "relevant_count": 0,
            "total_count": 0,
            "note": "No documents cited in answer"
        }

    # Check each citation's surrounding context for theme alignment
    relevant = 0
    for doc_id in set(cited_docs):
        # Find the context around this citation
        pattern = rf".{{0,200}}{re.escape(doc_id)}.{{0,200}}"
        matches = re.findall(pattern, answer, re.DOTALL)
        context = " ".join(matches).lower()

        # Check if any expected theme appears in context
        for theme in expected_themes:
            if theme.lower() in context:
                relevant += 1
                break

    total = len(set(cited_docs))
    score = relevant / total if total > 0 else 0.0

    return {
        "score": round(score, 3),
        "cited_docs": list(set(cited_docs)),
        "relevant_count": relevant,
        "total_count": total,
    }


# ─── Dimension 2: Recall ─────────────────────────────────────────────────────
def eval_recall(answer: str, steps: list[dict]) -> dict:
    """
    Measures: Coverage across the corpus.
    
    Method (without a fixed gold set):
    - Count unique DOC_XXX IDs retrieved during tool calls (not just in final answer)
    - Count unique DOC_XXX IDs in the final answer
    - Estimate: did the agent use multiple search strategies?
    
    Note: True recall requires a manually labeled gold set.
    After corpus review, update gold_set.json with supporting/adverse doc_ids
    and this function will compute exact recall.
    """
    # Docs retrieved during search (from tool results in steps)
    retrieved_in_steps = set()
    for step in steps:
        if step.get("type") == "tool_result":
            docs = re.findall(r"DOC_\d+", step.get("content", ""))
            retrieved_in_steps.update(docs)

    # Docs cited in final answer
    cited_in_answer = set(re.findall(r"DOC_\d+", answer))

    # Check if gold set has been populated
    gold_supporting = set(GOLD_SET.get("supporting_precedents", {}).get("doc_ids", []))
    gold_adverse = set(GOLD_SET.get("adverse_precedents", {}).get("doc_ids", []))
    gold_all = gold_supporting | gold_adverse

    tool_call_count = sum(1 for s in steps if s.get("type") == "tool_call")
    used_multiple_strategies = len(set(
        s.get("tool_name", "") for s in steps if s.get("type") == "tool_call"
    )) > 1

    result = {
        "docs_retrieved_in_search": len(retrieved_in_steps),
        "docs_cited_in_answer": len(cited_in_answer),
        "tool_calls_made": tool_call_count,
        "used_multiple_search_strategies": used_multiple_strategies,
    }

    if gold_all:
        found = cited_in_answer & gold_all
        recall_score = len(found) / len(gold_all) if gold_all else 0
        result["exact_recall_score"] = round(recall_score, 3)
        result["gold_docs_found"] = list(found)
        result["gold_docs_missed"] = list(gold_all - found)
    else:
        # Proxy: coverage score based on unique docs retrieved
        # 10+ docs retrieved = good coverage for a 50-doc corpus
        coverage_proxy = min(len(retrieved_in_steps) / 10, 1.0)
        result["coverage_proxy_score"] = round(coverage_proxy, 3)
        result["note"] = "Gold set not yet labeled. Update eval/gold_set.json for exact recall."

    return result


# ─── Dimension 3: Reasoning Quality (LLM-as-Judge) ───────────────────────────
def eval_reasoning_quality(query: str, answer: str) -> dict:
    """
    Measures: Does the agent's reasoning hold up?
    
    Method: Use the LLM itself as a judge (LLM-as-judge pattern).
    Ask it to score the answer on 3 sub-dimensions:
    - Faithfulness: Does the answer stay grounded in cited documents?
    - Legal Coherence: Does the legal reasoning make sense?
    - Citation Quality: Are citations specific and correctly used?
    
    Returns scores 1-5 for each dimension.
    """
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    judge_prompt = f"""You are evaluating an AI legal research agent's response quality.

QUERY: {query}

AGENT RESPONSE:
{answer[:3000]}

Score the response on these three dimensions (1-5 each):

1. FAITHFULNESS (1-5): Does the response only make claims that are grounded in the cited documents? 
   Does it avoid hallucinating cases or legal principles not found in the corpus?
   1=Lots of hallucination, 5=Fully grounded in cited sources

2. LEGAL_COHERENCE (1-5): Does the legal reasoning make logical sense?
   Are the legal principles correctly applied? Is the argument structure sound?
   1=Incoherent reasoning, 5=Clear and logically sound legal reasoning

3. CITATION_QUALITY (1-5): Are document citations specific and correctly used?
   Does the agent cite specific DOC_XXX identifiers? Are citations linked to their claims?
   1=No citations or vague, 5=Every claim backed by specific document citation

Respond ONLY with a JSON object like this:
{{"faithfulness": 4, "legal_coherence": 3, "citation_quality": 5, "overall_comment": "Brief explanation"}}"""

    try:
        if LLM_PROVIDER == "google":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(judge_prompt)
            raw = response.text.strip()
        elif LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
        else:
            return {"error": f"Unknown provider: {LLM_PROVIDER}"}

        # Extract JSON from response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            avg = (scores.get("faithfulness", 0) +
                   scores.get("legal_coherence", 0) +
                   scores.get("citation_quality", 0)) / 3
            scores["average_score"] = round(avg, 2)
            return scores
        else:
            return {"error": "Could not parse judge response", "raw": raw[:200]}

    except Exception as e:
        return {"error": str(e)}


# ─── Dimension 4: Adverse Identification ─────────────────────────────────────
def eval_adverse_identification(answer: str) -> dict:
    """
    Measures: Does the agent surface unfavorable precedents?
    
    Method:
    - Check if answer contains an "Adverse Precedents" section
    - Check if risk levels are mentioned (HIGH/MEDIUM/LOW)
    - Check if counter-strategies are provided
    - Penalize if only favorable cases are shown
    """
    answer_lower = answer.lower()

    has_adverse_section = bool(re.search(
        r"adverse\s+precedent|against.*client|unfavorable|opposing|risk",
        answer_lower
    ))

    has_risk_assessment = bool(re.search(
        r"risk level|high risk|medium risk|low risk|risk:.*high|risk:.*medium",
        answer_lower, re.IGNORECASE
    ))

    has_counter_strategy = bool(re.search(
        r"distinguish|counter|overcome|however|nevertheless|despite",
        answer_lower
    ))

    adverse_doc_count = len(re.findall(
        r"DOC_\d+",
        re.sub(r"(?i)supporting.*?(?=adverse|strategy|$)", "", answer, flags=re.DOTALL)
    ))

    # Score: 0-4 points
    score_components = {
        "has_adverse_section": has_adverse_section,
        "has_risk_assessment": has_risk_assessment,
        "has_counter_strategy": has_counter_strategy,
        "adverse_docs_cited": adverse_doc_count > 0,
    }
    score = sum(score_components.values()) / 4

    return {
        "score": round(score, 3),
        "components": score_components,
        "adverse_docs_cited_count": adverse_doc_count,
    }


# ─── Run Full Eval Suite ──────────────────────────────────────────────────────
def run_evaluation():
    print("\n" + "=" * 60)
    print("  LEXI AGENT EVALUATION SUITE")
    print("=" * 60)

    all_results = []

    for i, eval_query in enumerate(EVAL_QUERIES, 1):
        query = eval_query["query"]
        expected_themes = eval_query.get("expected_themes", [])
        query_type = eval_query.get("query_type", "general")

        print(f"\n[{i}/{len(EVAL_QUERIES)}] Evaluating: {query[:60]}...")

        # Run agent
        start = time.time()
        result = run_agent(query)
        elapsed = time.time() - start

        answer = result["answer"]
        steps = result["steps"]

        print(f"  [OK] Agent responded in {elapsed:.1f}s")

        # Evaluate all 4 dimensions
        precision = eval_precision(answer, steps, query_type, expected_themes)
        recall = eval_recall(answer, steps)
        reasoning = eval_reasoning_quality(query, answer)
        adverse = eval_adverse_identification(answer) if query_type == "deep_research" else None

        query_result = {
            "query": query,
            "query_type": query_type,
            "response_time_s": round(elapsed, 1),
            "precision": precision,
            "recall": recall,
            "reasoning": reasoning,
            "adverse_id": adverse,
        }
        all_results.append(query_result)

        # Print summary
        print(f"  Precision Score:  {precision['score']:.3f}")
        if "exact_recall_score" in recall:
            print(f"  Recall Score:     {recall['exact_recall_score']:.3f}")
        else:
            print(f"  Coverage Proxy:   {recall.get('coverage_proxy_score', 'N/A')}")
        if "average_score" in reasoning:
            print(f"  Reasoning Score:  {reasoning['average_score']:.2f}/5")
        if adverse:
            print(f"  Adverse ID Score: {adverse['score']:.3f}")

        # Rate limit pause
        time.sleep(2)

    # Compute aggregate scores
    precision_avg = sum(r["precision"]["score"] for r in all_results) / len(all_results)
    reasoning_avg = sum(
        r["reasoning"].get("average_score", 0) for r in all_results
    ) / len(all_results)
    adverse_results = [r for r in all_results if r["adverse_id"] is not None]
    adverse_avg = (
        sum(r["adverse_id"]["score"] for r in adverse_results) / len(adverse_results)
        if adverse_results else 0
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(all_results),
        "aggregate_scores": {
            "precision_avg": round(precision_avg, 3),
            "reasoning_avg": round(reasoning_avg, 3),
            "adverse_id_avg": round(adverse_avg, 3),
        },
        "query_results": all_results,
    }

    # Save raw results as JSON
    json_path = Path(__file__).parent / "eval_raw_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown report
    _write_report(summary)

    print(f"\n{'=' * 60}")
    print("  EVALUATION COMPLETE")
    print(f"  Precision Avg:  {precision_avg:.3f}")
    print(f"  Reasoning Avg:  {reasoning_avg:.2f}/5")
    print(f"  Adverse ID Avg: {adverse_avg:.3f}")
    print(f"\n  Report: eval/results_report.md")
    print("=" * 60)

    return summary


def _write_report(summary: dict):
    """Write evaluation results to markdown report."""
    agg = summary["aggregate_scores"]
    lines = [
        "# Lexi Agent Evaluation Report",
        f"\n**Run Date:** {summary['timestamp']}",
        f"**Total Queries Evaluated:** {summary['total_queries']}",
        "\n## Aggregate Scores\n",
        f"| Dimension | Score |",
        f"|-----------|-------|",
        f"| Precision | {agg['precision_avg']:.3f} |",
        f"| Reasoning Quality | {agg['reasoning_avg']:.2f}/5 |",
        f"| Adverse ID | {agg['adverse_id_avg']:.3f} |",
        "\n---\n",
        "## Per-Query Results\n",
    ]

    for i, r in enumerate(summary["query_results"], 1):
        lines.append(f"### Query {i}: {r['query'][:80]}")
        lines.append(f"- **Type:** {r['query_type']}")
        lines.append(f"- **Response Time:** {r['response_time_s']}s")
        lines.append(f"- **Precision Score:** {r['precision']['score']}")
        lines.append(f"- **Docs Cited:** {', '.join(r['precision'].get('cited_docs', [])) or 'None'}")

        recall = r["recall"]
        if "exact_recall_score" in recall:
            lines.append(f"- **Recall Score:** {recall['exact_recall_score']}")
        else:
            lines.append(f"- **Coverage Proxy:** {recall.get('coverage_proxy_score', 'N/A')}")
        lines.append(f"- **Docs Retrieved in Search:** {recall['docs_retrieved_in_search']}")

        reasoning = r["reasoning"]
        if "average_score" in reasoning:
            lines.append(f"- **Reasoning Score:** {reasoning['average_score']}/5")
            lines.append(f"  - Faithfulness: {reasoning.get('faithfulness')}/5")
            lines.append(f"  - Legal Coherence: {reasoning.get('legal_coherence')}/5")
            lines.append(f"  - Citation Quality: {reasoning.get('citation_quality')}/5")
            lines.append(f"  - Comment: {reasoning.get('overall_comment', '')}")

        if r.get("adverse_id"):
            adv = r["adverse_id"]
            lines.append(f"- **Adverse ID Score:** {adv['score']}")
            lines.append(f"  - Has adverse section: {adv['components']['has_adverse_section']}")
            lines.append(f"  - Has risk assessment: {adv['components']['has_risk_assessment']}")

        lines.append("")

    lines.extend([
        "---",
        "## Failure Analysis",
        "",
        "> **Instructions:** After running evals, fill in this section manually.",
        "> Identify the top 2-3 failure modes and what you would fix first.",
        "",
        "### Top Failure Modes",
        "1. [Fill in after running evals]",
        "2. [Fill in after running evals]",
        "3. [Fill in after running evals]",
        "",
        "### What I Would Fix First",
        "[Fill in after running evals — this section is discussed in the interview]",
    ])

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
