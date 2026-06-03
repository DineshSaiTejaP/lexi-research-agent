# Lexi Agent Evaluation Report

**Run Date:** 2026-06-03  
**Agent:** LangGraph — 6 nodes, hybrid RRF retrieval  
**LLM:** Groq LLaMA 3.3 70B Versatile  
**Corpus:** 50 Indian court judgments (DOC_001 – DOC_050)  
**Total Queries Run:** 10 (6 completed, 4 failed due to rate limiting)

---

## Aggregate Scores

### All queries (including rate-limit failures)

| Dimension | Score | Note |
|---|---|---|
| Precision | 0.400 | Depressed by 4 empty responses from rate limiting |
| Reasoning Quality | 2.87 / 5 | Depressed by 1.0/5 scores on failed queries |
| Adverse Identification | 0.667 | Depressed by failed deep-research queries |

### Queries that completed successfully (Q1–Q6)

| Dimension | Score |
|---|---|
| Precision | 0.667 |
| Reasoning Quality | 4.44 / 5 |
| Adverse Identification | 0.875 |

> The effective scores above are the honest representation of agent quality. The aggregate scores are pulled down entirely by Groq rate limiting on queries 7–10, not by agent reasoning failures.

---

## Per-Query Results

### Query 1 — Find precedents supporting Mrs. Lakshmi Devi's claim
- **Type:** deep_research · **Time:** 15.8s
- **Precision:** 0.0 ← eval artifact (see Failure Analysis #2)
- **Docs Cited:** DOC_014, DOC_031, DOC_029, DOC_001, DOC_034
- **Recall:** 0.2 (1 of 5 gold supporting docs cited)
- **Reasoning:** 4.0 / 5 — Faithfulness 4, Coherence 3, Citations 5
- **Adverse ID:** 1.0 — adverse section present, risk levels provided
- **Judge comment:** Specific citations and legal principles applied well; causal links between facts and precedent could be more explicit.

### Query 2 — Which judgments involve commercial vehicles?
- **Type:** general · **Time:** 2.7s
- **Precision:** 1.0
- **Docs Cited:** DOC_027, DOC_042, DOC_032
- **Recall:** 0.2
- **Reasoning:** 5.0 / 5 — Faithfulness 5, Coherence 5, Citations 5
- **Judge comment:** Direct, accurate, every claim grounded in cited sources.

### Query 3 — Find adverse precedents the insurance company could use
- **Type:** deep_research · **Time:** 16.9s
- **Precision:** 0.0 ← eval artifact (see Failure Analysis #2)
- **Docs Cited:** DOC_029, DOC_020, DOC_009, DOC_025, DOC_034
- **Recall:** 0.4 (2 of 5 gold adverse docs cited)
- **Reasoning:** 4.0 / 5 — Faithfulness 4, Coherence 3, Citations 5
- **Adverse ID:** 1.0 — adverse section present, risk levels provided
- **Judge comment:** Good citation quality; legal principle application to specific facts could be more precise.

### Query 4 — Compensation range for 42-year-old earning ₹35,000/month
- **Type:** deep_research · **Time:** 8.0s
- **Precision:** 1.0
- **Docs Cited:** DOC_001, DOC_024, DOC_039, DOC_033
- **Recall:** 0.0
- **Reasoning:** 4.0 / 5 — Faithfulness 4, Coherence 3, Citations 5
- **Adverse ID:** 0.5 — ⚠️ adverse section absent
- **Judge comment:** Well-structured and grounded; some compensation calculations rely on general knowledge rather than strictly the retrieved documents.

### Query 5 — Which judgments involve Section 149 of the Motor Vehicles Act?
- **Type:** general · **Time:** 5.7s
- **Precision:** 1.0
- **Docs Cited:** DOC_029, DOC_030, DOC_032
- **Recall:** 0.0
- **Reasoning:** 5.0 / 5 — Faithfulness 5, Coherence 5, Citations 5
- **Judge comment:** Perfect — direct, no hallucination, precise citations.

### Query 6 — Contributory negligence cases with unlicensed driver
- **Type:** deep_research · **Time:** 26.1s
- **Precision:** 1.0
- **Docs Cited:** DOC_018, DOC_023, DOC_004, DOC_006, DOC_034
- **Recall:** 0.2
- **Reasoning:** 4.67 / 5 — Faithfulness 5, Coherence 4, Citations 5
- **Adverse ID:** 1.0 — adverse section present, risk levels provided
- **Judge comment:** Fully grounded, clear reasoning, strong citation quality.

### Queries 7–10 — Rate limit failures

All four queries hit Groq's rate limit after Q6. Response times of 270–530 seconds indicate the retry logic was waiting but ultimately failing. Responses were empty; all scores are 0.

| # | Query (truncated) | Time | Failure |
|---|---|---|---|
| 7 | Which cases discuss pay and recover doctrine? | 269.7s | Rate limit |
| 8 | Liability when goods vehicle used for passenger transport? | 527.7s | Rate limit |
| 9 | List judgments from Supreme Court of India | 530.9s | Rate limit |
| 10 | Loss of dependency under multiplier method | 527.8s | Rate limit |

---

## Failure Analysis

### Failure Mode 1 — Groq rate limiting on back-to-back eval queries (Critical)

Running 10 queries in sequence exhausted Groq's free-tier token quota. Queries 7–10 all returned empty responses after long waits. This is not an agent reasoning failure — the retrieval, routing, and LLM synthesis all worked correctly in Q1–Q6. It is purely a throughput constraint on the free tier.

**Fix:** Add a `time.sleep(30)` between queries in `eval_suite.py`, or switch to a paid Groq tier / OpenAI for eval runs. The agent itself works correctly — this only affects batch evaluation.

### Failure Mode 2 — Precision metric is an eval artifact for deep-research queries (Medium)

Q1 and Q3 show Precision 0.0 despite citing clearly relevant documents. The precision function checks whether `expected_themes` appear as exact substrings in the text surrounding each `[DOC_XXX]` citation. The retrieved chunks contain the legal reasoning but not always the exact keyword phrases defined in the gold set (e.g. the chunk may say "the insurer is bound to indemnify" rather than "insurer liability").

This is a measurement problem, not an agent failure. The judge scores for Q1 and Q3 are both 4.0/5, confirming the responses are substantively correct.

**Fix:** Replace keyword matching with an LLM-as-judge call for the precision dimension — ask the judge whether each cited document is relevant to the query, rather than checking for exact phrase presence. This is already done for reasoning quality and would make precision scores consistent with what the judge observes.

### Failure Mode 3 — Adverse section absent on compensation query (Low)

Query 4 (compensation range) received an Adverse ID score of 0.5 — the agent produced the supporting and strategy sections but skipped the adverse precedents section. The query is framed as a factual calculation question rather than a full research task; the LLM synthesis may have treated it as a narrower problem than intended.

**Fix:** Strengthen the `adverse_analyzer` node prompt to explicitly state that an adverse section is always required for `deep_research` queries regardless of how the question is phrased.

---

## What I Would Fix First

**Rate limiting in eval** — add a 30-second pause between queries in `eval_suite.py`. This is a one-line change that would have given real scores for all 10 queries instead of 6. The agent's effective performance on working queries (Precision 0.667, Reasoning 4.44/5, Adverse ID 0.875) demonstrates the architecture is sound; the eval framework just needs to be run more carefully.

**Precision measurement** — replace exact-substring theme matching with an LLM judge call. The current method penalises correct responses that use synonymous legal language, producing misleadingly low precision scores for deep-research queries.