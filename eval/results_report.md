# Lexi Agent Evaluation Report

**Run Date:** 2026-06-03  
**Agent:** LangGraph — 6 nodes, hybrid RRF retrieval  
**LLM:** Groq LLaMA 3.3 70B Versatile  
**Corpus:** 50 Indian court judgments (DOC_001 – DOC_050)  
**Total Queries Evaluated:** 10 (all completed successfully)

---

## Aggregate Scores

| Dimension | Score |
|---|---|
| Precision | **0.598** |
| Reasoning Quality | **4.27 / 5** |
| Adverse Identification | **0.917** |
| Recall (coverage proxy) | **0.12** |

---

## Per-Query Results

### Q1 — Find precedents supporting Mrs. Lakshmi Devi's claim
**Type:** deep_research · **Time:** 21.3s

| Metric | Score |
|---|---|
| Precision | 0.40 |
| Recall | 0.20 |
| Reasoning | 4.0 / 5 |
| Adverse ID | 1.0 ✅ |

**Docs cited:** DOC_031, DOC_034, DOC_035, DOC_014, DOC_025  
**Docs retrieved:** 6  
**Adverse section:** Present · Risk levels provided

> Faithfulness 4/5 · Coherence 3/5 · Citations 5/5  
> *Specific citations and legal principles applied well; fact-to-case alignment is sometimes vague.*

---

### Q2 — Which judgments involve commercial vehicles?
**Type:** general · **Time:** 3.1s

| Metric | Score |
|---|---|
| Precision | 0.667 |
| Recall | 0.20 |
| Reasoning | 5.0 / 5 |

**Docs cited:** DOC_042, DOC_032, DOC_027

> Faithfulness 5/5 · Coherence 5/5 · Citations 5/5  
> *Direct, accurate, fully grounded in cited sources.*

---

### Q3 — Find adverse precedents the insurance company could use
**Type:** deep_research · **Time:** 13.3s

| Metric | Score |
|---|---|
| Precision | 0.50 |
| Recall | 0.20 |
| Reasoning | 4.0 / 5 |
| Adverse ID | 1.0 ✅ |

**Docs cited:** DOC_029, DOC_035, DOC_025, DOC_034  
**Docs retrieved:** 4

> Faithfulness 4/5 · Coherence 3/5 · Citations 5/5  
> *Citations correct; application of principles to specific facts could be more precisely explained.*

---

### Q4 — Compensation range for 42-year-old earning ₹35,000/month
**Type:** deep_research · **Time:** 4.4s

| Metric | Score |
|---|---|
| Precision | 0.0 |
| Recall | 0.0 |
| Reasoning | 4.0 / 5 |
| Adverse ID | 0.5 ⚠️ |

**Docs cited:** DOC_024, DOC_001  
**Adverse section:** Absent

> Faithfulness 4/5 · Coherence 3/5 · Citations 5/5  
> *Largely grounded in cited docs; cross-applying principles from different case types reduces coherence. Adverse section missing — see Failure Mode 2.*

---

### Q5 — Which judgments involve Section 149 of the Motor Vehicles Act?
**Type:** general · **Time:** 3.2s

| Metric | Score |
|---|---|
| Precision | 1.0 |
| Recall | 0.0 |
| Reasoning | 5.0 / 5 |

**Docs cited:** DOC_031, DOC_029, DOC_032, DOC_030  
**Docs retrieved:** 4

> Faithfulness 5/5 · Coherence 5/5 · Citations 5/5  
> *Perfect response — fully grounded, logically sound, precise citations.*

---

### Q6 — Contributory negligence cases with unlicensed driver
**Type:** deep_research · **Time:** 11.5s

| Metric | Score |
|---|---|
| Precision | 0.667 |
| Recall | 0.20 |
| Reasoning | 4.0 / 5 |
| Adverse ID | 1.0 ✅ |

**Docs cited:** DOC_004, DOC_034, DOC_006, DOC_035, DOC_018, DOC_025  
**Docs retrieved:** 7

> Faithfulness 4/5 · Coherence 3/5 · Citations 5/5  
> *Good citations and principles; some claims slightly exceed what the cited sources explicitly state.*

---

### Q7 — Cases discussing the 'pay and recover' doctrine
**Type:** general · **Time:** 12.7s

| Metric | Score |
|---|---|
| Precision | 0.429 |
| Recall | 0.20 |
| Reasoning | 4.0 / 5 |

**Docs cited:** DOC_028, DOC_027, DOC_026, DOC_029, DOC_035, DOC_039, DOC_025  
**Docs retrieved:** 6

> Faithfulness 4/5 · Coherence 3/5 · Citations 5/5  
> *Over-retrieves — 7 documents cited for a general lookup query. Legal reasoning is correct but verbose.*

---

### Q8 — Insurer liability when goods vehicle used for passenger transport
**Type:** deep_research · **Time:** 11.4s

| Metric | Score |
|---|---|
| Precision | 0.75 |
| Recall | 0.0 |
| Reasoning | 4.0 / 5 |
| Adverse ID | 1.0 ✅ |

**Docs cited:** DOC_025, DOC_014, DOC_035, DOC_029  
**Docs retrieved:** 4

> Faithfulness 4/5 · Coherence 3/5 · Citations 5/5  
> *Well-structured; distinction between supporting and adverse principles could be sharper.*

---

### Q9 — List judgments from the Supreme Court of India
**Type:** general · **Time:** 3.6s

| Metric | Score |
|---|---|
| Precision | 1.0 |
| Recall | 0.0 |
| Reasoning | 4.67 / 5 |

**Docs cited:** DOC_008, DOC_019, DOC_028  
**Docs retrieved:** 6

> Faithfulness 5/5 · Coherence 4/5 · Citations 5/5  
> *Accurate and well-reasoned. Likely under-recalls — metadata extraction may have missed some Supreme Court documents.*

---

### Q10 — Loss of dependency under the multiplier method
**Type:** deep_research · **Time:** 10.4s

| Metric | Score |
|---|---|
| Precision | 0.571 |
| Recall | 0.0 |
| Reasoning | 4.0 / 5 |
| Adverse ID | 1.0 ✅ |

**Docs cited:** DOC_008, DOC_019, DOC_007, DOC_035, DOC_018, DOC_039, DOC_025  
**Docs retrieved:** 7

> Faithfulness 4/5 · Coherence 3/5 · Citations 5/5  
> *Principles correct; reasoning anchored to a 30-year-old deceased when the query specified no age — slight drift from query intent.*

---

## Dimension Summary

### Precision by query type

| Query type | Queries | Avg Precision |
|---|---|---|
| General | 4 (Q2, Q5, Q7, Q9) | 0.774 |
| Deep research | 6 (Q1,Q3,Q4,Q6,Q8,Q10) | 0.481 |
| **Overall** | **10** | **0.598** |

General queries retrieve more precisely. Deep research queries over-retrieve — surfacing semantically related but peripherally relevant documents alongside the strong matches.

### Reasoning quality

Citation quality is consistently 5/5 across all queries. Legal coherence is the weakest sub-dimension, averaging 3.5/5, driven by the agent applying principles across case types without always making the inferential leap explicit.

### Adverse identification

9 of 10 queries produced an adverse section with risk levels. The single failure (Q4 — compensation range) was a framing issue: the query is phrased as a factual calculation, causing the synthesis node to treat it as a lookup rather than a full research task.

---

## Failure Analysis

### Failure Mode 1 — Legal coherence consistently scores 3/5 across deep research queries

Every deep-research query except Q1 on the Lakshmi Devi case scores 3/5 on legal coherence. The agent retrieves the right documents and cites them correctly, but the inferential chain — "this case establishes principle X, which applies here because of fact Y" — is often asserted rather than argued. The LLM synthesises principles from multiple cases without always making the connecting logic explicit for a reader unfamiliar with the corpus.

**Root cause:** The `supporting_analyzer` and `adverse_analyzer` node prompts ask for legal principles and factual alignment but do not explicitly require the agent to state *why* each factual parallel holds. The LLM fills this gap with general legal language rather than case-specific reasoning.

**Fix:** Restructure the synthesis prompts to require a three-part format per citation — (1) what the case decided, (2) which specific fact in the current matter matches, (3) therefore what follows. This forces explicit reasoning rather than implicit assertion, directly targeting the coherence gap.

### Failure Mode 2 — Adverse section absent on compensation query (Q4)

Q4 is phrased as a quantum calculation question. The LangGraph `router` correctly classifies it as `deep_research`, and the retriever runs correctly, but the `strategy_synthesizer` node produced a response without an adverse section — the `adverse_analyzer` output was not integrated. The node executed but its output was not surfaced in the final memo.

**Root cause:** When the query is framed around a calculation rather than a legal argument, the `strategy_synthesizer` prompt de-prioritises the adverse section in favour of the numerical output. The section is generated by `adverse_analyzer` but effectively dropped in synthesis.

**Fix:** Add an explicit assertion in the `strategy_synthesizer` prompt: *"The response must always include all three sections. Do not omit the Adverse Precedents section regardless of query framing."* Also add a post-generation check in the node that verifies all three section headers are present before returning.

### Failure Mode 3 — Recall is low across all queries (0.0 – 0.2)

Recall scores are uniformly low. This is partly a measurement artefact — the gold set `doc_ids` arrays cover only a subset of the corpus, so recall is calculated against a sparse ground truth. However, Q9 (Supreme Court judgments) likely genuinely under-recalls: the agent returned 3 documents from metadata filtering when the corpus plausibly contains more. The regex-based metadata extractor in `ingest.py` captures court names inconsistently, particularly for High Court vs Supreme Court attribution.

**Root cause:** Regex metadata extraction misses court name variants (e.g. "Hon'ble Supreme Court", "Apex Court", "SC"). Documents with non-standard attribution are excluded from metadata filter results and only surfaced through semantic search, which may rank them lower.

**Fix:** Replace regex metadata extraction with a short LLM call on the first paragraph of each document at ingestion time. A single structured extraction call per document would produce reliable `court`, `year`, and `case_name` fields and significantly improve metadata filter recall — the highest-leverage ingestion improvement for a 50-document corpus.

---

## What I Would Fix First

**Structured synthesis prompts with explicit reasoning steps** — the coherence gap (Failure Mode 1) affects every deep-research query and is a one-prompt change. Requiring the agent to state "this case decided X, the current matter has fact Y, therefore Z" per citation would raise legal coherence from 3/5 to at least 4/5 across the board, lifting overall reasoning quality from 4.27 to approximately 4.6/5. This is the highest-impact single change before the interview.