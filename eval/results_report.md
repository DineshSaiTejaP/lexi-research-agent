# Lexi Agent Evaluation Report

**Run Date:** 2026-06-02T22:15:00
**Total Queries Evaluated:** 5

## Aggregate Scores

| Dimension | Score |
|-----------|-------|
| Precision | 0.850 |
| Reasoning Quality | 4.20/5 |
| Adverse ID | 0.875 |
| Recall Proxy | 0.900 |

---

## Per-Query Results

### Query 1: Find precedents supporting Mrs. Lakshmi Devi's claim against the insurance company
- **Type:** deep_research
- **Response Time:** 28.5s
- **Precision Score:** 0.833
- **Docs Cited:** DOC_012, DOC_034, DOC_041
- **Recall Proxy:** 1.0 (Multiple search strategies used, 15 docs retrieved)
- **Reasoning Score:** 4.33/5
  - Faithfulness: 4/5
  - Legal Coherence: 4/5
  - Citation Quality: 5/5
  - Comment: Excellent use of the "pay and recover" doctrine based on cited docs.
- **Adverse ID Score:** 0.750
  - Has adverse section: True
  - Has risk assessment: True

### Query 2: Which judgments involve commercial vehicles?
- **Type:** general
- **Response Time:** 12.1s
- **Precision Score:** 1.000
- **Docs Cited:** DOC_005, DOC_012, DOC_022, DOC_041
- **Recall Proxy:** 0.8 (Keyword search was highly effective here)
- **Reasoning Score:** 4.67/5
  - Faithfulness: 5/5
  - Legal Coherence: 4/5
  - Citation Quality: 5/5
  - Comment: Straightforward factual retrieval.

### Query 3: Find adverse precedents the insurance company could use against the claimant
- **Type:** deep_research
- **Response Time:** 24.2s
- **Precision Score:** 0.800
- **Docs Cited:** DOC_009, DOC_027
- **Recall Proxy:** 0.9
- **Reasoning Score:** 4.00/5
  - Faithfulness: 4/5
  - Legal Coherence: 4/5
  - Citation Quality: 4/5
  - Comment: Correctly identified fundamental breach of policy precedents.
- **Adverse ID Score:** 1.000
  - Has adverse section: True
  - Has risk assessment: True

### Query 4: What compensation range is realistic for a 42-year-old with monthly income of Rs 35000?
- **Type:** deep_research
- **Response Time:** 32.4s
- **Precision Score:** 0.750
- **Docs Cited:** DOC_018, DOC_031
- **Recall Proxy:** 0.8
- **Reasoning Score:** 3.67/5
  - Faithfulness: 4/5
  - Legal Coherence: 3/5
  - Citation Quality: 4/5
  - Comment: Struggled slightly with exact multiplier calculations but cited relevant dependency cases.
- **Adverse ID Score:** 0.875
  - Has adverse section: True
  - Has risk assessment: True

### Query 5: Which judgments involve Section 149 of the Motor Vehicles Act?
- **Type:** general
- **Response Time:** 14.5s
- **Precision Score:** 0.867
- **Docs Cited:** DOC_012, DOC_015, DOC_034, DOC_045
- **Recall Proxy:** 1.0
- **Reasoning Score:** 4.33/5
  - Faithfulness: 5/5
  - Legal Coherence: 4/5
  - Citation Quality: 4/5
  - Comment: Good mix of vector and keyword search matches.

---

## Failure Analysis

### Top Failure Modes
1. **Mathematical Reasoning and Multipliers**: The agent struggles with precise quantum calculations (e.g., applying the exact multiplier for a 42-year-old under the Sarla Verma framework). It successfully retrieves the right compensation cases but occasionally misinterprets the mathematical formulas within them.
2. **Context Window Saturation on Adverse Extraction**: In deep research queries, if the initial search returns an overwhelming number of supporting cases, the agent's context window fills up, leading to a shallow analysis of adverse precedents unless specifically prompted.
3. **Keyword Search Brittle on Synonyms**: The BM25 retrieval occasionally misses documents if the exact phrasing isn't used (e.g., "lorry" vs "truck"), relying entirely on the vector search to pick up the slack. Sometimes the vector search ranks a weakly-related semantic match higher than a precise statutory match.

### What I Would Fix First
**1. Implement a Structured Output Parser for Citations**
Currently, the agent relies on the prompt to format citations as `[DOC_XXX]`. This occasionally breaks. I would implement LangChain's `StructuredOutputParser` or use Pydantic models to force the LLM to return a strictly typed list of citations and risk levels, ensuring the UI can reliably render links to the source PDFs.

**2. Query Expansion Step**
To fix the keyword brittleness, I would add a pre-retrieval LLM call that takes the user's query and generates 3-4 synonymous search queries (e.g., expanding "commercial vehicle" to include "truck, lorry, transport vehicle"). This would drastically improve the BM25 recall before merging via RRF.

**3. Separate "Adverse" Search Call**
Instead of trusting the agent to find adverse cases in a single large search, I would modify the ReAct loop to explicitly execute a second tool call conceptually framed as: *"Now search for the opposing counsel's strongest arguments."* This ensures adverse precedents aren't crowded out by supporting ones.