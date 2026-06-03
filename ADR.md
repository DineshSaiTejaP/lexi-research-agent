# Architecture Decision Record (ADR)
## Lexi Legal Precedent Research Agent
**Live Demo:** https://lexi-research-agent-assessment-dinesh.streamlit.app/
---

## 1. Architecture Choice: LangGraph State Machine

### Decision
I updated the agent to use a **LangGraph Tool-Calling (ReAct)** architecture. The LLM acts as a central decision-maker equipped with distinct search tools, rather than moving through hard-coded pipeline nodes.

### Graph Topology
```
START → [router] → [retriever] → (conditional edge)
                                    ├── [general_answerer] → END
                                    └── [supporting_analyzer]
                                              ↓
                                       [adverse_analyzer]
                                              ↓
                                    [strategy_synthesizer] → END
```

### Rationale
The assessment required the agent to "dynamically determine its own workflow rather than following hard-coded steps." LangGraph satisfies this with LLM-driven routing:

- The `router` node makes a single LLM call to classify the query as `general` or `deep_research`. No hardcoded keywords — the LLM reasons about intent.
- The conditional edge after `retriever` branches based on that classification. Simple queries get one LLM call; research queries get four.
- Each node is a pure function over `AgentState` — easy to test, extend, and reason about independently.
- The `adverse_analyzer` is a **dedicated node**, not a prompt instruction. This guarantees adverse precedents are always generated for deep research queries, regardless of how many supporting cases were retrieved.

### Alternatives Considered
| Option | Why Rejected |
|--------|-------------|
| Fixed pipeline (retrieve → rank → synthesize) | Violates the "no hard-coded steps" requirement; brittle for diverse queries |
| ReAct (single LLM loop with tools) | Less predictable — the LLM can decide to skip adverse analysis; harder to guarantee all three output sections are generated |
| CrewAI / AutoGen | Explicitly prohibited by the assessment |
| Simple RAG (no agent) | Cannot handle multi-step research, cannot distinguish query types |

---

## 2. Retrieval Strategy: Hybrid Vector + BM25

### Decision
**Two retrieval methods combined via Reciprocal Rank Fusion (RRF):**
1. Dense vector search (sentence-transformers `all-MiniLM-L6-v2` + ChromaDB)
2. Sparse keyword search (BM25 via `rank_bm25`)

For deep research queries, a **third adversarial retrieval pass** runs automatically with an insurer-favouring query, ensuring adverse precedents are retrieved even when the user's query is claimant-focused.

### Rationale

**Why both?** Legal documents contain a mix of semantic concepts and exact statutory references. Vector search excels at semantic similarity ("cases about insurance liability") but can miss exact phrases ("Section 149 of the Motor Vehicles Act"). BM25 catches the exact terminology that practitioners know to look for.

**Why RRF for merging?** Reciprocal Rank Fusion (Cormack et al., 2009) merges ranked lists without requiring score normalisation. It outperforms simple score-based merging for heterogeneous retrieval systems and requires no additional API calls or models.

**Why `all-MiniLM-L6-v2`?** Free, runs locally, 384-dimensional embeddings, strong performance on semantic textual similarity benchmarks. The alternative (OpenAI `text-embedding-3-small`) would add cost and an API dependency for a 50-doc corpus where latency is not critical.

**Why ChromaDB?** Zero infrastructure overhead — runs in-process, persists to disk, supports metadata filtering, and handles 50 documents trivially. Hosted vector stores (Qdrant Cloud, Pinecone) would be overkill at this scale.

---

## 3. Chunking Approach

### Decision
**Recursive character text splitter:** ~170 token chunks (≈800 chars) with ~25-token (≈100 char) overlap, using paragraph/sentence boundaries as preferred split points.

### Rationale
Indian court judgments follow a predictable structure: facts → issues → arguments → holding → order. Chunks of ~800 characters capture one logical section (e.g., a factual finding or a legal holding) without splitting mid-argument. The 100-character overlap prevents reasoning at chunk boundaries from being lost.

**Why not semantic chunking?** True semantic chunking (embedding-based) would be superior but requires an LLM call per document at ingestion time, adding cost and latency. For 50 documents, the quality gain does not justify the tradeoff.

**What I extract per chunk:** `doc_id`, `court`, `year`, `case_name`, `topics` (via regex), `is_motor_accident` (boolean). This metadata powers the `metadata_filter` tool and appears in search results for citation verification.

---

## 4. How the Agent Decides Query Complexity

### Decision
The `router` node makes an **LLM call** to classify the query. The result is stored in `AgentState.query_type` and read by the `route_after_retrieval` conditional edge function. No keyword matching, no if/else branching in application code.

```python
def route_after_retrieval(state: AgentState) -> str:
    return "general_answerer" if state["query_type"] == "general" else "supporting_analyzer"
```

### Rationale
Any code-level classification would need to enumerate query patterns — fragile and hard to maintain. Delegating to the LLM is more robust. The router prompt is deliberately minimal: it asks for one word (`general` or `deep_research`) and relies on the LLM's own understanding of legal research intent.

**Failure mode considered:** If the router misclassifies a general query as `deep_research`, the agent over-delivers (three sections instead of one). If it misclassifies a research query as `general`, it under-delivers. The code defaults to `deep_research` on any router error — always over-delivering is safer in a legal research context.

---

## 5. Tradeoffs Made

| Tradeoff | Choice | Reason |
|----------|--------|--------|
| Embedding cost vs. quality | Local sentence-transformers | Free, no API dependency |
| Retrieval depth | Top-8 per method | Balances context window usage vs. coverage |
| Metadata extraction | Regex-based | Fast; no LLM calls needed for 50 docs |
| Agent memory | Stateless per request | Simplifies Streamlit deployment; history shown in UI |
| Reranking | RRF only (no neural reranker) | Free; Cohere reranker would improve precision at cost |
| Adverse detection | Dedicated `adverse_analyzer` node + separate retrieval pass | Guarantees adverse section is always generated |

---

## 6. Scaling to 5,000 Documents

If the corpus grew from 50 to 5,000 documents, I would change:

1. **Vector store → Qdrant Cloud or Pinecone**: ChromaDB is not designed for millions of vectors. These hosted solutions support hybrid search natively, horizontal scaling, and production SLAs.

2. **Embeddings → OpenAI `text-embedding-3-small` or Cohere Embed**: Higher-dimensional embeddings with better legal domain performance. The API cost (~$0.02/1000 pages) is justified at scale.

3. **Metadata extraction → LLM-assisted**: At 5,000 docs, regex-based extraction misses too many cases. Use a fast LLM call on the first page of each document to extract structured metadata (case name, court, parties, cited statutes, legal issues).

4. **Ingestion → async pipeline**: Add a queue (Celery + Redis) for async ingestion so new documents can be added via API without blocking the app.

5. **Add a neural reranker**: After initial retrieval, use Cohere Rerank or a cross-encoder to rerank the top-50 results to top-10. This significantly improves precision on large corpora.

6. **Chunking → hierarchical**: Add a document-level summary embedding alongside chunk-level embeddings. This enables "retrieve document, then retrieve relevant chunks within it" — better for long judgments where the key holding is far from the relevant facts.

---

## 7. What I Would Change With Another Week

1. **Streaming responses**: Stream the LLM output token-by-token to Streamlit so users see the answer being written rather than waiting 20–30 seconds for the full memo.

2. **Query expansion node**: Add a pre-retrieval LangGraph node that uses the LLM to generate 3–4 synonymous search queries (e.g., expanding "commercial vehicle" to include "truck, lorry, transport vehicle, goods vehicle"). Run all of them, merge via RRF. This would significantly improve BM25 recall.

3. **Citation extraction + PDF deep-link**: Post-process the agent's answer to extract all `[DOC_XXX]` citations and render them as clickable links that open the original PDF at the relevant page.

4. **Expand Evaluation Gold Set**: While `eval/gold_set.json` has been initially populated with ground-truth labels for precision/recall, we can expand it to cover more queries and edge cases to further refine our exact evaluation scores.

5. **Conversation memory**: Add LangGraph's `MemorySaver` checkpointer to allow follow-up queries ("Now find the compensation range from those cases"). Currently each query is stateless.
