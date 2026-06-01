# Architecture Decision Record (ADR)
## Lexi Legal Precedent Research Agent

---

## 1. Architecture Choice: ReAct Agent

### Decision
I built the agent using the **ReAct (Reason + Act)** pattern via LangChain's `create_react_agent`, rather than a fixed pipeline or graph-based workflow.

### Rationale
The assessment explicitly required that the agent "dynamically determine its own workflow rather than following hard-coded steps." ReAct satisfies this naturally:

- The LLM decides, at runtime, whether a query needs 1 tool call or 6
- It can self-correct if initial retrieval returns poor results
- It handles the full spectrum from "Which cases involve trucks?" (1 tool call) to "Find precedents supporting our client's insurance claim" (4-6 tool calls with synthesis)

### Alternatives Considered
| Option | Why Rejected |
|--------|-------------|
| Fixed pipeline (retrieve → rank → synthesize) | Violates the "no hard-coded steps" requirement; brittle for diverse queries |
| LangGraph state machine | More complex than needed for 50 docs; harder to explain without domain-specific justification |
| CrewAI / AutoGen | Explicitly prohibited by the assessment |
| Simple RAG (no agent) | Cannot handle multi-step research, cannot distinguish query types |

---

## 2. Retrieval Strategy: Hybrid Vector + BM25

### Decision
**Two retrieval methods combined via Reciprocal Rank Fusion (RRF):**
1. Dense vector search (sentence-transformers `all-MiniLM-L6-v2` + ChromaDB)
2. Sparse keyword search (BM25 via `rank_bm25`)

### Rationale

**Why both?** Legal documents contain a mix of semantic concepts and exact statutory references. Vector search excels at semantic similarity ("cases about insurance liability") but can miss exact phrases ("Section 149 of the Motor Vehicles Act"). BM25 catches the exact terminology that practitioners know to look for.

**Why RRF for merging?** Reciprocal Rank Fusion is a well-established technique (Cormack et al., 2009) that merges ranked lists without requiring score normalization. It outperforms simple score-based merging for heterogeneous retrieval systems and requires no additional API calls or models.

**Why `all-MiniLM-L6-v2`?** Free, runs locally, 384-dimensional embeddings, achieves strong performance on semantic textual similarity benchmarks. The alternative (OpenAI `text-embedding-3-small`) would add cost and an API dependency for a 50-doc corpus where latency isn't critical.

**Why ChromaDB?** Zero infrastructure overhead — runs in-process, persists to disk, supports metadata filtering, and handles 50 documents trivially. The hosted vector store (Qdrant Cloud, Pinecone) would be overkill for this scale.

---

## 3. Chunking Approach

### Decision
**Recursive character text splitter:** ~500 token chunks (≈2000 chars) with 50-token (≈200 char) overlap, using paragraph/sentence boundaries as preferred split points.

### Rationale
Indian court judgments follow a predictable structure: facts → issues → arguments → holding → order. Chunks of 500 tokens capture one logical section (e.g., a factual finding or a legal holding) without splitting mid-argument. The 50-token overlap prevents reasoning at chunk boundaries from being lost.

**Why not semantic chunking?** True semantic chunking (embedding-based) would be superior but requires an LLM call per document at ingestion time, adding cost and latency. For 50 documents, the quality gain doesn't justify the tradeoff.

**What I extract per chunk:** `doc_id`, `page_number`, `court`, `year`, `case_name`, `topics` (via regex), `is_motor_accident` (boolean). This metadata powers the third tool (`metadata_filter`) and appears in search results.

---

## 4. How the Agent Decides Query Complexity

### Decision
The **system prompt** instructs the LLM to distinguish query types, not code-level branching:

```
For simple factual queries → use 1-2 tool calls, direct answer
For deep precedent research → use vector + keyword search, surface both 
                              supporting AND adverse precedents, synthesize
                              into three-section research memo
```

### Rationale
Any code-level if-else classification would need to enumerate query types — fragile and hard to maintain. The LLM's own judgment is more robust. The system prompt provides examples and structures the output format, giving the agent enough context to self-classify without external logic.

---

## 5. Tradeoffs Made

| Tradeoff | Choice | Reason |
|----------|--------|--------|
| Embedding cost vs. quality | Local sentence-transformers | Free, no API dependency |
| Retrieval depth | Top-10 per method → merge | Balances context window usage vs. coverage |
| Metadata extraction | Regex-based | Fast, no LLM calls needed for 50 docs |
| Agent memory | Stateless per request | Simplifies Streamlit deployment; history shown in UI |
| Reranking | RRF (no neural reranker) | Free; Cohere reranker would improve precision at cost |

---

## 6. Scaling to 5,000 Documents

If the corpus grew from 50 to 5,000 documents, I would change:

1. **Vector store → Qdrant Cloud or Pinecone**: ChromaDB is not designed for millions of vectors. These hosted solutions support hybrid search natively, horizontal scaling, and production SLAs.

2. **Embeddings → OpenAI `text-embedding-3-small` or Cohere Embed**: Higher-dimensional, better legal domain performance. The API cost is justified at scale (~$0.02/1000 pages).

3. **Metadata extraction → LLM-assisted**: At 5,000 docs, regex-based extraction will miss too many cases. Use a fast LLM call on the first page of each document to extract structured metadata (case name, court, parties, citations, legal issues).

4. **Ingestion → async pipeline**: Add a queue (Celery + Redis) for async ingestion. New documents added via API without blocking the app.

5. **Add a neural reranker**: After initial retrieval, use Cohere Rerank or a cross-encoder to rerank the top-50 results to top-10. This significantly improves precision.

6. **Chunking → hierarchical**: Add a document-level summary embedding alongside chunk-level embeddings. This enables "retrieve document, then retrieve relevant chunks within it" — better for long judgments.

---

## 7. What I Would Change With Another Week

1. **Streaming responses**: Stream the LLM output token-by-token to Streamlit using `StreamingStdOutCallbackHandler`, so users see the answer being written rather than waiting 20-30 seconds.

2. **Citation extraction**: Post-process the agent's answer to extract all DOC_XXX citations and display them as a clickable reference panel with the relevant excerpt highlighted.

3. **Query expansion**: Before retrieval, ask the LLM to generate 3 related search queries from the user's input, run all of them, and merge results. This improves recall significantly.

4. **Evaluation with human labels**: Manually review the 50 judgments to create a proper gold set for the Lakshmi Devi case (5-10 supporting, 3-5 adverse). This would enable exact precision/recall scores instead of proxies.

5. **Conversation memory**: Add LangChain's `ConversationBufferMemory` to allow follow-up queries like "Now find the compensation range from those cases." Currently each query is stateless.

6. **Better adverse detection**: Add a dedicated "adversarial search" step that specifically prompts the agent to search for cases where claimants lost, rather than relying on the general prompt to surface them.

---

*This ADR covers the key architectural decisions for the Lexi research agent. Prepared for the Lexi Backend Engineer take-home assessment, 2026.*
