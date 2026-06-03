# Lexi — Legal Precedent Research Agent

> **GitHub:** https://github.com/DineshSaiTejaP/lexi-research-agent  
> **Live Demo:** https://lexi-research-agent-assessment-dinesh.streamlit.app/

An AI-powered legal research agent that searches a corpus of 50+ Indian court judgments to identify supporting and adverse precedents, built for the Lexi Backend Engineer take-home assessment.

---

## Architecture Overview

```
User Query
    │
 [router]           ← LLM classifies: general vs deep_research
    │
 [retriever]        ← Vector search + BM25 keyword + adversarial pass
    │
 ┌──┴─────────────────────┐
[general_answerer]   [supporting_analyzer]
    │                      │
   END              [adverse_analyzer]
                           │
                   [strategy_synthesizer]
                           │
                          END
```

- **Agent**: LangGraph state machine — 6 nodes, LLM-driven conditional routing
- **Retrieval**: Hybrid vector search (sentence-transformers + ChromaDB) + BM25 keyword search, merged via Reciprocal Rank Fusion
- **LLM**: Groq LLaMA 3.3 70B Versatile (configurable — also supports Google Gemini, OpenAI)
- **UI**: Streamlit with full intermediate reasoning trace visible
- **Eval**: Automated framework measuring Precision, Recall, Reasoning Quality, Adverse Identification

See [ADR.md](./ADR.md) for full architecture decisions and tradeoffs.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/DineshSaiTejaP/lexi-research-agent
cd lexi-research-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API key
```

**Default (Groq — free tier, recommended):**
```
GROQ_API_KEY=your_groq_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
CORPUS_DIR=./data/judgments
CHROMA_PERSIST_DIR=./chroma_db
TOP_K_RETRIEVAL=10
```

Get a free Groq API key: https://console.groq.com/

**OR use Google Gemini:**
```
GOOGLE_API_KEY=your_google_key_here
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash
```

**OR use OpenAI:**
```
OPENAI_API_KEY=your_openai_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

### 4. Add the judgment corpus

Place the PDF judgment files (DOC_001.pdf through DOC_050.pdf) in:
```
data/judgments/
```

### 5. Run ingestion (one-time setup)

> **Skip this step if deploying** — `chroma_db/` is pre-built and committed to the repo.

```bash
python ingest.py
```

This will:
- Load all PDFs from `data/judgments/`
- Extract and chunk text (~800 chars per chunk)
- Generate embeddings with sentence-transformers (`all-MiniLM-L6-v2`)
- Store in ChromaDB at `./chroma_db/`

Takes approximately 5–10 minutes for 50 documents.

### 6. Launch the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Running the Evaluation

```bash
python eval/eval_suite.py
```

Results are saved to:
- `eval/results_report.md` — human-readable report
- `eval/eval_raw_results.json` — raw scores for all queries

**Note:** The reasoning quality dimension uses an LLM-as-judge call. Set `LLM_PROVIDER` and the corresponding API key in your `.env` before running.

**To enable exact recall scoring**, update `eval/gold_set.json` with manually labelled relevant document IDs after reviewing the corpus.

---

## Repository Structure

```
lexi-research-agent/
├── app.py                  # Streamlit UI
├── agent.py                # LangGraph agent (6 nodes, conditional routing)
├── ingest.py               # PDF ingestion pipeline
├── db.py                   # Shared ChromaDB + embedder singletons
├── smoke_test.py           # Quick retrieval sanity check
├── tools/
│   ├── __init__.py         # Hybrid RRF search
│   ├── vector_search.py    # ChromaDB semantic search
│   ├── keyword_search.py   # BM25 keyword search
│   └── metadata_filter.py  # Filter by court/year/topic
├── eval/
│   ├── eval_suite.py       # Automated evaluation framework
│   ├── gold_set.json       # Labelled relevant judgments + eval queries
│   └── results_report.md   # Evaluation results
├── data/
│   └── judgments/          # DOC_001.pdf ... DOC_050.pdf (gitignored)
├── chroma_db/              # Pre-built vector store (committed for deployment)
├── ADR.md                  # Architecture Decision Record
├── requirements.txt
└── .env.example
```

---

## How It Works

### Query Routing
The LangGraph `router` node makes an LLM call to classify query intent:

| Query Type | Example | Path |
|------------|---------|------|
| `general` | "Which cases involve trucks?" | router → retriever → general_answerer → END |
| `deep_research` | "Find supporting precedents for Mrs. Lakshmi Devi" | router → retriever → supporting_analyzer → adverse_analyzer → strategy_synthesizer → END |

### Research Output Structure
For deep research queries, the agent always produces all three sections:
1. **Supporting Precedents** — judgments that help the client, with legal principles and factual alignment
2. **Adverse Precedents** — judgments that hurt the client, with risk levels (HIGH/MEDIUM/LOW) and counter-strategies
3. **Strategy Recommendation** — priority arguments, realistic compensation range, next steps

### Intermediate Steps Visible in UI
- 💭 Thought steps (router classification, section completions)
- 🔧 Tool calls (which search ran, with query)
- 📄 Tool results (what was retrieved, truncated for display)

---

## Evaluation Results

See [eval/results_report.md](./eval/results_report.md) for full per-query breakdown and failure analysis.

| Dimension | Score |
|-----------|-------|
| Precision | 0.850 |
| Reasoning Quality | 4.20 / 5 |
| Adverse ID | 0.875 |
| Recall (Coverage Proxy) | 0.900 |

---

## Deployment

The app is deployed on **Streamlit Community Cloud**:

1. Push this repo to GitHub (keep `chroma_db/` committed — it is the pre-indexed vector store)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select this repo → `app.py`
3. Under **Secrets**, add:
```toml
GROQ_API_KEY = "your_key_here"
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
CHROMA_PERSIST_DIR = "./chroma_db"
TOP_K_RETRIEVAL = "10"
```
4. Deploy — the app will be live at a `*.streamlit.app` URL

