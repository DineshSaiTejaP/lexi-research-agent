# Lexi — Legal Precedent Research Agent

> **GitHub:** https://github.com/DineshSaiTejaP/lexi-research-agent
> **Live Demo:** [Deploying to Streamlit Cloud — URL coming soon]

An AI-powered legal research agent that searches a corpus of 50+ Indian court judgments to identify supporting and adverse precedents, built for the Lexi Backend Engineer take-home assessment.

---

## Architecture Overview

```
User Query → ReAct Agent (LangChain) → Tools → ChromaDB + BM25
                    ↓
         Hybrid RRF Retrieval (Vector + Keyword)
                    ↓
         Structured Research Memo
         (Supporting Precedents | Adverse Precedents | Strategy)
```

- **Agent**: ReAct (Reason + Act) — dynamically determines its own workflow
- **Retrieval**: Hybrid vector search (sentence-transformers + ChromaDB) + BM25 keyword search, merged via Reciprocal Rank Fusion
- **LLM**: Google Gemini 1.5 Flash (configurable to OpenAI)
- **UI**: Streamlit with intermediate reasoning steps visible
- **Eval**: Automated framework measuring Precision, Recall, Reasoning Quality, Adverse Identification

See [ADR.md](./ADR.md) for full architecture decisions and tradeoffs.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/lexi-research-agent
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

**Required in `.env`:**
```
GOOGLE_API_KEY=your_google_api_key_here
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash
```

Get a free Google API key: https://aistudio.google.com/

**OR use OpenAI:**
```
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

### 4. Add the judgment corpus

Place the 50 PDF judgment files (DOC_001.pdf through DOC_050.pdf) in:
```
data/judgments/
```

### 5. Run ingestion (one-time setup)

```bash
python ingest.py
```

This will:
- Load all PDFs from `data/judgments/`
- Extract and chunk text
- Generate embeddings with sentence-transformers
- Store in ChromaDB at `./chroma_db/`

Takes approximately 5-10 minutes for 50 documents.

### 6. Launch the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Running the Evaluation

After the app is running and you've tested some queries:

```bash
python eval/eval_suite.py
```

Results are saved to:
- `eval/results_report.md` — human-readable report
- `eval/eval_raw_results.json` — raw scores for all queries

**To enable exact recall scoring**, update `eval/gold_set.json` with manually labeled relevant document IDs after reviewing the corpus.

---

## Repository Structure

```
lexi-research-agent/
├── app.py                  # Streamlit UI
├── agent.py                # ReAct agent (LangChain)
├── ingest.py               # PDF ingestion pipeline
├── callbacks.py            # Step capture for UI display
├── tools/
│   ├── __init__.py         # Hybrid RRF search
│   ├── vector_search.py    # ChromaDB semantic search
│   ├── keyword_search.py   # BM25 keyword search
│   └── metadata_filter.py  # Filter by court/year/topic
├── eval/
│   ├── eval_suite.py       # Automated evaluation framework
│   ├── gold_set.json       # Labeled relevant judgments
│   └── results_report.md   # Evaluation results
├── data/
│   └── judgments/          # DOC_001.pdf ... DOC_050.pdf
├── chroma_db/              # Persisted vector store
├── ADR.md                  # Architecture Decision Record
├── requirements.txt
└── .env.example
```

---

## How It Works

### Query Handling
The ReAct agent decides its own strategy per query:

| Query Type | Example | Agent Behavior |
|------------|---------|----------------|
| General | "Which cases involve trucks?" | 1-2 tool calls, direct list |
| Research | "Find supporting precedents for Mrs. Lakshmi Devi" | 4-6 tool calls, full memo |
| Targeted | "Section 149 Motor Vehicles Act cases" | Keyword search + metadata filter |

### Research Output Structure
For deep research queries:
1. **Supporting Precedents** — judgments that help the client, with legal principles and factual alignment
2. **Adverse Precedents** — judgments that hurt the client, with risk levels and counter-strategies
3. **Strategy Recommendation** — priority arguments, compensation range, next steps

### Intermediate Steps
All agent reasoning steps are visible in the Streamlit UI:
- 💭 Agent thoughts
- 🔧 Tool calls (which search was run)
- 📄 Tool results (what was retrieved)

---

## Evaluation Results

See [eval/results_report.md](./eval/results_report.md) for full results.

| Dimension | Score |
|-----------|-------|
| Precision | [Run eval to populate] |
| Reasoning Quality | [Run eval to populate] |
| Adverse ID | [Run eval to populate] |

---

## Deployment

The app is deployed on **Streamlit Community Cloud**:

1. Push this repo to GitHub (keep `chroma_db/` committed — it's the pre-indexed vector store)
2. Connect the repo to [share.streamlit.io](https://share.streamlit.io)
3. Set secrets: `GOOGLE_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`
4. Deploy — the app will be live at a `streamlit.app` URL

---

## Contact

Assessment submission: pradeep.kumaar@zyoin.com  
Subject: `[LEXI-BE-2026] Your Full Name - Research Assessment Submission`
