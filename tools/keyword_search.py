"""BM25 keyword search — catches exact legal terms that semantic search can miss."""

import json
from functools import lru_cache

from rank_bm25 import BM25Okapi
from langchain_core.tools import Tool
from db import get_collection, TOP_K


@lru_cache(maxsize=1)
def _build_index():
    """Load all documents from ChromaDB and build a BM25 index. Cached per session."""
    col = get_collection()
    result = col.get(limit=col.count(), include=["documents", "metadatas"])
    docs = result["documents"]
    metas = result["metadatas"]
    ids = result["ids"]
    index = BM25Okapi([d.lower().split() for d in docs])
    return index, docs, metas, ids


def keyword_search(query: str, top_k: int = TOP_K) -> list[dict]:
    index, docs, metas, ids = _build_index()
    scores = index.get_scores(query.lower().split())

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    hits = []
    for idx in ranked:
        if scores[idx] <= 0:
            continue
        meta = metas[idx]
        hits.append({
            "doc_id": meta.get("doc_id", ""),
            "chunk_id": ids[idx],
            "case_name": meta.get("case_name", meta.get("doc_id", "")),
            "court": meta.get("court", ""),
            "year": meta.get("year"),
            "topics": json.loads(meta.get("topics", "[]")),
            "text": docs[idx],
            "score": round(float(scores[idx]), 4),
            "source": "keyword",
        })
    return hits


def _format(results: list[dict]) -> str:
    if not results:
        return "No judgments matched those keywords."
    lines = [f"Found {len(results)} results via keyword search:\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['doc_id']}] {r['case_name'][:50]} ({r['court'][:30]}, {r['year']})"
            f" | bm25={r['score']} | {r['text'][:120]}…\n"
        )
    return "\n".join(lines)


KeywordSearchTool = Tool(
    name="keyword_search",
    func=lambda q: _format(keyword_search(q)),
    description=(
        "Exact keyword search over Indian court judgments using BM25. "
        "Use for specific legal terms: 'Section 149 Motor Vehicles Act', "
        "'contributory negligence', 'pay and recover', insurer names. "
        "Complements semantic search for precise terminology."
    ),
)
