"""Vector similarity search over the judgment corpus."""

import json
from langchain_core.tools import Tool
from db import get_collection, get_embedder, TOP_K


def vector_search(query: str, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    embedding = get_embedder().encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "doc_id": meta.get("doc_id", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "case_name": meta.get("case_name", meta.get("doc_id", "")),
            "court": meta.get("court", ""),
            "year": meta.get("year"),
            "topics": json.loads(meta.get("topics", "[]")),
            "text": doc,
            "score": round(1 - dist, 4),
            "source": "vector",
        })
    return hits


def _format(results: list[dict]) -> str:
    if not results:
        return "No relevant judgments found."
    lines = [f"Found {len(results)} results via semantic search:\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r['doc_id']}] {r['case_name']} ({r['court']}, {r['year']})\n"
            f"   Score: {r['score']} | Topics: {', '.join(r['topics']) or 'general'}\n"
            f"   {r['text'][:300]}...\n"
        )
    return "\n".join(lines)


VectorSearchTool = Tool(
    name="vector_search",
    func=lambda q: _format(vector_search(q)),
    description=(
        "Semantic search over Indian court judgments. Use for conceptual queries "
        "like 'insurance liability for unlicensed driver' or 'compensation for death claim'. "
        "Returns the most relevant judgment excerpts with scores."
    ),
)
