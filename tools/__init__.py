from tools.vector_search import VectorSearchTool, vector_search
from tools.keyword_search import KeywordSearchTool, keyword_search
from tools.metadata_filter import MetadataFilterTool

ALL_TOOLS = [VectorSearchTool, KeywordSearchTool, MetadataFilterTool]


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Merge vector and keyword results using Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1/(k + rank) across result lists, where k=60 smooths
    the contribution of lower-ranked results. No score normalization needed.
    """
    K = 60

    vec_results = vector_search(query, top_k=top_k)
    kw_results = keyword_search(query, top_k=top_k)

    rrf: dict[str, float] = {}
    by_chunk: dict[str, dict] = {}

    for rank, r in enumerate(vec_results):
        cid = r["chunk_id"]
        rrf[cid] = rrf.get(cid, 0) + 1 / (K + rank + 1)
        by_chunk[cid] = r

    for rank, r in enumerate(kw_results):
        cid = r["chunk_id"]
        rrf[cid] = rrf.get(cid, 0) + 1 / (K + rank + 1)
        by_chunk[cid] = r

    merged = sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k]
    return [{**by_chunk[cid], "rrf_score": round(rrf[cid], 6)} for cid in merged]
