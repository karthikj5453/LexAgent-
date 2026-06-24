from config import settings

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = None,
    top_n: int = None,
) -> list[dict]:
    """
    Fuses dense and sparse search results using Reciprocal Rank Fusion (RRF).
    """
    k = k or settings.rrf_k
    top_n = top_n or settings.top_k_final
    
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    # Accumulate scores from dense retriever
    for rank, result in enumerate(dense_results):
        cid = result["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (rank + k)
        chunk_map[cid] = result

    # Accumulate scores from sparse retriever
    for rank, result in enumerate(sparse_results):
        cid = result["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (rank + k)
        if cid not in chunk_map:
            chunk_map[cid] = result

    # Sort candidates by combined RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {**chunk_map[cid], "rrf_score": round(score, 6), "rrf_rank": i}
        for i, (cid, score) in enumerate(ranked[:top_n])
    ]
