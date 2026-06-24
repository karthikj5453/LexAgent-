import asyncio
from llm.client import chat_structured
from llm.prompts import CONTRACT_REVIEW_SYSTEM
from llm.structured import ContractReviewResult
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from config import settings
import logging

logger = logging.getLogger("lexagent.agents.contract_review")

dense = DenseRetriever()
sparse = BM25Retriever()

REVIEW_QUERIES = [
    "termination clause conditions",
    "indemnification and liability",
    "intellectual property ownership assignment",
    "limitation of liability cap",
    "governing law jurisdiction dispute resolution",
    "confidentiality non-disclosure obligations",
    "payment terms conditions",
    "force majeure events",
    "representations warranties",
    "amendment modification procedure",
]

async def _retrieve(query: str, doc_id: str) -> list[dict]:
    try:
        dense_r, sparse_r = await asyncio.gather(
            dense.search(query, k=10, doc_id=doc_id),
            # BM25 is synchronous, but we wrap it/run it alongside dense search
            asyncio.to_thread(sparse.search, query, k=10, doc_id=doc_id),
        )
        return reciprocal_rank_fusion(dense_r, sparse_r, top_n=5)
    except Exception as e:
        logger.error(f"Retrieval error for query '{query}': {e}")
        return []

async def run_contract_review(doc_id: str) -> ContractReviewResult:
    logger.info(f"Running contract review for doc_id: {doc_id}")
    
    # Run all standard queries in parallel
    tasks = [_retrieve(query, doc_id) for query in REVIEW_QUERIES]
    results = await asyncio.gather(*tasks)
    
    # Deduplicate by chunk id
    all_chunks = []
    seen = set()
    for batch in results:
        for chunk in batch:
            if chunk["id"] not in seen:
                all_chunks.append(chunk)
                seen.add(chunk["id"])
    
    if not all_chunks:
        logger.warning(f"No chunks retrieved for doc_id: {doc_id}. Running with empty context.")
        context = "[No contract excerpts retrieved.]"
    else:
        # Build context from top 20 unique chunks
        context_parts = []
        for i, c in enumerate(all_chunks[:20]):
            page_info = c.get("metadata", {}).get("page_reference", "Unknown Page")
            context_parts.append(f"[Chunk {i+1} - {page_info}]\n{c['text']}")
        context = "\n\n---\n\n".join(context_parts)
    
    user_prompt = f"""Analyze the following contract excerpts and provide a complete review.

CONTRACT EXCERPTS:
{context}

Identify all significant clauses, assess risks, reference the exact pages/sections, and provide actionable recommendations."""

    try:
        return await chat_structured(
            system=CONTRACT_REVIEW_SYSTEM,
            user=user_prompt,
            output_schema=ContractReviewResult,
            model=settings.nim_reasoning_model,
        )
    except Exception as e:
        logger.error(f"Contract review generation failed: {e}")
        # Return fallback default response to keep system running
        from llm.structured import RiskLevel
        return ContractReviewResult(
            document_type="Unknown",
            parties=[],
            effective_date=None,
            clauses=[],
            overall_risk=RiskLevel.MEDIUM,
            executive_summary=f"Review failed due to an error: {str(e)}"
        )
