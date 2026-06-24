import asyncio
from llm.client import chat_structured
from llm.prompts import VERIFICATION_SYSTEM
from llm.structured import VerificationResult
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from config import settings
import logging

logger = logging.getLogger("lexagent.agents.verification")

dense = DenseRetriever()
sparse = BM25Retriever()

async def verify_output(
    agent_output: str,        # the text produced by another agent
    doc_id: str,              # source document to verify against
    claims: list[str],        # specific claims to verify (extracted from output)
) -> VerificationResult:
    logger.info(f"Running verification agent on {len(claims)} claims for doc_id: {doc_id}")
    
    if not claims:
        return VerificationResult(
            is_grounded=True,
            hallucinated_claims=[],
            verified_claims=[],
            confidence_score=1.0,
            verification_summary="No claims were provided for verification."
        )

    # For each claim, retrieve supporting evidence in parallel
    verification_context_parts = []
    
    async def process_claim(claim: str):
        try:
            dense_r, sparse_r = await asyncio.gather(
                dense.search(claim, k=5, doc_id=doc_id),
                asyncio.to_thread(sparse.search, claim, k=5, doc_id=doc_id),
            )
            chunks = reciprocal_rank_fusion(dense_r, sparse_r, top_n=3)
            evidence = "\n".join([f"  - [{c.get('metadata', {}).get('page_reference', 'Unknown')}]: {c['text'][:300]}..." for c in chunks])
            return f"CLAIM: {claim}\nSOURCE EVIDENCE FOUND:\n{evidence}"
        except Exception as e:
            logger.error(f"Error retrieving context for verification claim '{claim}': {e}")
            return f"CLAIM: {claim}\nSOURCE EVIDENCE FOUND:\n  - Error retrieving evidence."

    # Process up to 10 claims in parallel to save time and API tokens
    tasks = [process_claim(claim) for claim in claims[:10]]
    results = await asyncio.gather(*tasks)
    
    verification_context = "\n\n".join(results)
    
    user_prompt = f"""Verify the following claims against the source document evidence.

AI-GENERATED OUTPUT TO VERIFY:
{agent_output}

SOURCE DOCUMENT EVIDENCE (retrieved for each claim):
{verification_context}

For each claim in the AI output: is it directly supported by the source evidence? 
Flag anything that goes beyond what the source says."""

    try:
        return await chat_structured(
            system=VERIFICATION_SYSTEM,
            user=user_prompt,
            output_schema=VerificationResult,
            model=settings.nim_reasoning_model,
        )
    except Exception as e:
        logger.error(f"Verification process failed: {e}")
        return VerificationResult(
            is_grounded=False,
            hallucinated_claims=claims,
            verified_claims=[],
            confidence_score=0.0,
            verification_summary=f"Verification failed due to error: {str(e)}"
        )
