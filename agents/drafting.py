import asyncio
from llm.client import chat_structured
from llm.prompts import DRAFTING_SYSTEM
from llm.structured import DraftingResult
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from config import settings
import logging

logger = logging.getLogger("lexagent.agents.drafting")

dense = DenseRetriever()
sparse = BM25Retriever()

async def run_drafting(user_instruction: str, doc_id: str | None = None) -> DraftingResult:
    logger.info(f"Running drafting agent for instruction: {user_instruction}")
    
    context = ""
    if doc_id:
        try:
            # Retrieve relevant parts of the contract to base the redraft on
            dense_r, sparse_r = await asyncio.gather(
                dense.search(user_instruction, k=5, doc_id=doc_id),
                asyncio.to_thread(sparse.search, user_instruction, k=5, doc_id=doc_id),
            )
            chunks = reciprocal_rank_fusion(dense_r, sparse_r, top_n=3)
            
            if chunks:
                context_parts = []
                for i, c in enumerate(chunks):
                    page_info = c.get("metadata", {}).get("page_reference", "Unknown Page")
                    context_parts.append(f"[Reference text - {page_info}]\n{c['text']}")
                context = "\n\n---\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Drafting retrieval failed: {e}")
            
    user_prompt = f"Instruction: {user_instruction}\n\n"
    if context:
        user_prompt += f"SOURCE CONTRACT CLAUDE CONTEXT:\n{context}\n\n"
    user_prompt += "Please draft the clause as requested, adhering to standard Indian commercial practices, and explain key terms and business implications."

    try:
        return await chat_structured(
            system=DRAFTING_SYSTEM,
            user=user_prompt,
            output_schema=DraftingResult,
            model=settings.nim_reasoning_model,
        )
    except Exception as e:
        logger.error(f"Drafting failed: {e}")
        return DraftingResult(
            drafted_clause_type="Requested Clause",
            drafted_text=f"Drafting failed due to error: {str(e)}",
            key_terms_explained=[],
            commercial_implications="None"
        )
