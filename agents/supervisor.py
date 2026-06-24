from llm.client import chat_structured, chat
from llm.prompts import SUPERVISOR_SYSTEM
from llm.structured import SupervisorIntent
from agents.contract_review import run_contract_review
from agents.compliance import run_compliance_check
from agents.drafting import run_drafting
from agents.verification import verify_output
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
from config import settings
import asyncio
import logging

logger = logging.getLogger("lexagent.agents.supervisor")

dense = DenseRetriever()
sparse = BM25Retriever()

async def check_safety(text: str) -> bool:
    """
    Verify input safety using Llama-Guard-3 NIM.
    """
    try:
        response = await chat(
            system="""[INST] Task: Check if there is unsafe content in the user message in the context of legal RAG assistance.
Provide your safety assessment for User in the following format:
- The first line must be either "safe" or "unsafe".
- If unsafe, the second line must contain a comma-separated list of violated categories. [/INST]""",
            user=text,
            model=settings.nim_guardrail_model,
        )
        cleaned = response.strip().lower()
        if "unsafe" in cleaned:
            logger.warning(f"Llama-Guard flagged unsafe content: {cleaned}")
            return False
        return True
    except Exception as e:
        logger.error(f"Llama-Guard check failed: {e}. Defaulting to safe.")
        return True

async def handle_request(
    user_message: str,
    doc_id: str | None,
    session_history: list[dict],
) -> dict:
    logger.info(f"Supervisor handling request: '{user_message}' (doc_id: {doc_id})")
    
    # Step 0: Run safety guardrail check
    is_safe = await check_safety(user_message)
    if not is_safe:
        return {
            "response": "⚠️ **Safety Flagged**: The input message was flagged by our safety guardrails as unsafe or violating policy. Request blocked.",
            "intent": {
                "intent": "unknown",
                "sub_tasks": [],
                "requires_document": False
            },
            "agent_results": {
                "verification": {
                    "is_grounded": False,
                    "hallucinated_claims": [],
                    "verified_claims": [],
                    "confidence_score": 0.0,
                    "verification_summary": "Blocked by Safety Guardrails."
                }
            },
            "verification_passed": False,
        }
        
    # Step 1: classify intent
    intent_prompt = f"""User message: {user_message}
Document attached: {"yes, doc_id=" + doc_id if doc_id else "no"}
Conversation history: {len(session_history)} previous messages

Classify this request and list the sub-tasks needed."""

    try:
        intent = await chat_structured(
            system=SUPERVISOR_SYSTEM,
            user=intent_prompt,
            output_schema=SupervisorIntent,
            model=settings.nim_fast_model,
        )
        logger.info(f"Classified intent: {intent.intent}, sub_tasks: {intent.sub_tasks}")
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        # Default fallback intent
        intent = SupervisorIntent(
            intent="unknown",
            sub_tasks=["review"] if doc_id else ["query"],
            requires_document=True if doc_id else False
        )
    
    results = {}
    
    # Step 2: dispatch to specialist agents
    tasks = {}
    
    if "review" in intent.sub_tasks and doc_id:
        tasks["review"] = run_contract_review(doc_id)
    
    if "compliance" in intent.sub_tasks and doc_id:
        tasks["compliance"] = run_compliance_check(doc_id)
    
    if "draft" in intent.sub_tasks:
        tasks["draft"] = run_drafting(user_message, doc_id)
        
    # If it is a generic query and a document is uploaded, retrieve context to answer directly
    if ("query" in intent.sub_tasks or not tasks) and doc_id:
        async def run_query_rag(q: str, d_id: str) -> str:
            dense_r, sparse_r = await asyncio.gather(
                dense.search(q, k=8, doc_id=d_id),
                asyncio.to_thread(sparse.search, q, k=8, doc_id=d_id),
            )
            chunks = reciprocal_rank_fusion(dense_r, sparse_r, top_n=5)
            context = "\n\n---\n\n".join([f"[Chunk {i+1} - {c.get('metadata', {}).get('page_reference', 'Unknown Page')}]\n{c['text']}" for i, c in enumerate(chunks)])
            
            qa_prompt = f"""Answer the user's question based ONLY on the provided contract excerpts. 
Cite exact page references (e.g. Page X) where the information was found. 
If the information is not present, state that it is not found.

Question: {q}

CONTRACT EXCERPTS:
{context}"""
            return await chat(
                system="You are LexAgent, an expert assistant answering questions about legal documents.",
                user=qa_prompt
            )
        tasks["query"] = run_query_rag(user_message, doc_id)

    # Await all dispatched tasks in parallel
    if tasks:
        awaited_keys = list(tasks.keys())
        awaited_vals = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for key, val in zip(awaited_keys, awaited_vals):
            if isinstance(val, Exception):
                logger.error(f"Agent '{key}' failed with exception: {val}")
                results[key] = f"Agent execution failed: {str(val)}"
            else:
                results[key] = val
    
    # Step 3: always verify if there's a document and results were generated
    verification_passed = True
    if doc_id and results:
        combined_output = ""
        for k, v in results.items():
            if isinstance(v, str):
                combined_output += f"\nAgent {k}: {v}"
            else:
                # Convert Pydantic object to json/dict dump
                combined_output += f"\nAgent {k}: {v.model_dump_json() if hasattr(v, 'model_dump_json') else str(v)}"
                
        # Extract specific factual claims for verification
        try:
            claims = await _extract_claims(combined_output)
            verification_res = await verify_output(
                agent_output=combined_output,
                doc_id=doc_id,
                claims=claims,
            )
            results["verification"] = verification_res
            verification_passed = verification_res.is_grounded
        except Exception as e:
            logger.error(f"Factual verification pipeline failed: {e}")
    
    # Step 4: assemble final response
    try:
        final_response = await _assemble_response(
            user_message, intent, results, session_history
        )
    except Exception as e:
        logger.error(f"Response assembly failed: {e}")
        final_response = f"An error occurred while assembling response: {str(e)}"
    
    # Clean up results serialization for return
    serialized_results = {}
    for k, v in results.items():
        if hasattr(v, "model_dump"):
            serialized_results[k] = v.model_dump()
        else:
            serialized_results[k] = v

    return {
        "response": final_response,
        "intent": intent.model_dump(),
        "agent_results": serialized_results,
        "verification_passed": verification_passed,
    }

async def _extract_claims(text: str) -> list[str]:
    """Extract verifiable factual claims from agent output."""
    raw = await chat(
        system="Extract specific, verifiable factual claims from this legal analysis. Return one claim per line. Claims must be specific facts, not general statements.",
        user=text[:3000],
        model=settings.nim_fast_model,
    )
    return [line.strip() for line in raw.split("\n") if line.strip()][:10]

async def _assemble_response(user_msg: str, intent: SupervisorIntent, results: dict, history: list) -> str:
    context_parts = []
    verification_summary = ""
    
    for k, v in results.items():
        if k == "verification":
            verification_summary = f"\n\n**Grounding Verification Report:**\nGrounding Confidence: {int(v.confidence_score * 100)}%\nStatus: {'VERIFIED' if v.is_grounded else 'WARNING (Potential Hallucination Detected)'}\nSummary: {v.verification_summary}"
            if v.hallucinated_claims:
                verification_summary += "\nUnverified claims:\n" + "\n".join([f"- {claim}" for claim in v.hallucinated_claims])
            continue
            
        val_str = v.model_dump_json() if hasattr(v, "model_dump_json") else str(v)
        context_parts.append(f"Agent: {k}\nResult: {val_str[:2000]}")
        
    context = "\n\n".join(context_parts)
    
    history_context = ""
    if history:
        history_context = "Conversation history:\n" + "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history[-4:]]) + "\n\n"
        
    prompt = f"""{history_context}User asked: {user_msg}
 
Synthesize the specialist agent results into a professional, clear response for the user.
Explain the findings, cite specific clauses/page references, and list action items.
 
Specialist agent results:
{context}"""

    response = await chat(
        system="You are LexAgent, a helpful, precise AI legal assistant. Do not add metadata, JSON, or code formatting to your response. Provide a beautifully formatted professional legal advice response in markdown.",
        user=prompt,
        model=settings.nim_reasoning_model,
    )
    
    # Append the verification report to the end of the user response
    return response + verification_summary
