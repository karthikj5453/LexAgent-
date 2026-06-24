from openai import AsyncOpenAI
from typing import Type, TypeVar
from pydantic import BaseModel
import json
import logging
from config import settings

logger = logging.getLogger("lexagent.llm")

T = TypeVar("T", bound=BaseModel)

# Initialize the client. We check settings.nim_api_key inside calls to allow lazy configuration
client = None

def get_client() -> AsyncOpenAI:
    global client
    if client is None:
        api_key = settings.nim_api_key
        if not api_key:
            # Fall back to environment variable directly
            import os
            api_key = os.environ.get("NIM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy_key"
            if api_key == "dummy_key":
                logger.warning("NIM_API_KEY is not set. API calls will fail until configured.")
        client = AsyncOpenAI(
            base_url=settings.nim_base_url,
            api_key=api_key,
        )
    return client

async def chat(
    system: str,
    user: str,
    temperature: float = 0.1,   # low temp for legal tasks — determinism matters
    max_tokens: int = 2048,
) -> str:
    openai_client = get_client()
    try:
        response = await openai_client.chat.completions.create(
            model=settings.nim_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling NIM chat endpoint: {e}")
        raise e

async def chat_structured(
    system: str,
    user: str,
    output_schema: Type[T],
    temperature: float = 0.1,
) -> T:
    """Forces LLM to return valid JSON matching the schema with robust extraction."""
    schema_str = json.dumps(output_schema.model_json_schema(), indent=2)
    augmented_system = f"""{system}

CRITICAL: Respond ONLY with valid JSON matching this exact schema. Do not include markdown formatting, preambles, explanation, or code fences (like ```json).
Schema:
{schema_str}"""
    
    raw = await chat(augmented_system, user, temperature)
    
    # Robust cleaning and JSON boundary extraction
    cleaned = raw.strip()
    try:
        # Strip markdown fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        
        # Strip potential language identifier "json" from the beginning
        if cleaned.lower().startswith("json\n"):
            cleaned = cleaned[5:].strip()
            
        # Extract the JSON object substring between the first '{' and last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned[start_idx:end_idx+1]
        else:
            json_str = cleaned
            
        return output_schema.model_validate_json(json_str)
    except Exception as e:
        logger.error(f"Failed to parse structured output. Raw response:\n{raw}\nCleaned response:\n{cleaned}")
        logger.error(f"Parsing error: {e}")
        raise e
