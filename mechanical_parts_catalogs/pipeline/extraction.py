import hashlib
import os
import json
from dataclasses import dataclass
from llama_cloud import AsyncLlamaCloud
from typing import List, Any, Dict, Optional
from pipeline.schemas import PartSchema, TableRowSchema
from config.settings import LLAMA_CLOUD_API_KEY
from pathlib import Path

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------
import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Cache config
# -----------------------------------------------------------------------
CACHE_DIR = Path(".cache/extraction")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ExtractionResult:
    data: list
    field_metadata: list  # parallel to data, contains page_number per item

# ---------------------------------------------------------------------
# Agent versioning
# ---------------------------------------------------------------------
def compute_agent_fingerprint(config: dict, data_schema: dict) -> str:
    payload = json.dumps({"config": config, "schema": data_schema}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]

async def get_or_create_agent(client, name, config, data_schema):
    fingerprint = compute_agent_fingerprint(config, data_schema)
    versioned_name = f"{name}__{fingerprint}"

    existing = await client.extraction.extraction_agents.list()
    for agent in existing:
        if agent.name == versioned_name:
            logger.info(f"Reusing agent: {versioned_name}")
            return agent

    logger.info(f"Creating new agent: {versioned_name}")
    return await client.extraction.extraction_agents.create(
        config=config,
        data_schema=data_schema,
        name=versioned_name,
    )

async def cleanup_old_agents(client, base_name, keep_name):
    existing = await client.extraction.extraction_agents.list()
    for agent in existing:
        if agent.name.startswith(base_name) and agent.name != keep_name:
            logger.info(f"Deleting stale agent: {agent.name}")
            await client.extraction.extraction_agents.delete(agent.id)

# ---------------------------------------------------------------------
# Result caching
# ---------------------------------------------------------------------
def compute_file_hash(pdf_path: str) -> str:
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_extraction_cache_key(
    pdf_path: str,
    config: dict,
    data_schema: dict,
    sys_prompt: str,
    layer_name: str,
) -> str:
    payload = {
        "layer_name": layer_name,
        "pdf_hash": compute_file_hash(pdf_path),
        "config": config,
        "schema": data_schema,
        "system_prompt": sys_prompt,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def _to_jsonable_item(item: Any) -> Any:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    if isinstance(item, dict):
        return item
    return item

def save_extraction_cache(cache_key: str, result: ExtractionResult) -> None:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    payload = {
        "cache_key": cache_key,
        "data": [_to_jsonable_item(x) for x in result.data],
        "field_metadata": result.field_metadata,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_extraction_cache(cache_key: str) -> Optional[ExtractionResult]:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None

    with open(cache_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("cache_key") != cache_key:
        return None

    return ExtractionResult(
        data=payload.get("data", []),
        field_metadata=payload.get("field_metadata", []),
    )

# ---------------------------------------------------------------------
# Shared extraction runner with cache
# ---------------------------------------------------------------------

async def extract_with_cache(
    *,
    pdf_path: str,
    sys_prompt: str,
    layer_schema,
    agent_name: str,
    extraction_target: str,
) -> ExtractionResult:
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

    config = {
        "chunk_mode": "PAGE",
        "cite_sources": True,
        "confidence_scores": True,
        "extraction_target": extraction_target,
        "extraction_mode": "PREMIUM",
        "parse_model": "anthropic-sonnet-4.5",
        "system_prompt": sys_prompt,
    }
    schema = layer_schema.model_json_schema()

    cache_key = compute_extraction_cache_key(
        pdf_path=pdf_path,
        config=config,
        data_schema=schema,
        sys_prompt=sys_prompt,
        layer_name=agent_name,
    )

    cached = load_extraction_cache(cache_key)
    if cached is not None:
        logger.info(f"Cache hit for {agent_name}: {cache_key}")
        return cached

    logger.info(f"Cache miss for {agent_name}: {cache_key}")

    file_obj = await client.files.create(
        file=pdf_path,
        purpose="extract",
    )
    file_id = file_obj.id

    agent = await get_or_create_agent(
        client=client,
        name=agent_name,
        config=config,
        data_schema=schema,
    )
    await cleanup_old_agents(client, agent_name, keep_name=agent.name)

    try:
        result = await client.extraction.jobs.extract(
            extraction_agent_id=agent.id,
            file_id=file_id,
        )
    except Exception as e:
        logger.error(agent_name + f"Layer 1 extraction failed: {e}")
        raise

    extraction_result = ExtractionResult(
        data=[_to_jsonable_item(x) for x in result.data],
        field_metadata=result.extraction_metadata.get("field_metadata", []),
    )

    save_extraction_cache(cache_key, extraction_result)
    return extraction_result


# LAYER 1 — PER_PAGE (Part-level metadata)
async def extract_layer1_fields(pdf_path: str, sys_prompt: str, layer_schema) -> ExtractionResult:
    return await extract_with_cache(
        pdf_path=pdf_path,
        sys_prompt=sys_prompt,
        layer_schema=layer_schema,
        agent_name="Layer 1 Agent",
        extraction_target="PER_PAGE",
    )

# LAYER 2 — PER_TABLE_ROW (Dimension rows)
async def extract_layer2_fields(pdf_path: str, sys_prompt: str, layer_schema) -> ExtractionResult:
    return await extract_with_cache(
        pdf_path=pdf_path,
        sys_prompt=sys_prompt,
        layer_schema=layer_schema,
        agent_name="Layer 2 Agent",
        extraction_target="PER_TABLE_ROW",
    )