import hashlib
import json
from dataclasses import dataclass
from llama_cloud import AsyncLlamaCloud
from typing import List, Any, Dict, Optional
from pipeline.schemas import PartSchema, TableRowSchema

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------
import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os
load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMAINDEX_CLOUD_KEY")


# -----------------------------------------------------------------------
# look up the existing agent instead of creating a new one, and reuse it
# -----------------------------------------------------------------------

def compute_agent_fingerprint(config: dict, data_schema: dict) -> str:
    """Create a short hash from config + schema so any change produces a new agent."""
    payload = json.dumps({"config": config, "schema": data_schema}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]  # 12 chars is plenty

async def get_or_create_agent(client, name, config, data_schema):
    fingerprint = compute_agent_fingerprint(config, data_schema)
    versioned_name = f"{name}__{fingerprint}"

    # check if this exact version already exists
    existing = await client.extraction.extraction_agents.list()
    for agent in existing:
        if agent.name == versioned_name:
            logger.info(f"Reusing agent: {versioned_name}")
            return agent

    # config or schema changed — create a fresh agent
    logger.info(f"Creating new agent: {versioned_name}")
    return await client.extraction.extraction_agents.create(
        config=config,
        data_schema=data_schema,
        name=versioned_name,
    )

async def cleanup_old_agents(client, base_name, keep_name):
    """Delete old versioned agents that are no longer current."""
    existing = await client.extraction.extraction_agents.list()
    for agent in existing:
        if agent.name.startswith(base_name) and agent.name != keep_name:
            logger.info(f"Deleting stale agent: {agent.name}")
            await client.extraction.extraction_agents.delete(agent.id)

# -----------------------------------------------------------------------
# Define extraction approach
# -----------------------------------------------------------------------

@dataclass
class ExtractionResult:
    data: list
    field_metadata: list  # parallel to data, contains page_number per item

# LAYER 1 — PER_PAGE (Part-level metadata)
async def extract_layer1_fields(pdf_path: str, sys_prompt: str, layer_schema) -> ExtractionResult:
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

    file_obj = await client.files.create(
        file=pdf_path,
        purpose="extract",
    )
    file_id = file_obj.id

    agent = await get_or_create_agent(
        client=client,
        name="Layer 1 Agent",
        config={
            "chunk_mode": "PAGE",
            "cite_sources": True,
            "confidence_scores": True,
            "extraction_target": "PER_PAGE",
            "extraction_mode": "PREMIUM",
            "parse_model": "anthropic-sonnet-4.5",
            "system_prompt": sys_prompt,
        },
        data_schema=layer_schema.model_json_schema(),
    )
    await cleanup_old_agents(client, "Layer 1 Agent", keep_name=agent.name)

    # error handling around LlamaCloud API calls
    try:
        result_layer1 = await client.extraction.jobs.extract(
            extraction_agent_id=agent.id,
            file_id=file_id,
        )
    except Exception as e:
        logger.error(f"Layer 1 extraction failed: {e}")
        raise

    # part_data       -> ExtractionResult
    logger.info("Starting Layer 1 extraction (PER_PAGE)")
    logger.debug(f"Layer 1 extraction result: {result_layer1.data}")

    return ExtractionResult(
        data=result_layer1.data,
        field_metadata=result_layer1.extraction_metadata.get("field_metadata", []),
    )

# LAYER 2 — PER_TABLE_ROW (Dimension rows)
async def extract_layer2_fields(pdf_path: str, sys_prompt: str, layer_schema) -> ExtractionResult:
    client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

    file_obj = await client.files.create(
        file=pdf_path,
        purpose="extract",
    )
    file_id = file_obj.id

    agent = await get_or_create_agent(
        client=client,
        name="Layer 2 Agent",
        config={
            "chunk_mode": "PAGE",
            "cite_sources": True,
            "confidence_scores": True,
            "extraction_target": "PER_TABLE_ROW",
            "extraction_mode": "PREMIUM",
            "parse_model": "anthropic-sonnet-4.5",
            "system_prompt": sys_prompt,
        },
        data_schema=layer_schema.model_json_schema(),
    )
    await cleanup_old_agents(client, "Layer 2 Agent", keep_name=agent.name)

    # error handling around LlamaCloud API calls
    try:
        result_layer2 = await client.extraction.jobs.extract(
            extraction_agent_id=agent.id,
            file_id=file_id,
        )
    except Exception as e:
        logger.error(f"Layer 2 extraction failed: {e}")
        raise

    # table_row_data  -> ExtractionResult
    logger.info("Starting Layer 2 extraction (PER_TABLE_ROW)")
    logger.debug(f"Layer 2 extraction result: {result_layer2.data}")

    return ExtractionResult(
        data=result_layer2.data,
        field_metadata=result_layer2.extraction_metadata.get("field_metadata", []),
    )