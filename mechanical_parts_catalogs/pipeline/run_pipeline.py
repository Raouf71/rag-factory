import asyncio
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.openai import OpenAIEmbedding

from pipeline.schemas import PartSchema, TableRowSchema
from pipeline.extraction import extract_layer1_fields, extract_layer2_fields
from pipeline.mapping import (
    attach_part_id_to_layer1_parts,
    attach_part_id_to_layer2_rows,
    map_parts_and_rows,
    log_mapping_diagnostics,
)
from pipeline.nodes import build_retrieval_nodes
from pipeline.indexing import build_pgvector_store, index_nodes_with_store
from pipeline.graph import get_or_build_property_graph_index
from retrieval.retriever import (
    build_basic_vector_retriever,
    build_basic_hybrid_retriever,
    build_kg_retriever,
    build_custom_retriever,
)
from retrieval.agent import build_agent_retriever

from config.settings import (
    PDF_PATH,
    EMBED_DIM,
    DEEPSEEK_KEY,
    OPENAI_KEY,
    SYSTEM_PROMPT_L1,
    SYSTEM_PROMPT_L2,
    CATALOG_QA_PROMPT,
)

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Model setup — done once here, not scattered across modules
# -----------------------------------------------------------------------
def setup_models():

    gpt_4o_mini_model=OpenAI(
        model="gpt-4o-mini", 
        api_key=OPENAI_KEY,
        temperature=0.3
    )
    deepseek_model=DeepSeek(
        model="deepseek-chat", 
        api_key=DEEPSEEK_KEY,
        temperature=0.1)

    embedding_3_small_model=OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_KEY,
    )

    # --- Setup models

    Settings.llm = gpt_4o_mini_model
    # Settings.llm = deepseek_model
    Settings.embed_model = embedding_3_small_model

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
async def run_pipeline(debug: bool = False):
    setup_models()

    # 1. Extract
    logger.info("Starting extraction...")
    extraction_result_layer1 = await extract_layer1_fields(PDF_PATH, SYSTEM_PROMPT_L1, PartSchema)
    extraction_result_layer2 = await extract_layer2_fields(PDF_PATH, SYSTEM_PROMPT_L2, TableRowSchema)

    # 2. Map layers
    logger.info("Mapping layers...")
    parts_with_ids = attach_part_id_to_layer1_parts(extraction_result_layer1)
    rows_with_ids  = attach_part_id_to_layer2_rows(extraction_result_layer2)
    mapped_parts   = map_parts_and_rows(parts_with_ids, rows_with_ids)
    log_mapping_diagnostics(mapped_parts)

    # 3. Build retrieval nodes
    logger.info("Building retrieval nodes...")
    part_nodes, row_nodes = build_retrieval_nodes(mapped_parts.parts)
    all_nodes = part_nodes + row_nodes

    if debug:
        logger.debug(f"Parent nodes: {len(part_nodes)}")
        logger.debug(f"Row nodes:    {len(row_nodes)}")
        if part_nodes:
            logger.debug(f"Sample part node text:\n{part_nodes[0].text}")
            logger.debug(f"Sample part node metadata: {part_nodes[0].metadata}")
        if row_nodes:
            logger.debug(f"Sample row node text:\n{row_nodes[0].text}")
            logger.debug(f"Sample row node metadata: {row_nodes[0].metadata}")

    # 4. Index
    logger.info("Indexing into pgvector...")
    basic_vector_store  = build_pgvector_store("basic")
    hybrid_vector_store = build_pgvector_store("hybrid")
    basic_index  = index_nodes_with_store(all_nodes, basic_vector_store)
    hybrid_index = index_nodes_with_store(all_nodes, hybrid_vector_store)

    # 5. Build KG
    logger.info("Building knowledge graph...")
    property_graph_index = get_or_build_property_graph_index(all_nodes=all_nodes)

    # 6. Build retrievers
    logger.info("Building retrievers...")
    vector_retriever = build_basic_vector_retriever(basic_index=basic_index)
    hybrid_retriever = build_basic_hybrid_retriever(hybrid_index=hybrid_index)
    kg_retriever     = build_kg_retriever(
        property_graph_index=property_graph_index,
        similarity_top_k=8,
        path_depth=3,
        include_text=True,
    )

    # 7. Agent implementation
    agent = build_agent_retriever(
        basic_index=basic_index,
        hybrid_index=hybrid_index,
        property_graph_index=property_graph_index,
    )

    return dict(
        basic_index=basic_index,
        hybrid_index=hybrid_index,
        property_graph_index=property_graph_index,
        vector_retriever=vector_retriever,
        hybrid_retriever=hybrid_retriever,
        kg_retriever=kg_retriever,
        agent=agent,
    )

# -----------------------------------------------------------------------
# Query helper — used by chatbot and CLI
# -----------------------------------------------------------------------
def run_query(query: str, pipeline: dict, debug: bool = False) -> str:
    custom_retriever = build_custom_retriever(
        query=query,
        basic_index=pipeline["basic_index"],
        hybrid_index=pipeline["hybrid_index"],
        kg_retriever=pipeline["kg_retriever"],
    )

    if debug:
        nodes = custom_retriever.retrieve(query)
        logger.debug(f"Retrieved {len(nodes)} nodes")
        for i, nws in enumerate(nodes, start=1):
            logger.debug(f"[{i}] score={nws.score} | node_id={nws.node.node_id} | metadata={nws.node.metadata}")

    query_engine = RetrieverQueryEngine.from_args(
        retriever=custom_retriever,
        text_qa_template=CATALOG_QA_PROMPT,
    )
    response = query_engine.query(query)
    logger.info(f"Query: '{query}' → response generated")
    return str(response)

def run_agent_query(query: str, pipeline: dict) -> str:
    agent = pipeline["agent"]
    response = agent.chat(query)
    logger.info(f"Agent query: '{query}' → response generated")
    return str(response)

# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = asyncio.run(run_pipeline(debug=False))
    response = run_query("POM spur gear with module 2.0", pipeline)
    print(response)