# For offical documentation, visit https://developers.llamaindex.ai/python/examples/agent/agent_workflow_basic/

import logging
from llama_index.core import Settings
# Deprecated (https://developers.llamaindex.ai/python/framework/changes/deprecated_terms/)
# from llama_index.core.agent.function_calling.step import FunctionCallingAgentWorker
# from llama_index.core.agent import AgentRunner
# newer version
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.property_graph import PropertyGraphIndex

from retrieval.intent import QueryIntent, extract_query_intent
from retrieval.filters import build_filtered_retriever
from retrieval.retriever import (
    build_basic_vector_retriever,
    build_basic_hybrid_retriever,
    build_kg_retriever,
)

logger = logging.getLogger(__name__)

def build_agent_retriever(
    basic_index: VectorStoreIndex,
    hybrid_index: VectorStoreIndex,
    property_graph_index: PropertyGraphIndex,
    similarity_top_k: int = 10,
):
    # ── pre-build the retrievers once ─────────────────────────────
    vector_ret  = build_basic_vector_retriever(basic_index, similarity_top_k)
    hybrid_ret  = build_basic_hybrid_retriever(hybrid_index, similarity_top_k)
    # KG is optional
    kg_ret = None
    if property_graph_index is not None:
        kg_ret = build_kg_retriever(property_graph_index)

    # ── tool definitions ──────────────────────────────────────────
    def vector_search(query: str) -> str:
        """
        Use for broad semantic questions about gear families, materials,
        or general catalog content. Best for open-ended natural language queries.
        """
        logger.info(f"[Agent] vector_search called: '{query}'")
        nodes = vector_ret.retrieve(query)
        return "\n\n".join(n.node.text for n in nodes)

    def hybrid_search(query: str) -> str:
        """
        Use when the query contains specific keywords, part names, or
        article numbers alongside natural language. Combines semantic
        and keyword search.
        """
        logger.info(f"[Agent] hybrid_search called: '{query}'")
        nodes = hybrid_ret.retrieve(query)
        return "\n\n".join(n.node.text for n in nodes)

    def filtered_search(query: str) -> str:
        """
        Use when the query contains numeric constraints (e.g. torque > 200 Ncm,
        weight < 50g, teeth count >= 20) or exact filters (module, material,
        family). Extracts intent and applies structured metadata filters.
        """
        logger.info(f"[Agent] filtered_search called: '{query}'")
        intent = extract_query_intent(query, Settings.llm)
        retriever = build_filtered_retriever(
            intent=intent,
            basic_index=basic_index,
            hybrid_index=hybrid_index,
            use_hybrid=True,
        )
        nodes = retriever.retrieve(query)
        return "\n\n".join(n.node.text for n in nodes)

    def kg_search(query: str) -> str:
        """
        Use for relationship-based or multi-hop queries that require traversing
        connections between gears, materials, modules, and specifications.
        Best for queries like 'which steel gears have module 1.0'.
        """
        if kg_ret is None:
            return "KG search is unavailable because no property graph index was built."

        logger.info(f"[Agent] kg_search called: '{query}'")
        nodes = kg_ret.retrieve(query)
        return "\n\n".join(n.node.text for n in nodes)

    # ── wrap as LlamaIndex tools ──────────────────────────────────
    tools = [
        FunctionTool.from_defaults(
            fn=vector_search,
            name="vector_search",
            description=(
                "Use for broad semantic questions about gear families, materials, "
                "or general catalog content."
            ),
        ),
        FunctionTool.from_defaults(
            fn=hybrid_search,
            name="hybrid_search",
            description=(
                "Use for keyword-heavy queries with part names, article numbers, "
                "or mixed keyword + semantic search."
            ),
        ),
        FunctionTool.from_defaults(
            fn=filtered_search,
            name="filtered_search",
            description=(
                "Use for numeric constraints or exact filters such as torque, "
                "weight, teeth count, module, family, or material."
            ),
        ),
    ]

    # KG tool
    kg_tool = FunctionTool.from_defaults(
            fn=kg_search,
            name="kg_search",
            description=(
                "Use for relationship-based or multi-hop queries across gears, "
                "materials, modules, and specifications."
            ),
        )
    
    if kg_ret is not None:
        tools.append(kg_tool)

    # ── build agent ───────────────────────────────────────────────
    agent = FunctionAgent(
        tools=tools,
        llm=Settings.llm,
        verbose=True,
        system_prompt=(
            "You are an expert assistant for a mechanical gear catalog. "
            "You have access to four search tools. "
            "Choose the most appropriate tool(s) based on the query type. "
            "For simple semantic questions use vector_search. "
            "For keyword-heavy queries use hybrid_search. "
            "For numeric constraints or exact filters use filtered_search. "
            "For relationship or multi-hop queries use kg_search. "
            "Combine tools only when necessary to avoid extra cost. "
            "Always ground your answer strictly in the retrieved catalog data."
        ),
        # optional if model chokes on streaming:
        # streaming=False,
    )

    logger.info("Agent retriever built with 4 tools")
    return agent