import logging
from typing import Optional
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.indices.property_graph import (
    PropertyGraphIndex,
    VectorContextRetriever,
    LLMSynonymRetriever,
)
from retrieval.intent import QueryIntent, extract_query_intent
from retrieval.filters import build_filtered_retriever

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# vector+hybrid retrievers
# -----------------------------------------------------------------------

def build_basic_vector_retriever(
    basic_index: VectorStoreIndex,
    similarity_top_k: int = 8,
):
    logger.info("Building basic vector retriever")
    return basic_index.as_retriever(
        similarity_top_k=similarity_top_k,
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
    )

def build_basic_hybrid_retriever(
    hybrid_index: VectorStoreIndex,
    similarity_top_k: int = 8,
    sparse_top_k: int = 8,
):
    logger.info("Building basic hybrid retriever")
    return hybrid_index.as_retriever(
        similarity_top_k=similarity_top_k,
        sparse_top_k=sparse_top_k,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    )

# -----------------------------------------------------------------------
# kg retriever
# -----------------------------------------------------------------------

def build_kg_retriever(
    property_graph_index: PropertyGraphIndex,
    similarity_top_k: int = 8,
    path_depth: int = 3,
    include_text: bool = True,
):
    logger.info("Building KG retriever")
    vector_retriever = VectorContextRetriever(
        graph_store=property_graph_index.property_graph_store,
        embed_model=Settings.embed_model,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
        include_text=include_text,
    )

    logger.info("Expanding query keywords via LLM synonym retriever → traversing graph")
    synonym_retriever = LLMSynonymRetriever(
        graph_store=property_graph_index.property_graph_store,
        llm=Settings.llm,
        path_depth=path_depth,
        include_text=include_text,
    )

    return property_graph_index.as_retriever(
        sub_retrievers=[vector_retriever, synonym_retriever],
        include_text=include_text,
    )

# -----------------------------------------------------------------------
# custom retriever
# -----------------------------------------------------------------------

"""Custom retriever that performs both KG and hybrid search with metadata filtering"""
def build_custom_retriever(
    query: str,
    basic_index: VectorStoreIndex,
    hybrid_index: VectorStoreIndex,
    kg_retriever,
    intent: Optional[QueryIntent] = None,
) -> QueryFusionRetriever:
    if intent is None:
        intent = extract_query_intent(query, Settings.llm)
    metadata_retriever = build_filtered_retriever(
        intent=intent,
        basic_index=basic_index,
        hybrid_index=hybrid_index,
        use_hybrid=True,
    )
    logger.info("Building custom QueryFusionRetriever (hybrid + KG)")
    return QueryFusionRetriever(
        [metadata_retriever, kg_retriever],
        similarity_top_k=10,
        num_queries=3,
    )