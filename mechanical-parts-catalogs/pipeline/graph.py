import logging
import os
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv

from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.indices.property_graph import (
    PropertyGraphIndex,
    SchemaLLMPathExtractor,
    VectorContextRetriever,
    LLMSynonymRetriever,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from config.settings import KG_PERSIST_DIR

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger(__name__)

NEO4J_PASSWORD = os.getenv("NEO4J_CLOUD_PASSWORD")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

# -----------------------------------------------------------------------
# Init neo4j property graph store
# -----------------------------------------------------------------------

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password=NEO4J_PASSWORD,
    url="bolt://localhost:7687",
    database="neo4j",
)

# -----------------------------------------------------------------------
# define graph schemas
# -----------------------------------------------------------------------

from typing import Literal

EntityType = Literal[
    "GEAR", "MATERIAL", "MODULE", "FAMILY", "ARTICLE"
]


RelationType = Literal[
    "HAS_MATERIAL", "HAS_MODULE", "HAS_FAMILY", "HAS_ARTICLE_NR",
    "HAS_TORQUE", "HAS_TEETH_COUNT", "HAS_WEIGHT",
    "HAS_PITCH_CIRCLE_DIAMETER", "HAS_TIP_DIAMETER", "HAS_HUB_DIAMETER",
    "HAS_WIDTH", "HAS_LENGTH", "HAS_INNER_DIAMETER",
    "HAS_ANGLE_OF_ENGAGEMENT", "IS_STRAIGHT_TOOTHED",
    # "HAS_VARIANT",
]

SCHEMA_VALIDATION_SCHEMA = {
    "GEAR": [
        "HAS_MATERIAL", "HAS_MODULE", "HAS_FAMILY", "HAS_ARTICLE_NR",
        "HAS_TORQUE", "HAS_TEETH_COUNT", "HAS_WEIGHT",
        "HAS_PITCH_CIRCLE_DIAMETER", "HAS_TIP_DIAMETER", "HAS_HUB_DIAMETER",
        "HAS_WIDTH", "HAS_LENGTH", "HAS_INNER_DIAMETER",
        "HAS_ANGLE_OF_ENGAGEMENT", "IS_STRAIGHT_TOOTHED", 
        # "HAS_VARIANT",
    ],
}

# -----------------------------------------------------------------------
# Build/Load knowledge graph
# -----------------------------------------------------------------------

# BUILD
def build_property_graph_index(
    all_nodes: List[TextNode],
    persist_dir: str = KG_PERSIST_DIR,
) -> PropertyGraphIndex:
    
    logger.info("===================== Building new KG =====================")

    basic_kg_extractors=[
        ImplicitPathExtractor(),
        SimpleLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3),
            num_workers=4,
            max_paths_per_chunk=10,
        ),
    ]
    custom_kg_extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=EntityType,
        possible_relations=RelationType,
        kg_validation_schema=SCHEMA_VALIDATION_SCHEMA,
        strict=True,
        num_workers=4,
    )

    logger.info("============= Reading each KG-node's text and extracting structured triples")
    try:
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        index = PropertyGraphIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
            llm=Settings.llm,
            llm=Settings.llm,
            kg_extractors=[custom_kg_extractor],
            # kg_extractors=basic_kg_extractors,
            property_graph_store=graph_store,
            show_progress=True,
        )
    except Exception as e:
        logger.error(f"Failed to build property graph index: {e}")
        raise

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)

    return index

# LOAD
def load_property_graph_index(
    persist_dir: str = KG_PERSIST_DIR,
) -> PropertyGraphIndex:
    
    logger.info("===================== Loading existing KG =====================")
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir,
            graph_store=graph_store,
        )
        return load_index_from_storage(storage_context)
    except Exception as e:
        logger.error(f"Failed to load property graph index: {e}")
        raise

# GET OR BUILD
def get_or_build_property_graph_index(
    all_nodes: List[TextNode],
    persist_dir: str = KG_PERSIST_DIR,
) -> PropertyGraphIndex:
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        return load_property_graph_index(persist_dir=persist_dir)
    return build_property_graph_index(all_nodes=all_nodes, persist_dir=persist_dir)

# -----------------------------------------------------------------------
# Build knowledge graph retriever
# -----------------------------------------------------------------------

def build_kg_retriever(
    property_graph_index: PropertyGraphIndex,
    synonym_retriever_llm: str,
    similarity_top_k: int = 8,
    path_depth: int = 3,
    include_text: bool = True,
):
    vector_retriever = VectorContextRetriever(
        graph_store=property_graph_index.property_graph_store,
        embed_model=Settings.embed_model,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
        include_text=include_text,
    )

    print("============= Expanding query keywords → traverse graph")
    synonym_retriever = LLMSynonymRetriever(
        graph_store=property_graph_index.property_graph_store,
        llm=synonym_retriever_llm,
        path_depth=path_depth,
        include_text=include_text,
    )

    return property_graph_index.as_retriever(
        sub_retrievers=[vector_retriever, synonym_retriever],
        include_text=include_text,
    )