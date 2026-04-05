import logging
import os
from dotenv import load_dotenv
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from config.settings import EMBED_DIM

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------

load_dotenv()
logger = logging.getLogger(__name__)

DB_NAME    = os.getenv("POSTGRES_DB")
DB_HOST    = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT    = int(os.getenv("POSTGRES_PORT", 5432))
DB_USER    = os.getenv("POSTGRES_USER", "postgres")
DB_PASS    = os.getenv("POSTGRES_PASSWORD")

TABLE_NAME_BASIC = "prod_mechanical_parts_vector"
TABLE_NAME_HYBRID = "prod_mechanical_parts_hybrid"

# --------------------------------------------------
# BUILD LLAMAINDEX PGVECTOR STORES
# --------------------------------------------------

def build_pgvector_store(store_type: str) -> PGVectorStore:
    common_kwargs = {
        "database": DB_NAME,
        "host":     DB_HOST,
        "password": DB_PASS,
        "port":     DB_PORT,
        "user":     DB_USER,
        "embed_dim": EMBED_DIM,
        "hnsw_kwargs": {
            "hnsw_m":              16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search":       40,
            "hnsw_dist_method":    "vector_cosine_ops",
        },
    }
    if store_type == "basic":
        return PGVectorStore.from_params(
            table_name=TABLE_NAME_BASIC,
            **common_kwargs,
        )
    if store_type == "hybrid":
        return PGVectorStore.from_params(
            table_name=TABLE_NAME_HYBRID,
            hybrid_search=True,
            text_search_config="simple",
            **common_kwargs,
        )
    raise ValueError("store_type must be 'basic' or 'hybrid'")

# --------------------------------------------------
# INDEX NODES INTO POSTGRES
# --------------------------------------------------

def index_nodes_with_store(all_nodes, vector_store) -> VectorStoreIndex:
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )
    except Exception as e:
        logger.error(f"Failed to index nodes into vector store: {e}")
        raise