import os
from dotenv import load_dotenv
from llama_index.core import PromptTemplate

load_dotenv()

# -----------------------------------------------------------------------
# API Keys
# -----------------------------------------------------------------------
LLAMA_CLOUD_API_KEY = os.getenv("LLAMAINDEX_CLOUD_KEY")
DEEPSEEK_KEY        = os.getenv("DEEPSEEK_API_KEY")
OPENAI_KEY          = os.getenv("OPENAI_API_KEY")
NEO4J_PASSWORD      = os.getenv("NEO4J_PASSWORD")

# -----------------------------------------------------------------------
# PostgreSQL
# -----------------------------------------------------------------------
POSTGRES_DB   = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD")

# -----------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------
EMBED_DIM = 1536  # text-embedding-3-small

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
PDF_PATH       = os.getenv("PDF_PATH", "mechanical_parts_catalogs/data/gear_diff_m.pdf")
KG_PERSIST_DIR = "./storage/property_graph"

# -----------------------------------------------------------------------
# Table names
# -----------------------------------------------------------------------
TABLE_NAME_BASIC  = "prod_mechanical_parts_vector"
TABLE_NAME_HYBRID = "prod_mechanical_parts_hybrid"

# -----------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------
SYSTEM_PROMPT_L1 = (
    "You are an expert at extracting specifications of spur gears from catalog documents."
)

SYSTEM_PROMPT_L2 = """
You are an expert at extracting rows from dense dimension tables.
For every extracted table row, also extract the parent part identity fields:
- family
- module
- spur_gear_material
Use page heading, section title, and local table context to infer them when they are not repeated in every row.
"""

# -----------------------------------------------------------------------
# QA prompt
# -----------------------------------------------------------------------
CATALOG_QA_PROMPT = PromptTemplate(
    "You are an expert in mechanical gear catalogs.\n"
    "Answer the question using only the catalog data provided below.\n"
    "If the answer is not in the context, say so explicitly.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
)