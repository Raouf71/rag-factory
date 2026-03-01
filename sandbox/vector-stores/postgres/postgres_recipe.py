from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
import textwrap

documents = SimpleDirectoryReader("../data/paul_graham").load_data()
print("Document ID:", documents[0].doc_id)

# ------------------------ vector search ------------------------

# Create PGVectorStore instance
url = make_url(connection_string)
db_name = "vector_db"
vector_store = PGVectorStore.from_params(
    database=db_name,               # tells LlamaIndex to use PostgreSQL as the backend
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="paul_graham_essay", # is where chunks + embeddings + metadata will be stored
    embed_dim=1536,                 # openai embedding dimension
    hnsw_kwargs={                   # configure an HNSW ANN index using cosine distance
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

# Bind index to Postgres table (table_name) 
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Use existing index OR Build the RAG index from documents 
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     documents, 
#     storage_context=storage_context, 
#     show_progress=True
# )

# Embed user query + retrieve top-k chunks + send context to LLM to generate answer
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do?")

# ------------------------ enhanced hybrid search ------------------------

from sqlalchemy import make_url

url = make_url(connection_string)
hybrid_vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="paul_graham_essay_hybrid_search",
    embed_dim=1536,  # openai embedding dimension
    hybrid_search=True,
    text_search_config="english",
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

storage_context = StorageContext.from_defaults(
    vector_store=hybrid_vector_store
)

# Create index
# TODO: Implement ingestion guard to avoid re-processing UNCHANGED docs
hybrid_index = VectorStoreIndex.from_vector_store(vector_store=hybrid_vector_store)
# hybrid_index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context
# )

from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Check following link for documentation (VectorStoreQueryMode)
# https://developers.llamaindex.ai/python/framework-api-reference/storage/vector_store/?utm_source=chatgpt.com
vector_retriever = hybrid_index.as_retriever(
    vector_store_query_mode="default",      # dense vector search (KNN over embeddings). It uses embedding column + distance operator (cosine)
    similarity_top_k=5, 
)
text_retriever = hybrid_index.as_retriever(
    vector_store_query_mode="sparse",       # sparse / keyword-style retrieval (token-based / BM25)
    similarity_top_k=5,  # interchangeable with sparse_top_k in this context
)
retriever = QueryFusionRetriever(
    [vector_retriever, text_retriever],
    similarity_top_k=5,
    num_queries=1,  # set this to 1 to disable query generation
    mode="relative_score",  
    use_async=False,
)

response_synthesizer = CompactAndRefine()
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

response = query_engine.query(
    "Who does Paul Graham think of with the word schtick, and why?"
)
print(response)