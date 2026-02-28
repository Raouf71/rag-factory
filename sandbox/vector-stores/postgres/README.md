``VectorStoreIndex.from_documents(...)`` does:

1. Generate embeddings for each document.
2. Store them in PostgreSQL.
3. Create a vector index (HNSW) on the embedding column.


In RAG, an index is an optimized search structure on top of the data that makes semantic retrieval fast and scalable. HNSW is a graph-based algorithm that:

1. Organizes vectors in a search-optimized form
2. Avoids full table scan (brute-force comparison with all vectors)
3. Enables Approximate Nearest Neighbor (ANN) search

> Without index:<br>
Compare query vector against ALL stored vectors
→ O(N)

>With HNSW index:<br>
Navigate graph structure
→ ~O(log N) approximate search

HNSW index tuning parameters (speed ↔ recall ↔ memory) for pgvector’s HNSW, using cosine distance via vector_cosine_ops.

Key Parameters:

``m``  : The maximum number of connections per node per layer.<br>
``ef_construction``  : Controls index quality/graph build time trade-off.<br>
``ef_search`` : Controls search accuracy/speed trade-off.

`vector_cosine_ops` means the HNSW index computes similarity using cosine distance (appropriate for most text embeddings, especially if normalized / direction matters more than magnitude).