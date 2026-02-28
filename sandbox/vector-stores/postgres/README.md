## Indexing with PostgreSQL

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
``ef_search`` : Controls search accuracy/speed trade-off.<br>
`vector_cosine_ops` means the HNSW index computes similarity using cosine distance (appropriate for most text embeddings, especially if normalized / direction matters more than magnitude).


## SQL Commands:

* SELECT - extracts data from a database
* UPDATE - updates data in a database
* DELETE - deletes data from a database
* INSERT INTO - inserts new data into a database
* CREATE DATABASE - creates a new database
* ALTER DATABASE - modifies a database
* CREATE TABLE - creates a new table
* ALTER TABLE - modifies a table
* DROP TABLE - deletes a table
* CREATE INDEX - creates an index (search key)
* DROP INDEX - deletes an index 

### From terminal
1) List tables in vector_db (fastest check):

    ```bash
    psql -h localhost -U postgres -d vector_db -c "\dt"
    ```

<div align="center">

| Schema |                 Name                 | Type  |   Owner  |
|--------|:------------------------------------:|-------|:--------:|
| public | data_paul_graham_essay               | table | postgres |
| public | data_paul_graham_essay_hybrid_search | table | postgres |

</div>



2) Verify you’re in the right database:

    ```bash
    psql -h localhost -U postgres -d vector_db -c "SELECT current_database(), current_schema();"
    ```
<div align="center">

| current_database | current_schema |
|:----------------:|:--------------:|
|     vector_db    |     public     |
</div>

3) Dump/store database files

    ```bash
    pg_dump -h localhost -U postgres -d vector_db -F c -f sandbox/vector-stores/postgres/vector_db.dump
    ```