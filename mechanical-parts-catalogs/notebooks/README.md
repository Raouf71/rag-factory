# Step-by-Step Breakdown of the Pipeline


## RAG Pipeline Stages

Each domain goes through the full ingestion-to-answer pipeline:

- **Document Ingestion** — PDF/HTML/CSV parsing and cleaning
- **Chunking** — fixed-size, sentence-aware, and semantic chunking strategies
- **Embedding** — dense vector representations (e.g. `text-embedding-3-small`, `bge-m3`)
- **Indexing** — vector stores (FAISS, Qdrant, Weaviate) + graph stores for KG-RAG
- **Retrieval** — top-k similarity search, hybrid search, graph traversal, agent-driven retrieval

## Pipeline Details

<details>
<summary>basic pipeline</summary>

---

## Step 1 — Parsing / Loading
**Raw PDF → `Document` objects**

- Tool: `LlamaParse(...).load_data("../data/bevel_gear.pdf")`
- Output: a list of LlamaIndex `Document` objects (`docs`)

---

## Step 2 — Splitting
**`Document` objects → smaller `Document` / Node chunks**

- Strategy: manual page-level split using `doc.text.split("\n---\n")`
- Each chunk is re-wrapped: `sub_doc = Document(text=..., metadata=...)`
- Output: `sub_docs` — a list of page-level `Document` objects

---

## Step 3 — Indexing (Vector)
**`sub_docs` → searchable vector index**

- Index: `VectorStoreIndex.from_documents(sub_docs, embed_model=...)`
- Retriever: `base_index.as_retriever(similarity_top_k=10)`
- Query interface: `RetrieverQueryEngine`

---

## Step 4 — Knowledge Graph Extraction + Indexing
**Entity/relationship extraction → graph store (Neo4j)**

- Graph store: `Neo4jPGStore`
- Graph index: `PropertyGraphIndex.from_documents(...)`
- KG extractors:
  - `ImplicitPathExtractor()` — implicit path inference
  - `SimpleLLMPathExtractor(llm=OpenAI(...), max_paths_per_chunk=..., num_workers=...)` — LLM-driven path extraction
- Embeddings for graph context retrieval: `OpenAIEmbedding("text-embedding-3-small")`

---

## Step 5 — Retrieval (Vector + KG → Merged)
**Two retrievers → fused results**

- Vector retriever: `vector_retriever = base_index.as_retriever(...)`
- KG retriever: `VectorContextRetriever(...)` — embedding similarity + graph expansion via `path_depth`
- Fusion: `CustomRetriever(BaseRetriever)` — merges results by `node_id`, deduplicating across both retrievers

</details>


<details>
<summary>advanced pipeline</summary>

---

## Step 1 — Parsing / Loading

---

## Step 2 — Splitting

---

## Step 3 — Indexing
---

## Step 4 — Knowledge Graph Extraction + Indexing

---

## Step 5 — Retrieval (Vector + KG → Merged)

</details>

[🔼 Back to top](#rag-pipeline-stages)
