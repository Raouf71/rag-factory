# Step-by-Step Breakdown of the Pipeline


## RAG Pipeline Stages

Each domain goes through the full ingestion-to-answer pipeline:

- **Document Parsing** — PDF/HTML/CSV.<br>
Parsing can be done in two different ways: <br>
  * ``Parse into Document``:
    * Output = larger logical units (often per page or file)
    * Chunking happens **later** via your chunking strategy (chunk_size, overlap)
    * More flexible
    * Best when:
      * control over splitting is required
      * experimenting with chunk sizes
      * combined with structured extraction
      * hybrid/KG pipelines are built <br>

    👉 Recommended for production RAG.
  * ``Parse into Node``:
    * Output = already pre-chunked pieces
    * Chunking happens **during parsing**
    * Less control afterward
    * Best when:
      * quick setup is required
      * parser’s splitting is trusted
      * no custom chunk logic needed<br>

    👉 Recommended for fast prototyping.

- **Splitting**<br>
semantic grouping of document pages:
  * e.g., 1 page = 1 mechanical part
  * preserves entity boundaries<br>
```(TODO)
👉 Split before extraction, because extraction should **operate on a clean entity unit**.
    <br>
    
    If you don’t split:
      * Extractor may mix multiple parts
      * Wrong DIN/material/torque assigned<br>

    If you split:
      * Correct metadata association
      * No cross-part contamination
```

- **Document Extracting**

  1. TWO-LAYER EXTRACTION ARCHITECTURE:
      * Layer 1 — ``PER_PAGE``
        Extract:
        - Part-level metadata (DIN, module, material, torque range, fits, tolerance class, description)

          Why:
          - Each page = one entity
          - Clean schema
          - Enables structured filtering + KG node creation
          - Stable and deterministic

      * Layer 2 - ``PER_TABLE_ROW``:
        Extract:
        - Each dimension/spec row separately
        (e.g., size variant, bore diameter, torque value, tolerance per row)

          Why:
          - Long tables cause LLM truncation in PER_DOC
          - Guarantees exhaustive row coverage
          - Enables exact numeric lookups without hallucination
          - Supports range filtering and comparison

        > Metadata extraction gives entity identity + filters (**“Who is the part?”**) and row extraction gives precise engineering lookup (**“What are its exact dimension variants?”**). 
        
        The architecture prevents:
          * Missing table rows
          * Numeric hallucinations
          * Mixed specs across parts

  2. mapping layer 1 to layer 2:
      - Builds a `dict` keyed by `part_id` from all Layer 1 parts
      - Iterates every Layer 2 row and looks up its `part_id` in that dict
      - Appends matching rows into `part.dimension_rows`
      - Returns the final `Dict[str, PartWithRows]` — each part fully assembled with all its dimension rows nested inside

      ### Data shape after mapping

      ```
      {
        "spur_gear_m1_0_pom": PartWithRows(
            family="spur_gear",
            module=1.0,
            spur_gear_material="POM",
            dimension_rows=[
                TableRowSchema(ZZ=10, ZB=..., art_nr="12345", ...),
                TableRowSchema(ZZ=12, ZB=..., art_nr="12346", ...),
                ...
            ]
        ),
        ...
      }
      ```
  

- **Chunking** — fixed-size, sentence-aware, and semantic chunking strategies
- **Embedding** — dense vector representations (e.g. `text-embedding-3-small`, `bge-m3`)
- **Indexing** — vector stores (FAISS, Qdrant, Weaviate, PostgreSQL) + graph stores for KG-RAG
- **Retrieval** — top-k similarity search, hybrid search, graph traversal, agent-driven retrieval

## Challenges of RAG for catalogs



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
