# RAG Pipeline — Engineering Catalog Retrieval System

## Architecture & Design Decisions

### Explicit Model Configuration
- Set global **LLM** (generation) and **Embedding Model** (vectorization) explicitly
- Version/pin model names for reproducibility (e.g., embed dim = 1536)

---

### High-Quality PDF Parsing for Engineering Catalogs
- Use OCR + layout-aware parsing
- Robust long-table extraction (preserve rows/columns)
- Export tables as HTML/Markdown for better downstream retrieval + answer grounding
- Keep page boundaries and document structure markers

For ``mechanical parts catalog``, do both:
- **Parse** the PDF reliably (especially long tables) → for chunking + retrieval,
- **Extract** key fields/entities → for structured filtering + knowledge graph.
---
### Mapping extracted layer 1 to extracted layer 2
(TODO)
---

### Page-Level Document Decomposition
- Split parsed output into per-page "sub-documents"
- Preserve metadata: `page_number`, `source_file`, `part_id` (if extracted), `section`, `table_id`

---

### Deterministic Chunking Strategy
- Configure `chunk_size` + `chunk_overlap`
- Choose chunking unit: page-first (catalog page = one part) then optional sub-chunking for long pages/tables

---

### Metadata Normalization + Structured Schema Mapping
- Extract mechanical fields into structured metadata/columns:
  - DIN/ISO, module, pressure angle, torque, material, tolerances, fits, dimensions, units
- Normalize units (`mm` / `Nm` / `deg`) and enforce types (numeric ranges, enums)

---

### Persistent Vector Store Backend (PostgreSQL + pgvector)
- Store: chunks/nodes + embeddings + metadata in a Postgres table
- Enable `pgvector` extension and maintain persistence across restarts
- Add ANN index (HNSW) with cosine distance

---

### Hybrid Retrieval (Dense + Sparse)
- **Dense:** vector KNN over embeddings (HNSW)
- **Sparse:** keyword/BM25-like text search (Postgres FTS) with language config
- Combine both signals in one unified retrieval pipeline

---

### Fusion Ranking for Hybrid Search
- Use a fusion retriever to normalize/merge dense + sparse result lists
- Modes like relative-score/rank fusion to reduce score-scale mismatch

---

### Structured Filtering in Retrieval
- Apply SQL-level filters on metadata columns:
  - Range filters (`module` between x–y, `torque` ≥ z)
  - Exact match (`DIN code`, `material`)
  - Potential nested-like metadata via `JSONB` + indexed keys (if needed)

---

### Ingestion Guard (Production Hygiene)
- Detect unchanged source files via hash/version
- Skip parse/split/embed when unchanged
- Support incremental updates (re-ingest only changed parts/pages)

---

### Observability / Debug Visibility
- Ability to inspect:
  - Parsed text + extracted tables per page
  - Chunk boundaries and overlaps
  - Stored embeddings count + row counts
  - Raw dense scores vs sparse scores vs fused scores

---

### Query Pipeline Control
- Explicit top-k settings per retriever (`dense_top_k`, `sparse_top_k`)
- Control query generation on/off (keep deterministic for engineering use cases)

---

### Knowledge-Graph-Ready Outputs
- Produce entity candidates and relations from parsed docs:
  - `part ↔ material`, `part ↔ fits`, `part ↔ tolerance class`, `part ↔ compatible parts`
- Store edges in relational tables now (or later sync to Neo4j)
- Support "KG filter + vector/hybrid rerank" retrieval pattern later

---

### Multimodal Hooks *(Future)*
- Keep references to page images / 2D–3D drawings in metadata
- Enable future image embeddings or vision-based attribute extraction without changing the core pipeline

<details>
<summary>Challenges in Mechanical RAG-Catalogs</summary>

## Table-Heavy Content

- **Dimension/tolerance tables poorly handled by naive text splitters**
- **Loss of row/column structure → wrong value grounding**

## Unit Normalization

- mm vs inch, Nm vs kNm, degrees vs radians
- Retrieval must not mix incompatible units

## Highly Structured Queries

- Example: `"DIN5412, d=55, module 2.5, torque ≥ 120Nm"`
- Requires strong numeric filtering + range queries

## Small Numeric Differences

- 2.5 vs 2.0 module → semantically similar but technically different
- Pure vector similarity can confuse close specs

## Part-Level Granularity

- Each page = one entity
- Over-chunking may break entity cohesion

## Metadata Extraction

- DIN, material, tolerance class must become structured fields
- Not just free text embeddings

## Hybrid Necessity

- Exact code match (DIN, ISO, SKU) → keyword search
- Descriptive specs → vector search

## Cross-Part Relations

- Part ↔ compatible part ↔ material ↔ fit
- Requires graph-style modeling beyond flat chunks

## Visual Information

- 2D/3D images not embedded by default
- Important geometric info may be lost

## Ambiguous Engineering Language

- "Fit H7" or "IT6 tolerance" requires domain grounding

## Precision Requirement

- Engineering search demands correctness, not approximate "semantic closeness"

## Catalog Evolution

- Versioning of parts/spec changes must invalidate embeddings correctly

---

## Why Mechanical RAG Systems Need

These challenges are why mechanical RAG systems require:

- **Strong structured schema**
- **Hybrid retrieval**
- **Careful chunking strategy**
- **Eventually KG integration**

</details>