## Production-grade techniques for RAG

* Caching:
    * Extraction results (most important)
    * Knowledge-Graph index
    * Embeddings caching. <br>
    LlamaIndex has ``IngestionCache`` with SimpleDocumentStore built in — it skips re-embedding nodes that haven't changed.
    * Query caching
* Parallel document ingestion
* Prompt re-phrasing
* Model routing (small + large models)
* Security+Hallucination Guardrails
* Evaluation (gold QA sets, retrieval metrics)
* Token-cost management
* Latency management
* RBCA

---
### <ins>Implemented techniques so far</ins>:

* Caching extraction results: Hash key inputs like:3
    * Extraction agent  
    * PDF content hash
    * extraction prompt version
    * schema version
    * model/provider version
    * pipeline code version for extraction logic
    * extraction results<br>
&rarr; Store cached data as plain JSON-serializable dict/list.

> cache_key = hash(pdf_bytes + prompt_version + schema_version + extractor_version + extraction_results)
