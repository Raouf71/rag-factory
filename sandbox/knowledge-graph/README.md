

* Free-form extractors (``SimpleLLMPathExtractor``):
are for unstructured documents (PDFs, articles) where you don't know the schema upfront. Our catalog is the opposite — schema is fully known, so enforcing it is strictly better.

* ``SchemaLLMPathExtractor with strict=True`` is netter — because our catalog is highly structured and domain-specific:

    * We already know exactly what entities exist: GEAR, MATERIAL, MODULE, FAMILY, ARTICLE
    * We already know exactly what relations exist: HAS_MATERIAL, HAS_MODULE, HAS_TEETH_COUNT, * HAS_TORQUE etc.
    * These map 1:1 to your existing metadata fields in build_part_node_metadata and build_row_node_metadata — single source of truth again