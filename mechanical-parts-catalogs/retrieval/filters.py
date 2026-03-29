import logging
from typing import Tuple
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from mapping import normalize_family, normalize_material
from retrieval.intent import QueryIntent

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------

logger = logging.getLogger(__name__)

# --------------------------------------------------
# FIELD ROUTING MAPS
# --------------------------------------------------

EXACT_FIELDS = {
    "node_type", "family", "module",
    "spur_gear_material", "angle_of_engagement", "art_nr",
}

RANGE_FIELD_MAP = {
    "torque_min": ("DM", FilterOperator.GTE),
    "torque_max": ("DM", FilterOperator.LTE),
    "ZZ_min":  ("ZZ",  FilterOperator.GTE),
    "ZZ_max":  ("ZZ",  FilterOperator.LTE),
    "ZB_min":  ("ZB",  FilterOperator.GTE),
    "ZB_max":  ("ZB",  FilterOperator.LTE),
    "G_min":   ("G",   FilterOperator.GTE),
    "G_max":   ("G",   FilterOperator.LTE),
    "OTK_min": ("ØTK", FilterOperator.GTE),
    "OTK_max": ("ØTK", FilterOperator.LTE),
}

# --------------------------------------------------
# FILTERED RETRIEVER — uses extracted intent
# --------------------------------------------------

def build_filtered_retriever(
    intent: QueryIntent,
    basic_index: VectorStoreIndex,
    hybrid_index: VectorStoreIndex,
    use_hybrid: bool = False,
    similarity_top_k: int = 8,
    sparse_top_k: int = 8,
):
    active_filters = []
    intent_dict = intent.model_dump(exclude_none=True)

    for field, value in intent_dict.items():
        if field in EXACT_FIELDS:
            if field == "module" and isinstance(value, float):
                value = round(value, 4)
            if field == "spur_gear_material" and isinstance(value, str):
                value = normalize_material(value)
            if field == "family" and isinstance(value, str):
                value = normalize_family(value)
            active_filters.append(
                MetadataFilter(key=field, value=value, operator=FilterOperator.EQ)
            )
        elif field in RANGE_FIELD_MAP:
            metadata_key, operator = RANGE_FIELD_MAP[field]
            active_filters.append(
                MetadataFilter(key=metadata_key, value=value, operator=operator)
            )

    filters = (
        MetadataFilters(filters=active_filters, condition=FilterCondition.AND)
        if active_filters else None
    )

    index = hybrid_index if use_hybrid else basic_index
    query_mode = VectorStoreQueryMode.HYBRID if use_hybrid else VectorStoreQueryMode.DEFAULT

    kwargs = dict(
        similarity_top_k=similarity_top_k,
        vector_store_query_mode=query_mode,
        filters=filters,
    )
    if use_hybrid:
        kwargs["sparse_top_k"] = sparse_top_k
        # kwargs["alpha"] = alpha

    return index.as_retriever(**kwargs)