import logging
import hashlib
import json
from typing import List, Dict, Tuple
from llama_index.core.schema import TextNode
from pipeline.schemas import PartSchema, TableRowSchema
from pipeline.mapping import PartWithRows

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Create text and metadata for each node
# -----------------------------------------------------------------------

# 1) FINAL TEXT TEMPLATES
def build_part_node_text(part: PartWithRows) -> str:
    """Only semantic, natural-language content goes here — this is what gets embedded."""
    lines = []

    if getattr(part, "title", None):
        lines.append(f"Title: {part.title}")

    lines.append(
        f"This is a {part.family.replace('_', ' ')} made of {part.spur_gear_material} "
        f"with module {part.module}. "
        f"It is {'straight toothed' if part.straight_toothed else 'helical'} "
        f"and has an engagement angle of {part.angle_of_engagement} degrees. "
        f"This part family contains {len(part.dimension_rows)} catalog variants."
    )

    if getattr(part, "description", None):
        lines.append(part.description)

    return "\n".join(lines)

def build_row_node_text(part: PartWithRows, row: TableRowSchema) -> str:
    """
    Embed only the fields a user would describe in natural language.
    Numeric dimensions go to metadata for range filtering — not here.
    """
    return (
        f"This catalog row describes a {part.family.replace('_', ' ')} variant "
        f"made of {part.spur_gear_material} with module {part.module}. "
        f"It has {int(row.ZZ)} teeth, a max torque of {row.DM} Ncm, "
        f"a pitch circle diameter of {row.ØTK} mm, "
        f"and a tip diameter of {row.ØKK} mm. "
        f"Article number: {row.art_nr}."
    )

# 2) FINAL METADATA PAYLOADS
def build_part_node_metadata(part: PartWithRows) -> dict:
    """All filterable fields — used for exact match and structured filters."""
    return {
        "node_type":          "part",
        "part_id":            part.part_id,
        "page_number":         part.page_number,
        "family":             part.family,
        "module":             part.module,
        "spur_gear_material": part.spur_gear_material,
        "straight_toothed":   part.straight_toothed,
        "angle_of_engagement": part.angle_of_engagement,
        "variant_count":      len(part.dimension_rows),
    }

def build_row_node_metadata(part: PartWithRows, row: TableRowSchema, parent_node_id: str) -> dict:
    """
    All numeric dimensions go here for >= / <= filtering in pgvector.
    parent_node_id enables fetching the parent part context after retrieval.
    """
    return {
        "node_type":          "row",
        "part_id":            part.part_id,
        "page_number":        part.page_number,
        "parent_node_id":     parent_node_id,
        "family":             part.family,
        "module":             part.module,
        "spur_gear_material": part.spur_gear_material,
        "art_nr":             row.art_nr,
        # ── numeric dims — all filterable ──────────────────
        "ZZ":  row.ZZ,    # teeth count
        "ZB":  row.ZB,    # tooth width
        "ØB":  row.ØB,    # inner diameter
        "ØTK": row.ØTK,   # pitch circle diameter
        "ØKK": row.ØKK,   # tip diameter
        "ØN":  row.ØN,    # hub diameter
        "L":   row.L,     # length
        "ØFM": row.ØFM,   # ring gear diameter
        "WS":  row.WS,    # girder width
        "G":   row.G,     # weight (g)
        "DM":  row.DM,    # max torque (Ncm)
    }

# 3) NODE GENERATION FOR ALL CATALOGS
def make_part_node_id(part: PartWithRows) -> str:
    return f"part::{part.part_id}"

def make_row_node_id(part: PartWithRows, row: TableRowSchema) -> str:
    if row.art_nr:
        return f"row::{part.part_id}::{row.art_nr}"

    fingerprint = hashlib.md5(
        json.dumps([
            row.ZZ,
            row.ZB,
            row.ØB,
            row.ØTK,
            row.ØKK,
            row.ØN,
            row.L,
            row.ØFM,
            row.WS,
            row.G,
            row.DM,
        ]).encode()
    ).hexdigest()[:12]

    return f"row::{part.part_id}::anon_{fingerprint}"

# -----------------------------------------------------------------------
# Build entity-centric nodes
# -----------------------------------------------------------------------

def build_retrieval_nodes(
    mapped_parts: Dict[str, PartWithRows],
) -> Tuple[List[TextNode], List[TextNode]]:
    parent_nodes: List[TextNode] = []
    child_nodes: List[TextNode] = []

    for _, part in mapped_parts.items():
        parent_node_id = make_part_node_id(part)

        part_node = TextNode(
            id_=parent_node_id,
            text=build_part_node_text(part),
            metadata=build_part_node_metadata(part),
        )
        parent_nodes.append(part_node)

        for row in part.dimension_rows:
            row_node = TextNode(
                id_=make_row_node_id(part, row),
                text=build_row_node_text(part, row),
                metadata=build_row_node_metadata(part, row, parent_node_id),
            )
            child_nodes.append(row_node)

    return parent_nodes, child_nodes
