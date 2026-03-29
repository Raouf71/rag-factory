import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pydantic import Field
from schemas import PartSchema, TableRowSchema
from extraction import ExtractionResult

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# normalizing
# -----------------------------------------------------------------------

class PartWithRows(PartSchema):
    part_id: str
    page_number: Optional[int] = None         
    dimension_rows: List[TableRowSchema] = Field(default_factory=list)

class TableRowWithPartId(TableRowSchema):
    part_id: str
    page_number: Optional[int] = None         

FAMILY_SYNONYMS = {
    "spur_gear": [
        "stirnrad",
        "stirnraeder",
        "stirnräder",
        "spur gear",
        "spur gears",
        "straight gear"
    ],
    "bevel_gear": [
        "kegelrad",
        "kegelraeder",
        "kegelräder",
        "bevel gear",
        "bevel gears"
    ]
}

def normalize_family(family: str) -> str:
    f = family.lower()

    for canonical, synonyms in FAMILY_SYNONYMS.items():
        for s in synonyms:
            if s in f:
                return canonical

    return f.replace(" ", "_")

def normalize_material(material: str) -> str:
    material_norm = material.strip().lower()

    if "polyacetal" in material_norm or "pom" in material_norm:
        return "pom"
    if "polyketon" in material_norm or "pk" in material_norm:
        return "pk"

    return (
        material_norm
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(".", "")
    )

def normalize_module(module: float) -> str:
    return f"{round(module, 4):g}".replace(".", "_")

# -----------------------------------------------------------------------
# build part_id
# -----------------------------------------------------------------------

def build_part_id(family: str, module: float, material: str) -> str:
    family_key = normalize_family(family)
    module_key = normalize_module(module)
    material_key = normalize_material(material)
    return f"{family_key}_m{module_key}_{material_key}"

# -----------------------------------------------------------------------
# map layers to each other
# -----------------------------------------------------------------------

def attach_part_id_to_layer1_parts(
    extraction_result: ExtractionResult,
) -> List[PartWithRows]:
    enriched_parts: List[PartWithRows] = []
    seen_part_ids: set[str] = set()

    for part, meta in zip(extraction_result.data, extraction_result.field_metadata):
        if isinstance(part, dict):
            part = PartSchema(**part)

        normalized_family   = normalize_family(part.family)
        normalized_material = normalize_material(part.spur_gear_material)
        part_id = build_part_id(
            family=normalized_family,
            module=part.module,
            material=normalized_material,
        )

        if part_id in seen_part_ids:
            logger.info(f"Duplicate part skipped: '{part_id}'")
            continue

        seen_part_ids.add(part_id)
        enriched_parts.append(
            PartWithRows(
                part_id=part_id,
                page_number=meta.get("page_number"),      
                family=normalized_family,
                spur_gear_material=normalized_material,
                module=part.module,
                straight_toothed=part.straight_toothed,
                angle_of_engagement=part.angle_of_engagement,
            )
        )
    return enriched_parts

def attach_part_id_to_layer2_rows(
    extraction_result: ExtractionResult,
) -> List[TableRowWithPartId]:
    enriched_rows: List[TableRowWithPartId] = []

    for row, meta in zip(extraction_result.data, extraction_result.field_metadata):
        if isinstance(row, dict):
            row = TableRowSchema(**row)

        normalized_family   = normalize_family(row.family)
        normalized_material = normalize_material(row.spur_gear_material)

        enriched_rows.append(
            TableRowWithPartId(
                part_id=build_part_id(
                    family=normalized_family,
                    module=row.module,
                    material=normalized_material,
                ),
                page_number=meta.get("page_number"),    
                family=normalized_family,
                spur_gear_material=normalized_material,
                module=row.module,
                ZZ=row.ZZ, ZB=row.ZB, ØB=row.ØB,
                ØTK=row.ØTK, ØKK=row.ØKK, ØN=row.ØN,
                L=row.L, ØFM=row.ØFM, WS=row.WS,
                G=row.G, DM=row.DM, art_nr=row.art_nr,
            )
        )
    return enriched_rows

# -----------------------------------------------------------------------
# silent data loss on join misses
# -----------------------------------------------------------------------

# Add a proper return type — no silent tuple unpacking errors
@dataclass
class MappingResult:
    parts: Dict[str, PartWithRows]
    orphaned_rows: List[TableRowWithPartId] = field(default_factory=list)

    @property
    def orphaned_count(self) -> int:
        return len(self.orphaned_rows)

    @property
    def orphaned_part_ids(self) -> set:
        return {r.part_id for r in self.orphaned_rows}
    
def map_parts_and_rows(
    part_data: List[PartWithRows],
    table_row_data: List[TableRowWithPartId],
) -> MappingResult:
    parts_by_id: Dict[str, PartWithRows] = {
        part.part_id: part for part in part_data
    }
    orphaned_rows: List[TableRowWithPartId] = []

    for row in table_row_data:
        if row.part_id in parts_by_id:
            parts_by_id[row.part_id].dimension_rows.append(
                TableRowSchema(
                    family=row.family,
                    module=row.module,
                    spur_gear_material=row.spur_gear_material,
                    ZZ=row.ZZ,
                    ZB=row.ZB,
                    ØB=row.ØB,
                    ØTK=row.ØTK,
                    ØKK=row.ØKK,
                    ØN=row.ØN,
                    L=row.L,
                    ØFM=row.ØFM,
                    WS=row.WS,
                    G=row.G,
                    DM=row.DM,
                    art_nr=row.art_nr,
                )
            )
        else:
            orphaned_rows.append(row)

    return MappingResult(parts=parts_by_id, orphaned_rows=orphaned_rows)

def log_mapping_diagnostics(result: MappingResult) -> None:
    total_rows = sum(len(p.dimension_rows) for p in result.parts.values())

    logger.info(f"Mapped parts   : {len(result.parts)}")
    logger.info(f"Joined rows    : {total_rows}")

    if result.orphaned_rows:
        logger.warning(f"Orphaned rows  : {result.orphaned_count} rows could not be joined to any parent part. Unmatched part_ids:")
        
        for part_id in sorted(result.orphaned_part_ids):
            count = sum(1 for r in result.orphaned_rows if r.part_id == part_id)
            logger.warning(f"  → '{part_id}' ({count} row(s))")
    else:
        logger.info("Orphaned rows  : 0 — all rows joined successfully ✓")

