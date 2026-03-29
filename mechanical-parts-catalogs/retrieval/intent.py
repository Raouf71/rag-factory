import logging
from typing import Optional
from pydantic import BaseModel, Field
from llama_index.core import PromptTemplate
from llama_index.core import Settings

# -----------------------------------------------------------------------
# Setting up (Credentials, logger, etc.)
# -----------------------------------------------------------------------

logger = logging.getLogger(__name__)

# --------------------------------------------------
# QUERY INTENT — mirrors stored metadata fields
# --------------------------------------------------

class QueryIntent(BaseModel):
    # exact match fields
    node_type: Optional[str] = Field(
        default=None,
        description="'part' for general/summary questions, 'row' for specific dimensions or article numbers"
    )
    family: Optional[str] = Field(
        default=None,
        description="Mechanical part family, e.g. 'spur_gear', 'bevel_gear'"
    )
    module: Optional[float] = Field(
        default=None,
        description="Gear module e.g. 0.5, 0.7, 1.0, 1.5, 2.0, 2.5"
    )
    spur_gear_material: Optional[str] = Field(
        default=None,
        description="The material from which the spur gear was manufactured (e.g. Steel, Stainless Steel, Plastics (Polyketon (PK), Polyacetal (POM)), etc.)"
    )
    angle_of_engagement: Optional[int] = Field(
        default=None,
        description="Pressure angle in degrees. It is often written in Degrees (°)."
    )
    art_nr: Optional[str] = Field(
        default=None,
        description="'Art.-Nr.' is an abbreviation for the German term Artikelnummer, which translates to Article Number, e.g. 'SH2023HF' or 'SH20110HF'."
    )

    # range fields — teeth count (ZZ)
    ZZ_min: Optional[float] = Field(default=None, description="Minimum number of teeth (ZZ > or >= value)")
    ZZ_max: Optional[float] = Field(default=None, description="Maximum number of teeth (ZZ < or <= value)")

    # range fields — width (ZB)
    ZB_min: Optional[float] = Field(default=None, description="Minimum gear width in mm")
    ZB_max: Optional[float] = Field(default=None, description="Maximum gear width in mm")

    # range fields — weight (G)
    G_min: Optional[float] = Field(default=None, description="Minimum weight in grams")
    G_max: Optional[float] = Field(default=None, description="Maximum weight in grams")

    # range fields — pitch circle diameter (ØTK)
    OTK_min: Optional[float] = Field(default=None, description="Minimum pitch circle diameter in mm")
    OTK_max: Optional[float] = Field(default=None, description="Maximum pitch circle diameter in mm")

    # range fields — torque (DM**)
    torque_min: Optional[float] = Field(default=None, description="Minimum Torque in Ncm")
    torque_max: Optional[float] = Field(default=None, description="Maximum Torque in Ncm")

# --------------------------------------------------
# INTENT EXTRACTION
# --------------------------------------------------
 
INTENT_EXTRACTION_PROMPT = PromptTemplate(
    "You extract structured filter parameters from a mechanical gear catalog query.\n\n"
    "Rules:\n"
    "- Set node_type='row' if the user asks for specific dimensions, diameters, weight, torque, or article numbers.\n"
    "- Set node_type='part' if the user asks for general info, summaries, or variant counts.\n"
    "- For range queries (e.g. 'torque > 200'), set the corresponding _min/_max field.\n"
    "- 'greater than' or '>' → use _min field.\n"
    "- 'less than' or '<' → use _max field.\n"
    "- Only populate fields explicitly mentioned or clearly implied.\n"
    "- Return null for any field not mentioned.\n\n"
    "Normalization rules:\n"
    "- Normalize family to exactly one of: 'spur_gear', 'bevel_gear'.\n"
    "  e.g. 'Stirnrad', 'spur gear', 'straight gear' → 'spur_gear'\n"
    "  e.g. 'Kegelrad', 'bevel gear' → 'bevel_gear'\n"
    "- Normalize spur_gear_material to exactly one of: 'steel', 'stainless_steel', 'pom', 'pk'.\n"
    "  e.g. 'Stahl', 'Steel' → 'steel'\n"
    "  e.g. 'Edelstahl', 'Stainless' → 'stainless_steel'\n"
    "  e.g. 'Polyacetal', 'POM' → 'pom'\n"
    "  e.g. 'Polyketon', 'PK' → 'pk'\n\n"
    "Query: {query}\n"
)

OLD_INTENT_EXTRACTION_PROMPT = PromptTemplate(
    "You extract structured filter parameters from a mechanical gear catalog query.\n"
    "Rules:\n"
    "- Set node_type='row' if the user asks for specific dimensions, diameters, weight, torque, or article numbers.\n"
    "- Set node_type='part' if the user asks for general info, summaries, or variant counts.\n"
    "- For range queries (e.g. 'torque > 200'), set the corresponding _min/_max field.\n"
    "- 'greater than' or '>' → use _min field\n"
    "- 'less than' or '<' → use _max field\n"
    "- Only populate fields explicitly mentioned or clearly implied.\n"
    "- Return null for any field not mentioned.\n\n"
    "Query: {query}\n"
)

def extract_query_intent(user_query: str, llm) -> QueryIntent:
    return llm.structured_predict(
        QueryIntent,
        prompt=INTENT_EXTRACTION_PROMPT,
        query=user_query,
    )

