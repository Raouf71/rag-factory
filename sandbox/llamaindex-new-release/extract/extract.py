from llama_cloud_services import LlamaExtract
from llama_cloud import ExtractConfig
from pydantic import BaseModel, Field

class PartSchema(BaseModel):
    spur_gear_material: str = Field(description="The material from which the spur gear was manufactured (e.g. Steel, Stainless Steel, Plastics (Polyketon (PK), Polyacetal (POM)), etc.)")
    straight_toothed: bool = Field(description="It indicates whether, the teeth are aligned longitudinally with the shaft, meaning there is no \"helix angle\".")
    angle_of_engagement: int = Field(description="It refers to the angular position, or the arc, during which two gear teeth are in contact and transmitting power. It is often written in Degrees (°).")
    module: float = Field(description="The gear module of a gear represents the ratio of the pitch (distance between teeth) to pi (\\(\\pi \\)), effectively defining how thick a gear tooth is and, consequently, how strong it is.")
    
extractor = LlamaExtract(api_key="llx-...")

# Data Schema and Config
json_data = {
    "dataSchema": PartSchema,
    "config": {
        "priority": None,
        "extraction_target": "PER_PAGE",
        "extraction_mode": "PREMIUM",
        "parse_model": "anthropic-haiku-4.5",
        "extract_model": "openai-gpt-4-1",
        "multimodal_fast_mode": False,
        "system_prompt": "You are an expert at extracting specifications of spur gears from catalog documents",
        "use_reasoning": False,
        "citation_bbox": False,
        "confidence_scores": False,
        "chunk_mode": "PAGE",
        "high_resolution_mode": False,
        "invalidate_cache": False,
        "num_pages_context": None,
        "page_range": None,
        "cite_sources": True
    }
}

try:
    # Use schema and config from playground
    data_schema = json_data["dataSchema"]
    config = ExtractConfig(**json_data["config"])

    # Extract data directly from document - no agent needed!
    file = "/home/daghbeji/rag-factory/mechanical-parts-catalogs/data/gear_m2.pdf" 
    result = extractor.extract(data_schema, config, file)
    print(result.data)
    
except Exception as e:
    print(f"Error: {e}")