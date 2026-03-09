import asyncio
from pydantic import BaseModel, Field
from llama_cloud import LlamaCloud, AsyncLlamaCloud

# Define schema using Pydantic
class PartSchema(BaseModel):
    spur_gear_material: str = Field(description="The material from which the spur gear was manufactured (e.g. Steel, Stainless Steel, Plastics (Polyketon (PK), Polyacetal (POM)), etc.)")
    straight_toothed: bool = Field(description="It indicates whether, the teeth are aligned longitudinally with the shaft, meaning there is no \"helix angle\".")
    angle_of_engagement: int = Field(description="It refers to the angular position, or the arc, during which two gear teeth are in contact and transmitting power. It is often written in Degrees (°).")
    module: float = Field(description="The gear module of a gear represents the ratio of the pitch (distance between teeth) to pi (\\(\\pi \\)), effectively defining how thick a gear tooth is and, consequently, how strong it is.")
    

async def extract_from_document():
    client = LlamaCloud(api_key="llx-...")
    
    # Create extraction agent
    agent = client.extraction.extraction_agents.create(
        name="part-parser3",
        data_schema=PartSchema.model_json_schema(),
        config={}
    )

    # Upload a file to extract from
    PDF_PATH = "/home/daghbeji/rag-factory/mechanical-parts-catalogs/data/gear_m2.pdf"
    file_obj = client.files.create(file=PDF_PATH, purpose="extract")
    file_id = file_obj.id

    # Extract data from document
    result = client.extraction.jobs.extract(
        extraction_agent_id=agent.id,
        file_id=file_id,
    )
    print(result.data)

if __name__ == "__main__":
    asyncio.run(extract_from_document())