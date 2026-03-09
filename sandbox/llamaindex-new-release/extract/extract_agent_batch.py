import asyncio
from pydantic import BaseModel, Field
from llama_cloud import LlamaCloud, AsyncLlamaCloud

# Define schema using Pydantic
class PartSchema(BaseModel):
    spur_gear_material: str = Field(description="The material from which the spur gear was manufactured (e.g. Steel, Stainless Steel, Plastics (Polyketon (PK), Polyacetal (POM)), etc.)")
    straight_toothed: bool = Field(description="It indicates whether, the teeth are aligned longitudinally with the shaft, meaning there is no \"helix angle\".")
    angle_of_engagement: int = Field(description="It refers to the angular position, or the arc, during which two gear teeth are in contact and transmitting power. It is often written in Degrees (°).")
    module: float = Field(description="The gear module of a gear represents the ratio of the pitch (distance between teeth) to pi (\\(\\pi \\)), effectively defining how thick a gear tooth is and, consequently, how strong it is.")
    

# Create extraction agent
client = AsyncLlamaCloud(api_key="llx-...")
semaphore = asyncio.Semaphore(5)  # Limit concurrency
agent = client.extraction.extraction_agents.create(
    name="part-parser3",
    data_schema=PartSchema.model_json_schema(),
    config={}
)

async def process_path(file_path: str):
    async with semaphore:
        file_obj = await client.files.create(file=file_path, purpose="extract")
        file_id = file_obj.id

        result = await client.extraction.jobs.extract(
            extraction_agent_id=agent.id,
            file_id=file_id,
        )
    return result

if __name__ == "__main__":
    file_paths = ["/home/daghbeji/rag-factory/mechanical-parts-catalogs/data/gear_m2.pdf",\
                  "/home/daghbeji/rag-factory/mechanical-parts-catalogs/data/mid_size_table.pdf"]
    results = asyncio.gather(*(process_path(path) for path in file_paths))

# ---------------------------------------------------
# # Updating Schemas
# client.extraction.extraction_agents.update(
#     extraction_agent_id=agent.id,
#     data_schema=new_schema,
#     config={},
# )

# # Managing Agents
# agents = client.extraction.extraction_agents.list() # List all agents
# agent = client.extraction.extraction_agents.get(extraction_agent_id="agent_id") # Get specific agent
# client.extraction.extraction_agents.delete(extraction_agent_id="agent_id") # Delete agent