import asyncio
from llama_cloud import AsyncLlamaCloud

LLAMA_KEY="llx-..."

# Upload and parse a document
PDF_PATH = "/home/daghbeji/rag-factory/mechanical-parts-catalogs/data/gear_m2.pdf"

async def parse_document() -> None:
    client = AsyncLlamaCloud(api_key=LLAMA_KEY)
    file_obj = await client.files.create(file=PDF_PATH, purpose="parse")

    result = await client.parsing.parse(
        file_id=file_obj.id,
        # The parsing tier. Options: fast, cost_effective, agentic, agentic_plus,
        tier="agentic",
        # The version of the parsing tier to use. Use 'latest' for the most recent version,
        version="latest",
        # 'expand' controls which result fields are returned in the response.,
        # Without it, only job metadata is returned. Common fields:,
        # - "markdown_full", "text_full": Full document content,
        # - "markdown", "text", "items": Page-level content,
        # - "images_content_metadata": Presigned URLs for images,
        expand=["markdown_full", "text_full"],
        # expand=["text", "markdown", "items"],
    )

    # Access the full document content
    print("Full markdown:")
    print(result.markdown_full)

    print("Full text:")
    print(result.text_full)

    if result.items:
        print(result.items.pages[0])

if __name__ == "__main__":
    asyncio.run(parse_document())