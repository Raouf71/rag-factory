from llama_cloud import LlamaCloud, AsyncLlamaCloud
import httpx
import re

async def parse_document() -> None:
    # Upload and parse a document
    client = AsyncLlamaCloud(api_key="llx-...")
    PDF_PATH = "/home/daghbeji/rag-factory/mechanical-parts-catalogs/data/gear_m2.pdf"
    file_obj = await client.files.create(file=PDF_PATH, purpose="parse")

    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",

        # Options specific to the input file type, e.g. html, spreadsheet, presentation, etc.
        input_options={},

        # Control the output structure and markdown styling
        output_options={
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": False,
                },
            },
            # Saving images for later retrieval
            "images_to_save": ["screenshot"],
        },

        # Options for controlling how we process the document
        processing_options={
            "ignore": {
                "ignore_diagonal_text": True,
            },
            "ocr_parameters": {
                "languages": ["fr"]
            }
        },

        # Parsed content to include in the returned response
        expand=["text", "markdown", "items", "images_content_metadata"],
    )

    print(result.markdown.pages[0].markdown)
    print(result.text.pages[0].text)

    # Iterate over page items to find tables
    for page in result.items.pages:
        for item in page.items:
            if isinstance(item, ItemsPageStructuredResultPageItemTableItem):
                print(f"Table found on page {page.page_number} with {len(item.rows)} rows and {item.b_box} location")

    def is_page_screenshot(image_name: str) -> bool:
        return re.match(r"^page_(\d+)\.jpg$", image_name) is not None

    # Iterate over results looking for page screenshots
    for image in result.images_content_metadata.images:
        if image.presigned_url is None or not is_page_screenshot(image.filename):
            continue

        print(f"Downloading {image.filename}, {image.size_bytes} bytes")
        with open(f"{image.filename}", "wb") as img_file:
            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(image.presigned_url)
                img_file.write(response.content)

if __name__ == "__main__":
    asyncio.run(parse_document())