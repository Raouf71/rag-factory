from llama_cloud_services import LlamaParse

parser = LlamaParse(
    api_key="llx-...", 
    high_res_ocr=True,
    adaptive_long_table=True,
    outlined_table_extraction=True,
    output_tables_as_HTML=True,
    max_pages=0,
    preserve_very_small_text=True,
    preserve_layout_alignment_across_pages=True,
    page_separator="\n\n---\n\n",
    take_screenshot=True,
    extract_layout=True,
    ###################################### Invalid
    # inline_images_in_markdown=True,
    # tier="agentic_plus",
    # precise_bounding_box=True,
    # version="latest",
    # specialized_image_parsing=True,
    
    # Whether to use line level bounding box extraction (experimental)
    line_level_bounding_box=True,
    # If true, the job will use specialized chart parsing (agentic) to parse chart in the document
    specialized_chart_parsing_agentic=True,
    # If true, the job will use specialized image parsing to parse images in the document, cost 3 credits per image in document
    specialized_image_parsing=True,
    # If true, the job will inline images in the markdown output, require precise_bounding_box to be true
    inline_images_in_markdown=True,
    language="de",
    )

# sync
base_path = "/home/daghbeji/rag-factory/sandbox/parse/docs/"
# file_name = base_path + "gear_m2.pdf"
file_name = base_path + "gear___scanned.pdf"

result = parser.parse(file_name)

# get the llama-index markdown documents
markdown_documents = result.get_markdown_documents(split_by_page=True)

# get the llama-index text documents
text_documents = result.get_text_documents(split_by_page=False)

# get the image documents
image_documents = result.get_image_documents(
    include_screenshot_images=True,
    include_object_images=True,
    # Optional: download the images to a directory
    # (default is to return the image bytes in ImageDocument objects)
    image_download_dir="./gear_scanned/images",
)

# access the raw job result
# Items will vary based on the parser configuration
for page in result.pages:
    print("===================================================printing text")
    print(page.text)
    print("===================================================printing md")
    print(page.md)
    print("===================================================printing images")
    print(page.images)
    print("===================================================printing layout")
    print(page.layout)
    print("===================================================printing structured data")
    print(page.structuredData)