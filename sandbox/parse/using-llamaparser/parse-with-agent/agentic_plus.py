import os
from llama_cloud_services import LlamaParse

api_key="llx-..."
os.environ["LLAMA_CLOUD_API_KEY"] = api_key

base_path = "/home/daghbeji/rag-factory/sandbox/parse/docs/"
# file_name = base_path + "gear_m2.pdf"
# file_name = base_path + "gear___scanned.pdf"
# file_name = base_path + "gear_m1_m1.25.pdf"
file_name = base_path + "lager.pdf"

# Initialize Agentic Plus Mode Parser
agentic_plus_parser = LlamaParse(
    parse_mode="parse_page_with_agent",
    model="anthropic-sonnet-4.0",
    high_res_ocr=True,
    adaptive_long_table=True,
    outlined_table_extraction=True,
    output_tables_as_HTML=False,
    result_type="markdown",
    language="de",
    max_pages=0,
    preserve_very_small_text=True,
    # preserve_layout_alignment_across_pages=True,
    # page_separator="\n\n---\n\n",
    extract_layout=True,
    # # Take screenshot of the page
    take_screenshot=True,
)

print("================================= Agentic Plus Mode Parser initialized")

# parse
result_agentic_plus = agentic_plus_parser.parse(file_name)

# get the llama-index markdown documents
markdown_documents = result_agentic_plus.get_markdown_documents(split_by_page=True)

# get the llama-index text documents
text_documents = result_agentic_plus.get_text_documents(split_by_page=False)

# get the image documents
image_documents = result_agentic_plus.get_image_documents(
    include_screenshot_images=True,
    include_object_images=True,
    # Optional: download the images to a directory
    # (default is to return the image bytes in ImageDocument objects)
    image_download_dir="./images",
)

# access the raw job result_agentic_plus
# Items will vary based on the parser configuration
for page in result_agentic_plus.pages:
    # print("===================================================printing text")
    # print(page.text)
    print("===================================================printing md")
    print(page.md)
    # print("===================================================printing images")
    # print(page.images)
    # print("===================================================printing layout")
    # print(page.layout)
    # print("===================================================printing structured data")
    # print(page.structuredData)

for page in result_agentic_plus.pages:
    # print(page)
    with open("./agentic_plus_page"+str(page.page)+".md", "w", encoding="utf-8") as f:
        f.write(page.md)

# Get markdown nodes
nodes_agentic_plus = result_agentic_plus.get_markdown_nodes(
    split_by_page=True
)
print(f"Number of pages extracted: {len(nodes_agentic_plus)}")

print(nodes_agentic_plus)
# Display sample output from page 1
print("\n=== Sample Output - Page 1 (Agentic Plus Mode) ===")
text = nodes_agentic_plus[0].text
with open("out.txt", "w", encoding="utf-8") as f:
    f.write(text)

# md = nodes_agentic_plus[0].md
# with open("out.md", "w", encoding="utf-8") as f:
#     f.write(text)


