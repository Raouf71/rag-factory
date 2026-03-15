import os
from llama_cloud_services import LlamaParse

api_key="llx-..."
os.environ["LLAMA_CLOUD_API_KEY"] = api_key

base_path = "/home/daghbeji/rag-factory/sandbox/parse/docs/"
# file_name = base_path + "gear_m2.pdf"
file_name = base_path + "gear___scanned.pdf"

parser = LlamaParse(
    # Enable pure multimodal parsing
    parse_mode="parse_page_with_lvm",
    vendor_multimodal_model_name="anthropic-sonnet-4.0",
    # Pass in your own API key optionally
    # vendor_multimodal_api_key="fake",
    target_pages="0",
    high_res_ocr=True,
    adaptive_long_table=True,
    outlined_table_extraction=True,
    output_tables_as_HTML=True,
)
result = parser.parse(file_name)
sonnet_nodes = result.get_markdown_nodes(split_by_page=False)

print(sonnet_nodes[0].get_content(metadata_mode="all"))