import os
from llama_cloud_services import LlamaParse

api_key="llx-..."
os.environ["LLAMA_CLOUD_API_KEY"] = api_key

base_path = "/home/daghbeji/rag-factory/sandbox/parse/docs/"
# file_name = base_path + "gear_m2.pdf"
file_name = base_path + "gear___scanned.pdf"

parser_gpt4o = LlamaParse(
    # Enable pure multimodal parsing
    parse_mode="parse_page_with_lvm",
    vendor_multimodal_model_name="openai-gpt-4-1-mini",
    # Pass in your own API key optionally
    # vendor_multimodal_api_key="fake",
    target_pages="0",
    high_res_ocr=True,
    adaptive_long_table=True,
    outlined_table_extraction=True,
    output_tables_as_HTML=True,
)
result = parser_gpt4o.parse(file_name)
gpt_nodes = result.get_markdown_nodes(split_by_page=False)

print(gpt_nodes[0].get_content(metadata_mode="all"))