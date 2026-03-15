
import os
from llama_cloud_services import LlamaParse

api_key="llx-..."
os.environ["LLAMA_CLOUD_API_KEY"] = api_key

base_path = "/home/daghbeji/rag-factory/sandbox/parse/docs/"
# file_name = base_path + "gear_m2.pdf"
file_name = base_path + "gear___scanned.pdf"

parser = LlamaParse(
    parse_mode="parse_page_with_agent",
    model="openai-gpt-4-1-mini",
    high_res_ocr=True,
    adaptive_long_table=True,
    outlined_table_extraction=True,
    output_tables_as_HTML=True,
    language="de",
    max_pages=0,
    preserve_very_small_text=True,
    preserve_layout_alignment_across_pages=True,
    page_separator="\n\n---\n\n",
    extract_layout=True,
    # Take screenshot of the page
    take_screenshot=True,
)
result = parser.parse(file_name)

print(parser.parse_mode)
print(parser.model)
print(parser.auto_mode)
print(parser.auto_mode_trigger_on_image_in_page)
print(parser.auto_mode_trigger_on_table_in_page)
print(parser.ignore_document_elements_for_layout_detection)
print(parser.premium_mode)
print(parser.structured_output)
print(parser.bounding_box)

print(result.job_metadata)
#print(result.pages[0].model_dump().keys())
# print(result.pages[0].text[:1000])
text = result.pages[0].md
with open("result_page0.md", "w", encoding="utf-8") as f:
    f.write(text)

print(result.pages[0].images)

# Download images
# single image
image_data = result.get_image_data(result.pages[0].images[0].name)
# save an image to a file
output_path = result.save_image(
    result.pages[0].images[0].name, "./output/parse_with_agent/json_tour_screenshots"
)
# save all images
output_paths = result.save_all_images("./output/parse_with_agent/json_tour_screenshots")


import json
print(result.pages[0].items[0])
print(result.pages[0].items[1])