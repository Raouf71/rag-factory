from llama_parse import LlamaParse


parser = LlamaParse(
   verbose = True,
   api_key="llx-...", 
   result_type="markdown",  # "markdown" and "text" are available,
   # auto_mode=True,
   use_vendor_multimodal_model = True,
   extract_charts=True,
   auto_mode_trigger_on_image_in_page=True,
   auto_mode_trigger_on_table_in_page=True,
   output_tables_as_HTML=True,
   do_not_unroll_columns=False,
   disable_image_extraction=False,
   language="de",
   disable_ocr=False,
   hide_headers=False,
   hide_footers=False,
   preserve_very_small_text=True,
   skip_diagonal_text=True,   # for invoices
   # target_pages="0,2,7",      # target pages
   page_separator="\n=================\n",
   # page_separator="\n== {pageNumber} ==\n" # Will transform to "\n== 4 ==\n" to separate page 3 and 4.
)

base_path = "/home/daghbeji/rag-factory/sandbox/parse/docs/"
# file_name = base_path + "gear_m2.pdf"
file_name = base_path + "gear___scanned.pdf"
extra_info = {"file_name": file_name}

with open(f"{file_name}", "rb") as f:
   # must provide extra_info with file_name key with passing file object
   documents = parser.load_data(f, extra_info=extra_info)

# with open('output.md', 'w') as f:
   # print(documents, file=f)

# Write the output to a file
with open("output/output_gear_scanned.md", "w", encoding="utf-8") as f:
   for doc in documents:
       f.write(doc.text)