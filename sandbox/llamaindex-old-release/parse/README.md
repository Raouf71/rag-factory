Testing and benchmarking different parsers on different table variations
---

- [x] Testing two separate tables within same page([2_tabs_1_page](https://github.com/Raouf71/rag-factory/tree/master/sandbox/parse/docs/2_tabs_1_page.pdf)):
    - Testing with **Mineru**:
        - It led to ❌ false headers/noisy in the table(check the output [here](https://github.com/Raouf71/rag-factory/tree/master/sandbox/parse/using-mineru-n-docling/two_tabs_1_page/mineru/no-vlm/auto/2_tabs_1_page.md) as markdown file). 
        - ⚠️ Mineru (a lot of other parsers too) struggles with subscripts, superscripts and special characters like diameter symbol.
    - Testing with **Docling**:
        - perfect output
        - ALL ROWS HEADERS ARE CORRECT ✅
        - - Check output [here](https://github.com/Raouf71/rag-factory/blob/master/sandbox/parse/using-mineru-n-docling/two_tabs_1_page/docling/no-ocr/2_tabs_1_page.md)

- [x] Testing two pages with same gear type but different module([same_gear_different_module](https://github.com/Raouf71/rag-factory/tree/master/sandbox/parse/docs/same_gear_different_module.pdf)):
    - When both pages were processed together via ```await rag.process_document_complete()```:
        - Both **Mineru** and **Docling** led to ❌ false/noisy headers in the table
    - When both pages were processed seperately via ```await rag.await rag.process_folder_complete()```:
        - It worked
    - Check both outputs [here](https://github.com/Raouf71/rag-factory/tree/master/sandbox/parse/using-mineru-n-docling/same-gear-diff-module)

- [x] Testing very complex table layout ([lager](https://github.com/Raouf71/rag-factory/tree/master/sandbox/parse/docs/lager.pdf)):
    - Same noisy output of parsers: 
        - headers being extracted incorrectly
        - false values
        - columns/rows getting mixed up
    - Check output [here](https://github.com/Raouf71/rag-factory/blob/master/sandbox/parse/using-mineru-n-docling/lager/auto/lager.md)


## Summary

|                Document               |      gear_m2     |                                   same_gear_diff_module                                   |  two_tabs_1_page |                                          lager                                       |
|:-------------------------------------:|:----------------:|:-----------------------------------------------------------------------------------------:|:----------------:|:------------------------------------------------------------------------------------:|
| Mineru                                | ✅ Perfect output | - **1st table**: noisy headers, correct table body <br> - **2nd table:**: correct headers+body | ❌ Noisy headers  | ❌ Noisy headers                                                                      |
| Docling                               | ✅ Perfect output | - **1st table**: noisy headers, correct table body <br> - **2nd table:**: correct headers+body | ✅ Perfect output | ❌ Noisy headers                                                                      |
| Deepseek-ocr (**runs only on Linux**) | ✅ Perfect output | ✅ Perfect output                                                                          | ✅ Perfect output | ✅Table bodies are correct ❌Table headers false, rows/columns  got merged incorrectly |

## Testing LlamaParse on same docs

LlamaParse has 3 parse modes:
1) **Cost-effective mode**: ``parse_mode="parse_page_with_llm"``
2) **Agentic mode (Default)**: ``parse_mode="parse_page_with_agent"`` (with gpt-4.1-mini)
3) **Agentic plus mode**: ``parse_mode="parse_page_with_agent`` (with anthropic-sonnet-4.0)"

- Check parsing results [here](https://github.com/Raouf71/rag-factory/tree/master/sandbox/parse/using-llamaparser)
