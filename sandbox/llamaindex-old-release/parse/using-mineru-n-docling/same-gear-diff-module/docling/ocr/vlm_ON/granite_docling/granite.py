from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Convert a public arXiv PDF; replace with a local path if preferred.
base_path = "/home/daghbeji/rag-factory/sandbox/parse/docs/"

###### USING SIMPLE DEFAULT VALUES
# - GraniteDocling model
# - Using the transformers framework

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())