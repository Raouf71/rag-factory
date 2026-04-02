# RAG-factory
> End-to-end exploration of Retrieval-Augmented Generation pipelines across two real-world domains, comparing frameworks, patterns, and evaluation strategies.

## Domains:

- **Mechanical Parts Catalogs** — structured, technical, terminology-heavy documents
<!-- - **Medical Bills** — semi-structured, entity-rich, compliance-sensitive documents -->

## 🚧 In Progress:

An annotated Jupyter notebook is available for exploring and testing the full pipeline step by step: &nbsp;&nbsp;&nbsp;&nbsp; [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raouf71/rag-factory/blob/master/mechanical_parts_catalogs/notebooks/llamaindex-v2/kg-rag-postgres-v2.ipynb)

- Visualize the pipeline [here](https://github.com/Raouf71/rag-factory/tree/master/mechanical_parts_catalogs)
- RAG-Chatbot with UI (coming soon)
- WIKI page hosting the project documentation

---

## Pipeline

<p align="center">
  <img src="rag-pipeline.png" width="65%">
</p>

---

## RAG Patterns

| Pattern | Description |
|---|---|
| **Naive RAG** | Baseline — embed, index, retrieve, generate |
| **Hybrid RAG** | Dense + sparse (BM25) retrieval with re-ranking |
| **KG-RAG** | Knowledge Graph-augmented retrieval for relational reasoning |
| **Agentic RAG** | Tool-calling agents with multi-step reasoning and answer validation |

### Agentic RAG Stack
- **PydanticAI** — typed tool-calling agents with structured outputs
- **ReAct** — Reasoning + Acting loop for multi-hop queries
- **smolagents** — lightweight agents with minimal boilerplate

Key agentic behaviors implemented:
- Dynamic tool selection (search, filter, compute)
- Multi-step reasoning before answer generation
- Self-verification — agent checks answer against retrieved context before responding

<!-- ---

## Frameworks

Both pipelines are implemented and compared across two frameworks:

- **LlamaIndex** — primary implementation (connectors, query engines, agent tools)
- **LangChain** — parallel implementation for direct comparison (chains, retrievers, agents) -->

---

## Project Structure

```
rag-factory/
├── mechanical_parts_catalogs  
│   ├── config/              
│       └── settings.py           # all env vars, constants, model settings
│   ├── pipeline/ 
│   │   ├── extraction.py         # extract_layer1_fields, extract_layer2_fields
│   │   ├── mapping.py            # normalize_*, build_part_id, map_parts_and_rows
│   │   ├── nodes.py              # build_retrieval_nodes, text/metadata builders
│   │   ├── indexing.py           # build_pgvector_store, index_nodes_with_store
│   │   ├── graph.py              # Neo4j KG build/load
│   │   └── run_pipeline.py       # orchestrates all steps end-to-end
│   ├── retrieval/
│   │   ├── intent.py             # QueryIntent, extract_query_intent
│   │   ├── filters.py            # RANGE_FIELD_MAP, build_filtered_retriever
│   │   └── retriever.py          # build_custom_retriever, reranker toggle
│   ├── app/
│       ├── chatbot.py           # Chatbot UI 
│   ├── data/                    # Raw documents
│   │   └── gears.pdf  
│   └── evaluation/           # RAGAS scoring + comparison notebooks
├── cache/                    # persisted extraction results
├── .env                      # API keys, passwords — never committed
├── .gitignore                # includes .env, cache/, *.ipynb outputs
├── requirements.txt
└── README.md
```