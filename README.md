# RAG-factory
> End-to-end exploration of Retrieval-Augmented Generation pipelines across two real-world domains, comparing frameworks, patterns, and evaluation strategies.

---

## Domains

- **Mechanical Parts Catalogs** — structured, technical, terminology-heavy documents
- **Medical Bills** — semi-structured, entity-rich, compliance-sensitive documents

---

## RAG Pipeline Stages

Each domain goes through the full ingestion-to-answer pipeline:

- **Document Ingestion** — PDF/HTML/CSV parsing and cleaning
- **Chunking** — fixed-size, sentence-aware, and semantic chunking strategies
- **Embedding** — dense vector representations (e.g. `text-embedding-3-small`, `bge-m3`)
- **Indexing** — vector stores (FAISS, Qdrant, Weaviate) + graph stores for KG-RAG
- **Retrieval** — top-k similarity search, hybrid search, graph traversal, agent-driven retrieval

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

---

## Frameworks

Both pipelines are implemented and compared across two frameworks:

- **LlamaIndex** — primary implementation (connectors, query engines, agent tools)
- **LangChain** — parallel implementation for direct comparison (chains, retrievers, agents)

---

## Evaluation — RAGAS

All pipelines are benchmarked using [RAGAS](https://docs.ragas.io/):

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevance** | Does the answer address the question? |
| **Context Precision** | Are retrieved chunks actually useful? |
| **Context Recall** | Did retrieval capture all necessary information? |

Results are logged and compared across patterns and frameworks.

---

## UI Chatbot

A conversational RAG interface built on top of the evaluated pipelines:

- Domain selector (Mechanical / Medical)
- Pattern selector (Naive / Hybrid / KG / Agentic)
- Source citation display with retrieved chunk previews
- RAGAS score display per response (dev mode)

---

## Project Structure

```
rag-factory/
├── rag-domain                 # Mechanical catalogs/Medical bills
│   ├── data/                  # Raw documents
│   ├── pipelines/ 
│   │   ├── ingestion/
│   │   ├── chunking/
│   │   ├── embedding/
│   │   ├── indexing/
│   │   ├── retrieval/
│   ├── patterns/ 
│   │   ├── naive_rag/
│   │   ├── hybrid_rag/
│   │   ├── kg_rag/
│   │   ├── agentic_rag/
│   │   │   ├── pydantic_ai/
│   │   │   ├── reAct/
│   │   │   ├── smolagents/
│   ├── evaluation/            # RAGAS scoring + comparison notebooks
│   ├── ui/                    # Chatbot interface
└── README.md
```

---

## Stack Summary

| Layer | Tools |
|---|---|
| Frameworks | LlamaIndex, LangChain |
| Agentic | PydanticAI, ReAct, smolagents |
| Vector Stores | FAISS, Qdrant, Weaviate |
| Hybrid Search | Elasticsearch (BM25 + knn), PostgreSQL (pgvector + FTS) |
| Graph Store | Neo4j / NetworkX |
| Embeddings | OpenAI, BGE |
| Re-ranking | JINA Reranker |
| Evaluation | RAGAS |
| UI | Streamlit / Gradio |