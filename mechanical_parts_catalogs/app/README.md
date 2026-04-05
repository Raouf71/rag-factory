# Mechanical Part AI-Assistant Frontend

A Streamlit-based RAG interface that provides a multi-tab UI for chat, model settings, pipeline execution, and evaluation previews.

---

## Overview

This frontend is designed to:

<!-- - provide a clean SaaS-style interface for a mechanical-parts RAG system -->
- let users run the backend pipeline step by step
- preview intermediate outputs for transparency
- chat with the AI assistant once the pipeline is ready
- Pattern selector (Naive / Hybrid / KG / Agentic)
- Source citation display with retrieved chunk previews

---

## Main Tabs

| Tab | Purpose |
|---|---|
| Chat | Ask questions about catalog parts and inspect answers with sources |
| Settings | Configure LLM, embedding model, and reranker model |
| Pipeline | Run the RAG pipeline step by step and inspect intermediate outputs |
| Evaluation | (WIP) |

---

## Frontend Structure

### 1. Top Navigation Bar
The app uses a horizontal top navbar with:

- app title on the left
- page navigation on the right:
  - Chat
  - Settings
  - Pipeline
  - Evaluation

This allows users to switch freely between all pages.

---

### 2. Chat Tab
The Chat tab is the main landing page.

#### Left side
- assistant title and description
- current retrieval mode badge
- chat history
- input form:
  - text area
  - Send button
  - Clear Chat button

#### Right side
- retrieval configuration
- quick status panel
- short usage instructions

#### Current chat behavior
- user message is displayed immediately after sending
- assistant response is generated right after
- sources are shown under assistant responses

---

### 3. Settings Tab
The Settings tab contains model configuration controls.

#### Available settings
- Language Model
- Embedding Model
- Reranker Model

This tab is meant for backend-related model selection, not retrieval-time interaction.

---

### 4. Pipeline Tab
The Pipeline tab is used to execute the backend pipeline step by step.

#### Pipeline steps
| Step | Name | Purpose |
|---|---|---|
| Step 1 | Two-Layer Extraction | Extract parent-level and row-level information from the PDF |
| Step 2 | Layer Mapping & Join | Attach row-level extraction results to their parent parts |
| Step 3 | Node Construction | Build parent and child retrieval nodes |
| Step 4 | pgvector Indexing | Index nodes into vector stores |
| Step 5 | Knowledge Graph | Build or load the property graph |
| Step 6 | Build Retrievers | Initialize vector, hybrid, and KG retrievers |

#### Pipeline-side controls
- run / re-run / retry each step
- inspect logs for each step
- upload a PDF source file
- reset all pipeline state

---

## Source Display in Chat

Assistant answers include source pills generated from retrieved nodes.

Each source currently summarizes:

- article number or part ID
- page number when available
- node type

This is based on the retrieved `source_nodes` returned by the chat engine.

---

## TODO:
- clickable chat sources with expandable node content
- richer evaluation dashboard
- better visualization for knowledge graph output
- clearer retrieval trace for each assistant answer
