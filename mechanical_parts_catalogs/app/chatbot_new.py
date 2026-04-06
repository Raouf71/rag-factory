import asyncio
import logging
import sys
import os
from typing import Optional

import streamlit as st
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pipeline.extraction import extract_layer1_fields, extract_layer2_fields
from pipeline.mapping import (
    attach_part_id_to_layer1_parts,
    attach_part_id_to_layer2_rows,
    map_parts_and_rows,
    log_mapping_diagnostics,
)
from pipeline.nodes import build_retrieval_nodes
from pipeline.indexing import reset_pgvector_schema, build_pgvector_store, index_nodes_with_store
from pipeline.graph import get_or_build_property_graph_index
from pipeline.schemas import PartSchema, TableRowSchema
from retrieval.retriever import (
    build_basic_vector_retriever,
    build_basic_hybrid_retriever,
    build_kg_retriever,
    build_custom_retriever,
)
from retrieval.filters import build_filtered_retriever
from retrieval.intent import extract_query_intent
from retrieval.agent import build_agent_retriever

from config.settings import (
    DEEPSEEK_KEY,
    OPENAI_KEY,
    SYSTEM_PROMPT_L1,
    SYSTEM_PROMPT_L2,
    PDF_PATH,
)
from streamlit_agraph import agraph, Node, Edge, Config
import io
import contextlib
import traceback
import pandas as pd

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Mechanical Part AI-Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------
# Custom CSS - SaaS / Startup style
# -----------------------------------------------------------------------
st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.5rem;
        max-width: 1450px;
    }

    header[data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    /* ---------- Navbar ---------- */
    .top-nav-shell {
        border: 1px solid #e4e4e7;
        border-radius: 20px;
        background: #ffffff;
        box-shadow: 0 10px 30px rgba(24, 24, 27, 0.05);
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.2rem;
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(139, 92, 246, 0.22);
    }

    .brand-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #18181b;
        line-height: 1.15;
    }

    .brand-subtitle {
        font-size: 0.88rem;
        color: #71717a;
        margin-top: 2px;
    }

    /* ---------- Generic cards ---------- */
    # .glass-card {
    #     background: #ffffff;
    #     border: 1px solid #e4e4e7;
    #     border-radius: 22px;
    #     box-shadow: 0 10px 30px rgba(24, 24, 27, 0.05);
    #     padding: 1.2rem;
    # }

    .subtle-card {
        background: #fafafa;
        border: 1px solid #e4e4e7;
        border-radius: 18px;
        padding: 1rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #18181b;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        font-size: 0.93rem;
        color: #71717a;
        margin-bottom: 1rem;
    }

    .active-pill {
        display: inline-block;
        background: #f5f3ff;
        color: #7c3aed;
        border: 1px solid #ddd6fe;
        border-radius: 999px;
        padding: 0.32rem 0.78rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .mode-badge {
        display: inline-block;
        background: #fafafa;
        border: 1px solid #e4e4e7;
        color: #18181b;
        border-radius: 999px;
        padding: 0.42rem 0.85rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 14px !important;
        border: 1px solid #e4e4e7 !important;
        background: #ffffff !important;
        color: #18181b !important;
        font-weight: 600 !important;
        transition: 0.2s ease !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: #c4b5fd !important;
        color: #7c3aed !important;
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.12);
    }

    /* ---------- Step cards ---------- */
    .step-card {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.65rem;
    }

    .step-card.done { border-left: 5px solid #22c55e; }
    .step-card.running { border-left: 5px solid #8b5cf6; }
    .step-card.error { border-left: 5px solid #ef4444; }
    .step-card.idle { border-left: 5px solid #d4d4d8; }

    .step-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .step-name {
        font-size: 1rem;
        font-weight: 700;
        color: #18181b;
        margin-top: 0.18rem;
    }

    .step-stat {
        color: #71717a;
        font-size: 0.84rem;
        margin-top: 0.3rem;
        line-height: 1.45;
    }

    .log-box {
        background: #fafafa;
        border: 1px solid #e4e4e7;
        border-radius: 14px;
        padding: 0.8rem 0.95rem;
        font-size: 0.78rem;
        color: #3f3f46;
        max-height: 180px;
        overflow-y: auto;
        white-space: pre-wrap;
        margin-top: 0.45rem;
    }

    /* ---------- Chat ---------- */
    # .chat-shell {
    #     background: #ffffff;
    #     border: 1px solid #e4e4e7;
    #     border-radius: 24px;
    #     box-shadow: 0 10px 30px rgba(24, 24, 27, 0.05);
    #     padding: 1rem;
    #     min-height: 420px;
    # }

    .msg-row {
        display: flex;
        width: 100%;
        margin-bottom: 0.9rem;
    }

    .msg-row.user {
        justify-content: flex-end;
    }

    .msg-row.assistant {
        justify-content: flex-start;
    }

    .msg-bubble {
        max-width: 74%;
        padding: 0.95rem 1rem;
        border-radius: 18px;
        font-size: 0.97rem;
        line-height: 1.55;
        word-wrap: break-word;
        border: 1px solid #e4e4e7;
    }

    .msg-bubble.user {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        border: none;
        border-bottom-right-radius: 6px;
        box-shadow: 0 10px 24px rgba(139, 92, 246, 0.18);
    }

    .msg-bubble.assistant {
        background: #fafafa;
        color: #18181b;
        border-bottom-left-radius: 6px;
    }

    .msg-meta {
        font-size: 0.72rem;
        color: #71717a;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }

    .sources-wrap {
        margin-top: 0.55rem;
    }

    .sources-label {
        font-size: 0.7rem;
        color: #71717a;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .source-pill {
        display: inline-block;
        background: #f5f3ff;
        border: 1px solid #ddd6fe;
        color: #7c3aed;
        font-size: 0.75rem;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }

    # .input-shell {
    #     margin-top: 1rem;
    #     background: #fafafa;
    #     border: 1px solid #e4e4e7;
    #     border-radius: 18px;
    #     padding: 1rem;
    # }

    .placeholder-card {
        border: 1px dashed #d4d4d8;
        background: #fafafa;
        border-radius: 18px;
        padding: 2rem 1.2rem;
        color: #71717a;
        text-align: center;
    }

    textarea {
        border-radius: 14px !important;
    }

    div[data-baseweb="select"] > div {
            border-radius: 14px !important;
        }

        div[data-testid="stExpander"] details summary p {
        display: inline-block;
        background: #4e524e;
        border: 1px solid #2d4a2d;
        color: #6abf6a;
        font-size: 0.85rem;
        padding: 2px 10px;
        border-radius: 999px;
        margin: 0;
        font-weight: 600;
    }

    div[data-testid="stExpander"] details summary {
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }

    div[data-testid="stExpander"] details summary p:empty {
        display: none;
    }

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# Logging capture helper
# -----------------------------------------------------------------------
class LogCapture(logging.Handler):
    """Captures log records into a list for display in the UI."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))

    def get_logs(self) -> str:
        return "\n".join(self.records)

def capture_logs() -> LogCapture:
    handler = LogCapture()
    handler.setFormatter(logging.Formatter("%(levelname)s  %(name)s — %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler

def release_logs(handler: LogCapture):
    logging.getLogger().removeHandler(handler)

# -----------------------------------------------------------------------
# Session state init
# -----------------------------------------------------------------------
STEP_KEYS = [
    "step1_extraction",
    "step2_mapping",
    "step3_nodes",
    "step4_indexing",
    "step5_kg",
    "step6_retrievers",
]

def init_session_state():
    defaults = {
        "active_tab": "Chat",
        "extraction_result_layer1": None,
        "extraction_result_layer2": None,
        "mapped_parts": None,
        "part_nodes": None,
        "row_nodes": None,
        "all_nodes": None,
        "basic_index": None,
        "hybrid_index": None,
        "property_graph_index": None,
        "vector_retriever": None,
        "hybrid_retriever": None,
        "kg_retriever": None,
        **{k: "idle" for k in STEP_KEYS},
        **{f"{k}_log": "" for k in STEP_KEYS},
        **{f"{k}_stat": "" for k in STEP_KEYS},
        "chat_history": [],
        "memory": ChatMemoryBuffer.from_defaults(token_limit=4096),
        "retrieval_mode": "Hybrid + Metadata Filtering",
        "use_reranker": False,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        "llm_choice": "DeepSeek (deepseek-chat)",
        "embed_choice": "OpenAI (text-embedding-3-small)",
        "models_initialized": False,
        "uploaded_pdf_path": None,
        "uploaded_pdf_name": None,
        "pending_user_prompt": None,
        "generate_response_now": False,
        "retrieval_top_k": 15,
        "rerank_top_k": 10,
        "dev_console_code": """# Example:
        print(f"Layer 1 parts extracted: {len(st.session_state.extraction_result_layer1.data)}")
        print(f"Layer 2 rows extracted:  {len(st.session_state.extraction_result_layer2.data)}")
        """,
        "dev_console_output": "",
        "benchmark_query": "",
        "benchmark_results": {},
        "agent_retriever": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# -----------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------
def setup_models():
    llm_choice = st.session_state.llm_choice
    embed_choice = st.session_state.embed_choice

    if llm_choice == "DeepSeek (deepseek-chat)":
        Settings.llm = DeepSeek(
            model="deepseek-chat",
            api_key=DEEPSEEK_KEY,
            temperature=0.1,
        )
    elif llm_choice == "OpenAI (gpt-4o-mini)":
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_KEY,
            temperature=0.1,
        )
    elif llm_choice == "OpenAI (gpt-4o)":
        Settings.llm = OpenAI(
            model="gpt-4o",
            api_key=OPENAI_KEY,
            temperature=0.1,
        )

    if embed_choice == "OpenAI (text-embedding-3-small)":
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=OPENAI_KEY,
        )
    elif embed_choice == "OpenAI (text-embedding-3-large)":
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=OPENAI_KEY,
        )

if not st.session_state.models_initialized:
    setup_models()
    st.session_state.models_initialized = True

# -----------------------------------------------------------------------
# Pipeline steps
# -----------------------------------------------------------------------
def step_icon(status: str) -> str:
    return {"idle": "○", "running": "◌", "done": "●", "error": "✕"}.get(status, "○")

def step_css_class(status: str) -> str:
    return {
        "idle": "idle",
        "running": "running",
        "done": "done",
        "error": "error",
    }.get(status, "idle")

def get_active_pdf_path() -> str:
    return st.session_state.uploaded_pdf_path 

def run_step1():
    setup_models()
    handler = capture_logs()
    st.session_state.step1_extraction = "running"
    try:
        active_pdf_path = get_active_pdf_path()
        r1 = asyncio.run(extract_layer1_fields(active_pdf_path, SYSTEM_PROMPT_L1, PartSchema))
        r2 = asyncio.run(extract_layer2_fields(active_pdf_path, SYSTEM_PROMPT_L2, TableRowSchema))
        # r1 = asyncio.run(extract_layer1_fields(PDF_PATH, SYSTEM_PROMPT_L1, PartSchema))
        # r2 = asyncio.run(extract_layer2_fields(PDF_PATH, SYSTEM_PROMPT_L2, TableRowSchema))
        st.session_state.extraction_result_layer1 = r1
        st.session_state.extraction_result_layer2 = r2
        st.session_state.step1_extraction = "done"
        st.session_state.step1_extraction_stat = (
            f"Layer 1: {len(r1.data)} record(s) · Layer 2: {len(r2.data)} row(s)"
        )
    except Exception as e:
        st.session_state.step1_extraction = "error"
        st.session_state.step1_extraction_stat = str(e)
    finally:
        release_logs(handler)
        st.session_state.step1_extraction_log = handler.get_logs()

def run_step2():
    handler = capture_logs()
    st.session_state.step2_mapping = "running"
    try:
        parts = attach_part_id_to_layer1_parts(st.session_state.extraction_result_layer1)
        rows = attach_part_id_to_layer2_rows(st.session_state.extraction_result_layer2)
        mapped = map_parts_and_rows(parts, rows)
        log_mapping_diagnostics(mapped)
        st.session_state.mapped_parts = mapped
        st.session_state.step2_mapping = "done"
        st.session_state.step2_mapping_stat = (
            f"{len(mapped.parts)} part(s) joined · {mapped.orphaned_count} orphaned row(s)"
        )
    except Exception as e:
        st.session_state.step2_mapping = "error"
        st.session_state.step2_mapping_stat = str(e)
    finally:
        release_logs(handler)
        st.session_state.step2_mapping_log = handler.get_logs()

def run_step3():
    handler = capture_logs()
    st.session_state.step3_nodes = "running"
    try:
        part_nodes, row_nodes = build_retrieval_nodes(st.session_state.mapped_parts.parts)
        st.session_state.part_nodes = part_nodes
        st.session_state.row_nodes = row_nodes
        st.session_state.all_nodes = part_nodes + row_nodes
        st.session_state.step3_nodes = "done"
        st.session_state.step3_nodes_stat = (
            f"{len(part_nodes)} parent node(s) · {len(row_nodes)} child node(s)"
        )
    except Exception as e:
        st.session_state.step3_nodes = "error"
        st.session_state.step3_nodes_stat = str(e)
    finally:
        release_logs(handler)
        st.session_state.step3_nodes_log = handler.get_logs()

def run_step4():
    handler = capture_logs()
    st.session_state.step4_indexing = "running"
    try:
        reset_pgvector_schema("public", True)
        basic_store = build_pgvector_store("basic")
        hybrid_store = build_pgvector_store("hybrid")
        basic_index = index_nodes_with_store(st.session_state.all_nodes, basic_store)
        hybrid_index = index_nodes_with_store(st.session_state.all_nodes, hybrid_store)
        st.session_state.basic_index = basic_index
        st.session_state.hybrid_index = hybrid_index
        st.session_state.step4_indexing = "done"
        st.session_state.step4_indexing_stat = (
            f"{len(st.session_state.all_nodes)} node(s) indexed into pgvector"
        )
    except Exception as e:
        st.session_state.step4_indexing = "error"
        st.session_state.step4_indexing_stat = str(e)
    finally:
        release_logs(handler)
        st.session_state.step4_indexing_log = handler.get_logs()

def run_step5():
    handler = capture_logs()
    st.session_state.step5_kg = "running"
    try:
        pg_index = get_or_build_property_graph_index(all_nodes=st.session_state.all_nodes)
        st.session_state.property_graph_index = pg_index
        st.session_state.step5_kg = "done"
        st.session_state.step5_kg_stat = "Knowledge graph ready (Neo4j)"
    except Exception as e:
        st.session_state.step5_kg = "error"
        st.session_state.step5_kg_stat = str(e)
    finally:
        release_logs(handler)
        st.session_state.step5_kg_log = handler.get_logs()

def run_step6():
    handler = capture_logs()
    st.session_state.step6_retrievers = "running"
    try:
        st.session_state.vector_retriever = build_basic_vector_retriever(
            basic_index=st.session_state.basic_index
        )
        st.session_state.hybrid_retriever = build_basic_hybrid_retriever(
            hybrid_index=st.session_state.hybrid_index
        )
        if st.session_state.property_graph_index is not None:
            st.session_state.kg_retriever = build_kg_retriever(
                property_graph_index=st.session_state.property_graph_index,
                similarity_top_k=8,
                path_depth=3,
                include_text=True,
            )
        else:
            st.session_state.kg_retriever = None

        # Agent retriever
        try:
            st.session_state.agent_retriever = build_agent_retriever(
                basic_index=st.session_state.basic_index,
                hybrid_index=st.session_state.hybrid_index,
                property_graph_index=st.session_state.property_graph_index,
                similarity_top_k=st.session_state.retrieval_top_k,
            )
        except Exception as e:
            logging.warning(f"Agent retriever unavailable: {e}")
            st.session_state.agent_retriever = None


        st.session_state.step6_retrievers = "done"
        st.session_state.step6_retrievers_stat = (
            "Vector · Hybrid retrievers ready"
            if st.session_state.kg_retriever is None
            else "Vector · Hybrid · KG retrievers ready"
        )
    except Exception as e:
        st.session_state.step6_retrievers = "error"
        st.session_state.step6_retrievers_stat = str(e)
    finally:
        release_logs(handler)
        st.session_state.step6_retrievers_log = handler.get_logs()

STEPS = [
    {
        "key": "step1_extraction",
        "label": "Step 1",
        "name": "Two-Layer Extraction",
        "run": run_step1,
        "requires": None,
    },
    {
        "key": "step2_mapping",
        "label": "Step 2",
        "name": "Layer Mapping & Join",
        "run": run_step2,
        "requires": "step1_extraction",
    },
    {
        "key": "step3_nodes",
        "label": "Step 3",
        "name": "Node Construction",
        "run": run_step3,
        "requires": "step2_mapping",
    },
    {
        "key": "step4_indexing",
        "label": "Step 4",
        "name": "pgvector Indexing",
        "run": run_step4,
        "requires": "step3_nodes",
    },
    {
        "key": "step5_kg",
        "label": "Step 5",
        "name": "Knowledge Graph (Optional)",
        "run": run_step5,
        "requires": "step3_nodes",
    },
    {
        "key": "step6_retrievers",
        "label": "Step 6",
        "name": "Build Retrievers",
        "run": run_step6,
        "requires": "step4_indexing",
    },
]

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def pipeline_ready() -> bool:
    return st.session_state.step6_retrievers == "done"

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

def reset_all():
    for k in STEP_KEYS:
        st.session_state[k] = "idle"
        st.session_state[f"{k}_log"] = ""
        st.session_state[f"{k}_stat"] = ""
    st.session_state.extraction_result_layer1 = None
    st.session_state.extraction_result_layer2 = None
    st.session_state.mapped_parts = None
    st.session_state.part_nodes = None
    st.session_state.row_nodes = None
    st.session_state.all_nodes = None
    st.session_state.basic_index = None
    st.session_state.hybrid_index = None
    st.session_state.property_graph_index = None
    st.session_state.vector_retriever = None
    st.session_state.hybrid_retriever = None
    st.session_state.kg_retriever = None
    clear_chat()

def switch_tab(tab_name: str):
    st.session_state.active_tab = tab_name

def extraction_result_to_df(result):
    if result is None:
        return None

    data = getattr(result, "data", None)
    if not data:
        return None

    rows = []
    for item in data:
        if hasattr(item, "model_dump"):
            rows.append(item.model_dump())
        elif hasattr(item, "dict"):
            rows.append(item.dict())
        elif isinstance(item, dict):
            rows.append(item)
        else:
            rows.append(vars(item))

    import pandas as pd
    return pd.DataFrame(rows)

def nodes_to_preview_df(nodes, max_text_length=160):
    if not nodes:
        return None

    rows = []
    for node in nodes:
        metadata = getattr(node, "metadata", {}) or {}
        text = ""

        if hasattr(node, "get_content"):
            try:
                text = node.get_content()
            except Exception:
                text = getattr(node, "text", "")
        else:
            text = getattr(node, "text", "")

        text = text or ""
        text_preview = text[:max_text_length] + ("..." if len(text) > max_text_length else "")

        rows.append({
            "node_id": getattr(node, "node_id", ""),
            "part_id": metadata.get("part_id", ""),
            "node_type": metadata.get("node_type", ""),
            "art_nr": metadata.get("art_nr", ""),
            "page_number": metadata.get("page_number", ""),
            "text_preview": text_preview,
        })

    import pandas as pd
    return pd.DataFrame(rows)

def build_step3_graph(part_nodes, row_nodes, max_children_per_parent=50):
    nodes = []
    edges = []
    detail_map = {}

    parent_ids_added = set()
    child_ids_added = set()
    child_count_by_parent = {}

    # Parent nodes
    for node in part_nodes or []:
        metadata = getattr(node, "metadata", {}) or {}
        node_id = getattr(node, "node_id", "")

        label = metadata.get("part_id") or node_id
        nodes.append(
            Node(
                id=node_id,
                label=label,
                size=28,
                shape="box",
            )
        )
        parent_ids_added.add(node_id)

        text = ""
        if hasattr(node, "get_content"):
            try:
                text = node.get_content()
            except Exception:
                text = getattr(node, "text", "")
        else:
            text = getattr(node, "text", "")

        detail_map[node_id] = {
            "node_id": node_id,
            "part_id": metadata.get("part_id", ""),
            "node_type": metadata.get("node_type", ""),
            # "art_nr": metadata.get("art_nr", ""),
            "page_number": metadata.get("page_number", ""),
            "text_preview": (text[:300] + "...") if len(text) > 300 else text,
        }

    # Child nodes + edges
    for node in row_nodes or []:
        metadata = getattr(node, "metadata", {}) or {}
        child_id = getattr(node, "node_id", "")
        part_id = metadata.get("part_id", "")
        node_type = metadata.get("node_type", "child")

        parent_node_id = None
        for pnode in part_nodes or []:
            pmeta = getattr(pnode, "metadata", {}) or {}
            if pmeta.get("part_id", "") == part_id:
                parent_node_id = getattr(pnode, "node_id", "")
                break

        if not parent_node_id or parent_node_id not in parent_ids_added:
            continue

        child_count_by_parent[parent_node_id] = child_count_by_parent.get(parent_node_id, 0) + 1
        if child_count_by_parent[parent_node_id] > max_children_per_parent:
            continue

        child_label = child_id.split("::")[-1] if "::" in child_id else child_id
        nodes.append(
            Node(
                id=child_id,
                label=child_label,
                # shape="dot",
                shape="ellipse",
                # shape="box",
                size=max(18, min(35, 8 + len(child_label))),
            )
        )
        child_ids_added.add(child_id)

        edges.append(
            Edge(
                source=parent_node_id,
                target=child_id,
            )
        )

        text = ""
        if hasattr(node, "get_content"):
            try:
                text = node.get_content()
            except Exception:
                text = getattr(node, "text", "")
        else:
            text = getattr(node, "text", "")

        detail_map[child_id] = {
            "node_id": child_id,
            "part_id": metadata.get("part_id", ""),
            "node_type": metadata.get("node_type", ""),
            "art_nr": metadata.get("art_nr", ""),
            "page_number": metadata.get("page_number", ""),
            "text_preview": (text[:300] + "...") if len(text) > 300 else text,
        }

    config = Config(
        width="100%",
        height=520,
        directed=False,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#ddd6fe",
        collapsible=False,
    )

    return nodes, edges, config, detail_map

def mapped_parts_to_preview(mapped_parts):
    if mapped_parts is None or not hasattr(mapped_parts, "parts"):
        return []

    preview_items = []

    for part_key, part in mapped_parts.parts.items():
        if hasattr(part, "model_dump"):
            part_dict = part.model_dump()
        elif hasattr(part, "dict"):
            part_dict = part.dict()
        else:
            part_dict = {
                "part_id": getattr(part, "part_id", part_key),
                "page_number": getattr(part, "page_number", None),
                "dimension_rows": [
                    row.model_dump() if hasattr(row, "model_dump")
                    else row.dict() if hasattr(row, "dict")
                    else vars(row)
                    for row in getattr(part, "dimension_rows", [])
                ],
            }

        rows = part_dict.get("dimension_rows", []) or []

        parent_summary = {
            k: v for k, v in part_dict.items()
            if k != "dimension_rows"
        }

        preview_items.append(
            {
                "part_id": parent_summary.get("part_id", part_key),
                "parent_summary": parent_summary,
                "rows": rows,
            }
        )

    return preview_items

def run_query_for_mode(user_query: str, mode: str) -> tuple[str, list[dict]]:
    if mode == "Vector Only":
        retriever = build_basic_vector_retriever(
            basic_index=st.session_state.basic_index,
            similarity_top_k=st.session_state.benchmark_retrieval_top_k,
        )
    elif mode == "Hybrid Only":
        retriever = build_basic_hybrid_retriever(
            hybrid_index=st.session_state.hybrid_index,
            similarity_top_k=st.session_state.benchmark_retrieval_top_k,
        )
    elif mode == "Hybrid + Metadata Filtering":
        intent = extract_query_intent(user_query, Settings.llm)
        retriever = build_filtered_retriever(
            intent=intent,
            basic_index=st.session_state.basic_index,
            hybrid_index=st.session_state.hybrid_index,
            use_hybrid=True,
            similarity_top_k=st.session_state.benchmark_retrieval_top_k,
        )
    elif mode == "Agent Mode":
        agent = st.session_state.agent_retriever

        if agent is None:
            return "⚠️ Agent Mode is not ready yet.", []

        async def _run_agent_query():
            response = await agent.run(user_query)
            return response

        response = asyncio.run(_run_agent_query())
        return str(response), []
    else:
        retriever = build_custom_retriever(
            query=user_query,
            basic_index=st.session_state.basic_index,
            hybrid_index=st.session_state.hybrid_index,
            kg_retriever=st.session_state.kg_retriever,
            # similarity_top_k=st.session_state.retrieval_top_k,
        )

    node_postprocessors = []
    if st.session_state.use_reranker:
        try:
            reranker = SentenceTransformerRerank(
                model=st.session_state.reranker_model,
                top_n=st.session_state.benchmark_rerank_top_k,
            )
            node_postprocessors.append(reranker)
        except Exception as e:
            logging.warning(f"Reranker unavailable: {e}")

    temp_memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=temp_memory,
        llm=Settings.llm,
        node_postprocessors=node_postprocessors,
        context_prompt=(
            "You are an expert in mechanical gear catalogs.\n"
            "Answer using only the catalog data provided below.\n"
            "If the answer is not in the context, say so explicitly.\n\n"
            "Context:\n{context_str}\n\nChat history:\n{chat_history}"
        ),
    )

    response = chat_engine.chat(user_query)
    source_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
    return str(response), build_sources_data(source_nodes)

def benchmark_all_modes(user_query: str) -> dict:
    import time

    modes = [
        "Vector Only",
        "Hybrid Only",
        "Hybrid + Metadata Filtering",
    ]

    if st.session_state.kg_retriever is not None:
        modes.append("KG + Hybrid + Metadata Filtering")

    if st.session_state.agent_retriever is not None:
        modes.append("Agent Mode")
        results = {}

    for mode in modes:
        started = time.perf_counter()
        try:
            answer, sources = run_query_for_mode(user_query, mode)
            elapsed = time.perf_counter() - started
            results[mode] = {
                "answer": answer,
                "sources": sources,
                "latency_sec": elapsed,
                "error": None,
            }
        except Exception as e:
            elapsed = time.perf_counter() - started
            results[mode] = {
                "answer": "",
                "sources": [],
                "latency_sec": elapsed,
                "error": str(e),
            }

    return results

# -----------------------------------------------------------------------
# Source formatting
# -----------------------------------------------------------------------
def format_sources(nodes) -> str:
    seen, pills = set(), []
    for node in nodes:
        meta = node.node.metadata
        page = meta.get("page_number")
        art_nr = meta.get("art_nr")
        part_id = meta.get("part_id", "")
        node_type = meta.get("node_type", "")

        label = ""
        if art_nr and art_nr not in seen:
            label = f"Art.{art_nr}"
            seen.add(art_nr)
        elif part_id and part_id not in seen:
            label = part_id.replace("_", " ")
            seen.add(part_id)

        if label:
            page_str = f" · p.{page}" if page else ""
            pills.append(
                f'<span class="source-pill">{label}{page_str} [{node_type}]</span>'
            )
    return "".join(pills)

def build_sources_data_old(nodes) -> list[dict]:
    sources = []
    seen = set()

    for node in nodes:
        meta = node.node.metadata
        page = meta.get("page_number")
        art_nr = meta.get("art_nr")
        part_id = meta.get("part_id", "")
        node_type = meta.get("node_type", "")

        label = ""
        if art_nr:
            label = f"Art.{art_nr}"
        elif part_id:
            label = part_id.replace("_", " ")
        else:
            label = "Unknown source"

        text_content = ""
        try:
            text_content = node.node.get_content()
        except Exception:
            text_content = getattr(node.node, "text", "")

        unique_key = (label, page, node_type, text_content)
        if unique_key in seen:
            continue
        seen.add(unique_key)

        sources.append(
            {
                "label": label,
                "page_number": page,
                "node_type": node_type,
                "content": text_content,
            }
        )

    return sources

def build_sources_data(nodes) -> list[dict]:
    sources = []
    seen = set()

    for source_node in nodes:
        meta = source_node.node.metadata or {}

        node_type = meta.get("node_type", "unknown")
        page_number = meta.get("page_number")
        score = getattr(source_node, "score", None)

        content = ""
        try:
            content = source_node.node.get_content()
        except Exception:
            pass

        if not content:
            content = getattr(source_node.node, "text", "")

        if not content:
            content = getattr(source_node, "text", "")

        if not content:
            content = str(source_node.node)

        unique_key = (node_type, page_number, score, content)
        if unique_key in seen:
            continue
        seen.add(unique_key)

        sources.append(
            {
                "node_type": node_type,
                "page_number": page_number,
                "score": score,
                "content": content,
            }
        )

    return sources
# -----------------------------------------------------------------------
# Query with memory
# -----------------------------------------------------------------------
def run_query_with_memory(user_query: str) -> tuple[str, str, list[dict]]:
    mode = st.session_state.retrieval_mode
    top_k = st.session_state.retrieval_top_k

    if mode == "Agent Mode":
        agent = st.session_state.agent_retriever

        if agent is None:
            return "⚠️ Agent Mode is not ready yet. Run Step 6 first.", "", []

        async def _run_agent_query():
            response = await agent.run(user_query)
            return response

        try:
            response = asyncio.run(_run_agent_query())
        except Exception as e:
            logging.error(f"Agent Mode failed: {e}")
            return f"⚠️ Agent retrieval error: {e}", "", []

        return str(response), "", []
    
    if mode == "Vector Only":
        # retriever = build_basic_vector_retriever(basic_index=st.session_state.basic_index)
        retriever = build_basic_vector_retriever(
            basic_index=st.session_state.basic_index,
            similarity_top_k=top_k,
        )
    elif mode == "Hybrid Only":
        # retriever = build_basic_hybrid_retriever(hybrid_index=st.session_state.hybrid_index)
        retriever = build_basic_hybrid_retriever(
            hybrid_index=st.session_state.hybrid_index,
            similarity_top_k=top_k,
        )
    elif mode == "Hybrid + Metadata Filtering":
        intent = extract_query_intent(user_query, Settings.llm)
        retriever = build_filtered_retriever(
            intent=intent,
            basic_index=st.session_state.basic_index,
            hybrid_index=st.session_state.hybrid_index,
            use_hybrid=True,
        )
    else:
        retriever = build_custom_retriever(
            query=user_query,
            basic_index=st.session_state.basic_index,
            hybrid_index=st.session_state.hybrid_index,
            kg_retriever=st.session_state.kg_retriever,
        )

    node_postprocessors = []
    if st.session_state.use_reranker:
        try:
            reranker = SentenceTransformerRerank(
                model=st.session_state.reranker_model,
                top_n=st.session_state.rerank_top_k,
                # top_n=5,
            )
            node_postprocessors.append(reranker)
        except Exception as e:
            logging.warning(f"Reranker unavailable: {e}")

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=st.session_state.memory,
        llm=Settings.llm,
        node_postprocessors=node_postprocessors,
        context_prompt=(
            "You are an expert in mechanical gear catalogs.\n"
            "Answer using only the catalog data provided below.\n"
            "If the answer is not in the context, say so explicitly.\n\n"
            "Context:\n{context_str}\n\nChat history:\n{chat_history}"
        ),
    )

    response = chat_engine.chat(user_query)
    source_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
    # return str(response), format_sources(source_nodes)

    sources_html = format_sources(source_nodes)
    sources_data = build_sources_data(source_nodes)
    
    return str(response), sources_html, sources_data

# -----------------------------------------------------------------------
# Developer Console Helper
# -----------------------------------------------------------------------
def run_dev_console(code: str) -> str:
    output_buffer = io.StringIO()

    local_ctx = {
        "st": st,
        "Settings": Settings,
        # handy aliases
        "extraction_result_layer1": st.session_state.extraction_result_layer1,
        "extraction_result_layer2": st.session_state.extraction_result_layer2,
        "mapped_parts": st.session_state.mapped_parts,
        "part_nodes": st.session_state.part_nodes,
        "row_nodes": st.session_state.row_nodes,
        "all_nodes": st.session_state.all_nodes,
        "basic_index": st.session_state.basic_index,
        "hybrid_index": st.session_state.hybrid_index,
        "property_graph_index": st.session_state.property_graph_index,
        "vector_retriever": st.session_state.vector_retriever,
        "hybrid_retriever": st.session_state.hybrid_retriever,
        "kg_retriever": st.session_state.kg_retriever,
    }

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {}, local_ctx)
    except Exception:
        output_buffer.write(traceback.format_exc())

    return output_buffer.getvalue()

# -----------------------------------------------------------------------
# Top navbar
# -----------------------------------------------------------------------
nav_left, nav_right = st.columns([4, 5])

with nav_left:
    st.markdown(
        """
        <div class="top-nav-shell">
            <div class="brand-wrap">
                <div class="brand-icon">⚙</div>
                <div>
                    <div class="brand-title">Mechanical Part AI-Assistant</div>
                    <div class="brand-subtitle">RAG-powered catalog assistant for engineering workflows</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Chat", use_container_width=True):
            switch_tab("Chat")
    with c2:
        if st.button("Settings", use_container_width=True):
            switch_tab("Settings")
    with c3:
        if st.button("Pipeline", use_container_width=True):
            switch_tab("Pipeline")
    with c4:
        if st.button("Evaluation", use_container_width=True):
            switch_tab("Evaluation")

st.markdown(
    f'<div class="active-pill">Current tab: {st.session_state.active_tab}</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# CHAT TAB
# -----------------------------------------------------------------------
if st.session_state.active_tab == "Chat":
    st.markdown('<div class="section-title">Chat with your assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Ask about spur gears, bevel gears, dimensions, materials, torque, or article numbers.</div>',
        unsafe_allow_html=True,
    )

    chat_tab_assistant, chat_tab_benchmark = st.tabs(
        ["Assistant Chat", "Retrieval Benchmarking Playground"]
    )

    with chat_tab_assistant:
        left_col, right_col = st.columns([3.8, 1.3], gap="large")

        with left_col:

            if pipeline_ready():
                mode_text = st.session_state.retrieval_mode
                if st.session_state.use_reranker:
                    mode_text += " · Reranker On"
                st.markdown(
                    f'<div class="mode-badge">{mode_text}</div>',
                    unsafe_allow_html=True,
                )
        
            with st.container(border=True, height=500):
                if not pipeline_ready() and not st.session_state.chat_history:
                    st.markdown(
                        """
                        <div class="placeholder-card">
                            <div style="font-size:2.2rem; margin-bottom:0.8rem;">⚙️</div>
                            Run the pipeline steps in the <b>Pipeline</b> tab to activate chat.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    for msg in st.session_state.chat_history:
                        role = msg["role"]
                        content = msg["content"]
                        sources_html = msg.get("sources_html", "")
                        sources_data = msg.get("sources_data", [])

                        if role == "user":
                            st.markdown(
                                f"""
                                <div class="msg-row user">
                                    <div>
                                        <div class="msg-meta" style="text-align:right;">You</div>
                                        <div class="msg-bubble user">{content}</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            assistant_html = (
                                f'<div class="msg-row assistant">'
                                f'<div style="width:100%;">'
                                f'<div class="msg-meta">AI Assistant</div>'
                                f'<div class="msg-bubble assistant">{content}</div>'
                                f'</div>'
                                f'</div>'
                            )

                            st.markdown(assistant_html, unsafe_allow_html=True)

                            if sources_data:
                                st.markdown(
                                    '<div class="sources-label" style="margin:0.5rem 0 0.35rem 0;">Sources</div>',
                                    unsafe_allow_html=True,
                                )

                                for source in sources_data:
                                    node_type = source.get("node_type", "unknown")
                                    page_number = source.get("page_number", "N/A")
                                    score = source.get("score", None)

                                    if score is not None:
                                        score_text = f"{float(score):.2f}"
                                    else:
                                        score_text = "N/A"

                                    source_card_html = (
                                        f'<div style="border:1px solid #e4e4e7; border-radius:14px; '
                                        f'padding:0.7rem 0.9rem; margin:0.45rem 0 0.2rem 0; background:#ffffff;">'
                                        f'<div style="display:flex; justify-content:space-between; align-items:center; '
                                        f'gap:0.75rem; flex-wrap:wrap;">'
                                        f'<div style="display:flex; align-items:center; gap:0.55rem; flex-wrap:wrap;">'
                                        f'<span style="display:inline-block; background:#f5f3ff; border:1px solid #ddd6fe; '
                                        f'color:#7c3aed; font-size:0.72rem; font-weight:600; padding:2px 8px; '
                                        f'border-radius:999px;">{node_type}</span>'
                                        f'<span style="color:#71717a; font-size:0.82rem; font-weight:600;">'
                                        f'Page {page_number}</span>'
                                        f'</div>'
                                        f'<div style="color:#16a34a; font-size:0.82rem; font-weight:700;">'
                                        f'Score: {score_text}</div>'
                                        f'</div>'
                                        f'</div>'
                                    )

                                    st.markdown(source_card_html, unsafe_allow_html=True)

                                    with st.expander("Show full context", expanded=False):
                                        st.text(source.get("content", ""))

                if st.session_state.generate_response_now and st.session_state.pending_user_prompt:
                    with st.spinner("Thinking..."):
                        try:
                            # response_text, sources_html = run_query_with_memory(
                            #     st.session_state.pending_user_prompt
                            # )
                            response_text, sources_html, sources_data = run_query_with_memory(
                                st.session_state.pending_user_prompt
                            )
                        except Exception as e:
                            response_text = f"⚠️ Retrieval error: {e}"
                            sources_html = ""
                            sources_data = []
                            logging.error(f"Query failed: {e}")

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                            # "sources_html": sources_html,
                            "sources_data": sources_data,
                        }
                    )
                    st.session_state.pending_user_prompt = None
                    st.session_state.generate_response_now = False
                    st.rerun()

            with st.container(border=True):
                with st.form("chat_form", clear_on_submit=True):
                    user_prompt = st.text_area(
                        "Message",
                        placeholder="e.g. Find steel spur gears with module 1.0 and torque > 200 Ncm",
                        height=110,
                        label_visibility="collapsed",
                        disabled=not pipeline_ready(),
                    )

                    b1, b2, b3 = st.columns([1.2, 1.2, 6])

                    with b1:
                        send_clicked = st.form_submit_button("Send", use_container_width=True)
                    with b2:
                        clear_clicked = st.form_submit_button("Clear Chat", use_container_width=True)

            if clear_clicked:
                clear_chat()
                st.session_state.pending_user_prompt = None
                st.session_state.generate_response_now = False
                st.rerun()

            if send_clicked and user_prompt.strip():
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_prompt.strip()}
                )
                st.session_state.pending_user_prompt = user_prompt.strip()
                st.session_state.generate_response_now = True
                st.rerun()

    with chat_tab_benchmark:
        st.markdown('<div class="section-title">Retrieval Benchmarking Playground</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Run one query across all retrieval modes and compare answer quality, sources, and latency.</div>',
            unsafe_allow_html=True,
        )

        if not pipeline_ready():
            st.info("Run all pipeline steps first to enable benchmarking.")
        else:
            with st.container(border=True):
                benchmark_query = st.text_area(
                    "Benchmark query",
                    value=st.session_state.benchmark_query,
                    height=100,
                    placeholder="e.g. Is there a POM spur gear with module 1.25 and ZZ=110?",
                )

                # Top_k sliders
                benchmark_retrieval_top_k, benchmark_rerank_top_k = st.columns(2)

                with benchmark_retrieval_top_k:
                    st.session_state.benchmark_retrieval_top_k = st.slider(
                        "Top-k (retriever)",
                        min_value=1,
                        max_value=20,
                        value=st.session_state.get("benchmark_retrieval_top_k", st.session_state.retrieval_top_k),
                        step=1,
                        disabled=not pipeline_ready(),
                    )

                with benchmark_rerank_top_k:
                    st.session_state.benchmark_rerank_top_k = st.slider(
                        "Top-k (reranker)",
                        min_value=1,
                        max_value=10,
                        value=st.session_state.get("benchmark_rerank_top_k", st.session_state.rerank_top_k),
                        step=1,
                        disabled=(not pipeline_ready() or not st.session_state.use_reranker),
                    )

                # Run comparison/Clear results
                button_col1, button_col2, _ = st.columns([1, 1, 4])
                with button_col1:
                    run_benchmark = st.button("Run comparison", use_container_width=True)

                with button_col2:
                    clear_benchmark = st.button("Clear results", use_container_width=True)

                if clear_benchmark:
                    st.session_state.benchmark_results = {}
                    st.rerun()

                # Query Suggestions
                st.markdown("#### Query suggestions")
                suggestion_cols = st.columns(3)

                suggestions_hybrid = [
                    "How many spur gear parts made of PK does this catalog have? Mention their module please.",
                    "Is there a POM spur gear with module 1.25 and ZZ=110?",
                    "Give me the spur gear with article number SH0555HF",
                ]

                suggestions_range_numeric_constraint = [
                    "Find spur gears with module 1.0 and torque above 180 Ncm.",
                    "Find PK gears with teeth number fewer than 14 teeth",
                    "Find me PK gear rows with torque > 200Ncm and teeth count < 23 teeth",
                ]

                suggestions_multi_hop = [
                    "Give me all articles that share the same module AND material as article SPK125110PK",
                    "Which material has the most gear variants overall?",
                    "Find gears that have the same teeth count as the heaviest gear in the catalog",
                ]

                # for i, suggestion in enumerate(suggestions):
                #     with suggestion_cols[i]:
                #         if st.button(suggestion, key=f"benchmark_suggestion_{i}", use_container_width=True):
                #             st.session_state.benchmark_query = suggestion
                #             st.rerun()

                st.markdown("**Hybrid / direct lookup**")
                hybrid_cols = st.columns(3)
                for i, suggestion in enumerate(suggestions_hybrid):
                    with hybrid_cols[i]:
                        if st.button(suggestion, key=f"benchmark_hybrid_suggestion_{i}", use_container_width=True):
                            st.session_state.benchmark_query = suggestion
                            st.rerun()

                st.markdown("**Range / numeric constraints**")
                range_cols = st.columns(3)
                for i, suggestion in enumerate(suggestions_range_numeric_constraint):
                    with range_cols[i]:
                        if st.button(suggestion, key=f"benchmark_range_suggestion_{i}", use_container_width=True):
                            st.session_state.benchmark_query = suggestion
                            st.rerun()

                st.markdown("**Multi-hop reasoning**")
                multi_cols = st.columns(3)
                for i, suggestion in enumerate(suggestions_multi_hop):
                    with multi_cols[i]:
                        if st.button(suggestion, key=f"benchmark_multihop_suggestion_{i}", use_container_width=True):
                            st.session_state.benchmark_query = suggestion
                            st.rerun()

                    
            if benchmark_query != st.session_state.benchmark_query:
                st.session_state.benchmark_query = benchmark_query

            if run_benchmark and st.session_state.benchmark_query.strip():
                with st.spinner("Running retrieval comparison across all modes..."):
                    st.session_state.benchmark_results = benchmark_all_modes(
                        st.session_state.benchmark_query.strip()
                    )
                st.rerun()

            # -----------------------------------
            # benchmark summary tab
            # -----------------------------------

            if st.session_state.benchmark_results:
                
                st.markdown("### Query")
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #e4e4e7;
                        border-radius:14px;
                        padding:0.8rem 1rem;
                        background:#fafafa;
                        color:#18181b;
                        font-weight:600;
                        margin-bottom:0.8rem;
                    ">
                        {st.session_state.benchmark_query}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                summary_rows = []
                for mode, result in st.session_state.benchmark_results.items():
                    answer_text = result.get("answer", "")
                    answer_preview = (
                        answer_text[:120] + "..."
                        if isinstance(answer_text, str) and len(answer_text) > 120
                        else answer_text
                    )

                    sources = result.get("sources", [])
                    top_score = None
                    if sources:
                        numeric_scores = []
                        for s in sources:
                            try:
                                if s.get("score") is not None:
                                    numeric_scores.append(float(s.get("score")))
                            except Exception:
                                pass
                        if numeric_scores:
                            top_score = max(numeric_scores)

                    latency = result.get("latency_sec", None)

                    summary_rows.append(
                        {
                            "Mode": mode,
                            "Answer": answer_preview if answer_preview else ("Error" if result.get("error") else ""),
                            "Sources": len(sources),
                            "Score": f"{top_score:.2f}" if top_score is not None else "N/A",
                            "Latency (s)": f"{latency:.2f}" if isinstance(latency, (int, float)) else "N/A",
                        }
                    )

                st.markdown("### Summary")
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
                st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

                # col1, col2 = st.columns(2, gap="large")
                # col3, col4 = st.columns(2, gap="large")

                # mode_columns = {
                #     "Vector Only": col1,
                #     "Hybrid Only": col2,
                #     "Hybrid + Metadata Filtering": col3,
                #     "KG + Hybrid + Metadata Filtering": col4,
                # }

                benchmark_modes = list(st.session_state.benchmark_results.keys())

                rows = [benchmark_modes[i:i+2] for i in range(0, len(benchmark_modes), 2)]

                for row_modes in rows:
                    row_cols = st.columns(len(row_modes), gap="large")
                    for mode, col in zip(row_modes, row_cols):
                        result = st.session_state.benchmark_results.get(mode, {})

                # for mode, col in mode_columns.items():
                #     result = st.session_state.benchmark_results.get(mode, {})

                        with col:
                            with st.container(border=True):
                                st.markdown(f"### {mode}")

                                latency = result.get("latency_sec")
                                latency_text = f"{latency:.2f} s" if isinstance(latency, (int, float)) else "N/A"

                                st.markdown(
                                    f"""
                                    <div style="font-size:0.85rem; color:#71717a; margin-bottom:0.35rem;">Latency</div>
                                    <div style="font-weight:700; color:#18181b; margin-bottom:0.9rem;">{latency_text}</div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                if result.get("error"):
                                    st.error(result["error"])
                                else:
                                    st.markdown("**Answer**")
                                    st.write(result.get("answer", ""))

                                    sources = result.get("sources", [])

                                    show_sources = st.toggle(
                                        "Show sources",
                                        value=False,
                                        key=f"show_sources_{mode}",
                                    )
                                    if show_sources:
                                        if sources:
                                            st.markdown("**Sources**")
                                            for source in sources:
                                                node_type = source.get("node_type", "unknown")
                                                page_number = source.get("page_number", "N/A")
                                                score = source.get("score", None)

                                                if score is not None:
                                                    score_text = f"{float(score):.2f}"
                                                else:
                                                    score_text = "N/A"

                                                source_card_html = (
                                                    f'<div style="border:1px solid #e4e4e7; border-radius:14px; '
                                                    f'padding:0.7rem 0.9rem; margin:0.45rem 0 0.2rem 0; background:#ffffff;">'
                                                    f'<div style="display:flex; justify-content:space-between; align-items:center; '
                                                    f'gap:0.75rem; flex-wrap:wrap;">'
                                                    f'<div style="display:flex; align-items:center; gap:0.55rem; flex-wrap:wrap;">'
                                                    f'<span style="display:inline-block; background:#f5f3ff; border:1px solid #ddd6fe; '
                                                    f'color:#7c3aed; font-size:0.72rem; font-weight:600; padding:2px 8px; '
                                                    f'border-radius:999px;">{node_type}</span>'
                                                    f'<span style="color:#71717a; font-size:0.82rem; font-weight:600;">'
                                                    f'Page {page_number}</span>'
                                                    f'</div>'
                                                    f'<div style="color:#16a34a; font-size:0.82rem; font-weight:700;">'
                                                    f'Score: {score_text}</div>'
                                                    f'</div>'
                                                    f'</div>'
                                                )

                                                st.markdown(source_card_html, unsafe_allow_html=True)

                                                with st.expander("Show full context", expanded=False):
                                                    st.text(source.get("content", ""))
                                        else:
                                            st.info("No sources returned.")

    with right_col:
        st.markdown('<div class="section-title">Retrieval configuration</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Adjust retrieval behavior directly from chat.</div>', unsafe_allow_html=True)

        with st.container(border=True):
            retrieval_options = [
                "Vector Only",
                "Hybrid Only",
                "Hybrid + Metadata Filtering",
                "KG + Hybrid + Metadata Filtering",
            ]

            if st.session_state.agent_retriever is not None:
                retrieval_options.append("Agent Mode")

            st.session_state.retrieval_mode = st.selectbox(
                "Retrieval Mode",
                retrieval_options,
                index=retrieval_options.index(st.session_state.retrieval_mode),
                disabled=not pipeline_ready(),
            )

            st.session_state.use_reranker = st.toggle(
                "Enable reranker model",
                value=st.session_state.use_reranker,
                disabled=not pipeline_ready(),
            )

            st.session_state.retrieval_top_k = st.slider(
                "Top-k (retrieval)",
                min_value=1,
                max_value=10,
                value=st.session_state.retrieval_top_k,
                step=1,
                disabled=not pipeline_ready(),
            )

            st.session_state.rerank_top_k = st.slider(
                "Top-k (rerank)",
                min_value=1,
                max_value=5,
                value=st.session_state.rerank_top_k,
                step=1,
                disabled=(not pipeline_ready() or not st.session_state.use_reranker),
            )

            if not pipeline_ready():
                st.caption("Available after pipeline setup.")

            st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
            # st.markdown('<div class="section-title">Quick status</div>', unsafe_allow_html=True)
            # st.markdown('<div class="section-subtitle">Current app and pipeline state.</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.5rem;">Pipeline</div>
                <div style="font-size:1.2rem; font-weight:700; color:#18181b; margin-bottom:1rem;">
                    {"Ready" if pipeline_ready() else "Not ready"}
                </div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.3rem;">LLM</div>
                <div style="font-weight:600; color:#18181b; margin-bottom:0.8rem;">{st.session_state.llm_choice}</div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.3rem;">Embeddings</div>
                <div style="font-weight:600; color:#18181b; margin-bottom:0.8rem;">{st.session_state.embed_choice}</div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.3rem;">Reranker model</div>
                <div style="font-weight:600; color:#18181b; margin-bottom:0.8rem;">
                    {st.session_state.reranker_model if st.session_state.use_reranker else "Disabled"}
                </div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.3rem;">Top-k (retrieval)</div>
                <div style="font-weight:600; color:#18181b; margin-bottom:0.8rem;">{st.session_state.retrieval_top_k}</div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.3rem;">Top-k (rerank)</div>
                <div style="font-weight:600; color:#18181b; margin-bottom:0.8rem;">
                    {st.session_state.rerank_top_k if st.session_state.use_reranker else "Disabled"}
                </div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.3rem;">Messages</div>
                <div style="font-weight:600; color:#18181b;">{len(st.session_state.chat_history)}</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size:1rem; font-weight:700; color:#18181b; margin-bottom:0.5rem;">How to use</div>
                <div style="color:#71717a; font-size:0.92rem; line-height:1.6;">
                    1. Open the <b>Settings</b> tab and configure model options.<br>
                    2. Go to <b>Pipeline</b> tab and run all six steps.<br>
                    3. Adjust retrieval options on the <b>Chat</b> page.<br>
                    4. Ask the AI-Assistant .
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------
# SETTINGS TAB
# -----------------------------------------------------------------------
elif st.session_state.active_tab == "Settings":
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Configure models and session controls.</div>', unsafe_allow_html=True,
    )

    s1, s2 = st.columns(2, gap="large")

    with s1:
        with st.container(border=True):
            st.markdown('<div class="section-title">Model configuration</div>', unsafe_allow_html=True)
            # st.markdown('<div class="section-subtitle">Choose your LLM and embedding model.</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Choose your LLM, embedding model, and reranker model.</div>', unsafe_allow_html=True)
            
            llm_choice = st.selectbox(
                "Language Model",
                ["DeepSeek (deepseek-chat)", "OpenAI (gpt-4o-mini)", "OpenAI (gpt-4o)"],
                index=["DeepSeek (deepseek-chat)", "OpenAI (gpt-4o-mini)", "OpenAI (gpt-4o)"].index(
                    st.session_state.llm_choice
                ),
            )
            if llm_choice != st.session_state.llm_choice:
                st.session_state.llm_choice = llm_choice
                setup_models()

            embed_choice = st.selectbox(
                "Embedding Model",
                ["OpenAI (text-embedding-3-small)", "OpenAI (text-embedding-3-large)"],
                index=["OpenAI (text-embedding-3-small)", "OpenAI (text-embedding-3-large)"].index(
                    st.session_state.embed_choice
                ),
            )
            if embed_choice != st.session_state.embed_choice:
                st.session_state.embed_choice = embed_choice
                setup_models()

            reranker_model = st.selectbox(
                "Reranker Model",
                ["cross-encoder/ms-marco-MiniLM-L-2-v2"],
                index=["cross-encoder/ms-marco-MiniLM-L-2-v2"].index(st.session_state.reranker_model),
            )

            if reranker_model != st.session_state.reranker_model:
                st.session_state.reranker_model = reranker_model

            # st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# PIPELINE TAB
# -----------------------------------------------------------------------
elif st.session_state.active_tab == "Pipeline":
    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Run the extraction, indexing, graph, and retriever build steps for your RAG pipeline.</div>',
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns([3.4, 1.2], gap="large")

    with p1:
        for step in STEPS:
            key = step["key"]
            status = st.session_state[key]
            stat = st.session_state.get(f"{key}_stat", "")
            log = st.session_state.get(f"{key}_log", "")
            req = step["requires"]

            blocked = bool(req and st.session_state[req] != "done")

            if key == "step6_retrievers":
                blocked = bool(
                    st.session_state["step4_indexing"] != "done"
                    # or st.session_state["step5_kg"] != "done"
                )

            css = step_css_class(status)
            icon = step_icon(status)

            st.markdown(
                f"""
                <div class="step-card {css}">
                    <div class="step-label">{icon} {step['label']}</div>
                    <div class="step-name">{step['name']}</div>
                    {"<div class='step-stat'>" + stat + "</div>" if stat else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns([1.2, 1, 4])

            with c1:
                btn_label = "Run" if status == "idle" else ("Re-run" if status == "done" else "Retry")
                if st.button(btn_label, key=f"btn_{key}", use_container_width=True, disabled=blocked):
                    with st.spinner(f"Running {step['name']}..."):
                        step["run"]()
                    st.rerun()

            with c2:
                if log:
                    st.toggle("Logs", key=f"log_toggle_{key}", value=False)

            if log and st.session_state.get(f"log_toggle_{key}", False):
                st.markdown(f'<div class="log-box">{log}</div>', unsafe_allow_html=True)

            if key == "step1_extraction" and st.session_state.step1_extraction == "done":
                with st.expander("Show extraction results", expanded=False):
                    layer1_df = extraction_result_to_df(st.session_state.extraction_result_layer1)
                    layer2_df = extraction_result_to_df(st.session_state.extraction_result_layer2)

                    tab1, tab2 = st.tabs(["Layer 1", "Layer 2"])

                    with tab1:
                        if layer1_df is not None and not layer1_df.empty:
                            st.caption(f"{len(layer1_df)} extracted parent record(s)")
                            st.dataframe(layer1_df, use_container_width=True)
                        else:
                            st.info("No Layer 1 extraction results available.")

                    with tab2:
                        if layer2_df is not None and not layer2_df.empty:
                            st.caption(f"{len(layer2_df)} extracted row record(s)")
                            st.dataframe(layer2_df, use_container_width=True)
                        else:
                            st.info("No Layer 2 extraction results available.")

            if key == "step2_mapping" and st.session_state.step2_mapping == "done":
                with st.expander("Show mapped parts and attached rows", expanded=False):
                    preview_items = mapped_parts_to_preview(st.session_state.mapped_parts)

                    if not preview_items:
                        st.info("No mapped parts available.")
                    else:
                        st.caption(f"{len(preview_items)} mapped part(s)")

                        for item in preview_items:
                            part_id = item["part_id"]
                            parent_summary = item["parent_summary"]
                            rows = item["rows"]

                            with st.container(border=True):
                                st.markdown(f"### {part_id}")

                                st.markdown("**Parent info**")
                                # st.json(parent_summary)
                                import pandas as pd
                                st.dataframe(pd.DataFrame([parent_summary]), use_container_width=True)

                                st.markdown("**Attached rows**")
                                if rows:
                                    import pandas as pd
                                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                                else:
                                    st.info("No rows attached to this parent.")   

            if key == "step3_nodes" and st.session_state.step3_nodes == "done":
                with st.expander("Show node graph", expanded=False):
                    graph_nodes, graph_edges, graph_config, graph_detail_map = build_step3_graph(
                        st.session_state.part_nodes,
                        st.session_state.row_nodes,
                        max_children_per_parent=50,
                    )

                    selected_node_id = agraph(
                        nodes=graph_nodes,
                        edges=graph_edges,
                        config=graph_config,
                    )

                    if selected_node_id:
                        details = graph_detail_map.get(selected_node_id, {})
                        st.markdown("### Selected node details")
                        # st.json(details)
                        details_md = f"""
                        **node_id:** {details.get("node_id", "")}  
                        **part_id:** {details.get("part_id", "")}  
                        **node_type:** {details.get("node_type", "")}  
                        **page_number:** {details.get("page_number", "")}  
                        """

                        if details.get("node_type") != "part" and "art_nr" in details:
                            details_md += f"**art_nr:** {details.get('art_nr', '')}  \n"

                        details_md += f"""
                        **text_preview:**  
                        {details.get("text_preview", "")}
                        """

                        st.markdown(details_md)
                    else:
                        st.info("Click a node in the graph to view its details.")

            if key == "step6_retrievers" and st.session_state.step6_retrievers == "done":
                with st.expander("Show retriever status", expanded=False):
                    vector_ready = st.session_state.vector_retriever is not None
                    hybrid_ready = st.session_state.hybrid_retriever is not None
                    kg_ready = st.session_state.kg_retriever is not None

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown(
                            f"""
                            <div style="
                                border:1px solid #e4e4e7;
                                border-radius:16px;
                                padding:1rem;
                                background:#ffffff;
                            ">
                                <div style="font-size:0.82rem; color:#71717a; margin-bottom:0.35rem;">Vector retriever</div>
                                <div style="font-size:1rem; font-weight:700; color:{'#16a34a' if vector_ready else '#dc2626'};">
                                    {'Ready' if vector_ready else 'Not ready'}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with c2:
                        st.markdown(
                            f"""
                            <div style="
                                border:1px solid #e4e4e7;
                                border-radius:16px;
                                padding:1rem;
                                background:#ffffff;
                            ">
                                <div style="font-size:0.82rem; color:#71717a; margin-bottom:0.35rem;">Hybrid retriever</div>
                                <div style="font-size:1rem; font-weight:700; color:{'#16a34a' if hybrid_ready else '#dc2626'};">
                                    {'Ready' if hybrid_ready else 'Not ready'}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with c3:
                        st.markdown(
                            f"""
                            <div style="
                                border:1px solid #e4e4e7;
                                border-radius:16px;
                                padding:1rem;
                                background:#ffffff;
                            ">
                                <div style="font-size:0.82rem; color:#71717a; margin-bottom:0.35rem;">KG retriever</div>
                                <div style="font-size:1rem; font-weight:700; color:{'#16a34a' if kg_ready else '#dc2626'};">
                                    {'Ready' if kg_ready else 'Not ready'}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    with p2:
        with st.container(border=True):
            uploaded_pdf = st.file_uploader(
                "Upload PDF source",
                type=["pdf"],
                accept_multiple_files=False,
            )

            if uploaded_pdf is not None:
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    st.session_state.uploaded_pdf_path = tmp_file.name
                    st.session_state.uploaded_pdf_name = uploaded_pdf.name

            st.markdown(
                f"""
                <div style="font-size:1rem; font-weight:700; color:#18181b; margin-bottom:0.7rem;">Pipeline status</div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.25rem;">PDF source</div>
                <div style="font-weight:600; color:#18181b; margin-bottom:0.8rem;">
                    {st.session_state.uploaded_pdf_name}
                </div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.25rem;">Ready for chat</div>
                <div style="font-weight:600; color:#18181b; margin-bottom:0.8rem;">{"Yes" if pipeline_ready() else "No"}</div>

                <div style="font-size:0.9rem; color:#71717a; margin-bottom:0.25rem;">Completed steps</div>
                <div style="font-weight:600; color:#18181b;">{sum(st.session_state[k] == "done" for k in STEP_KEYS)} / {len(STEP_KEYS)}</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

            if st.button("Reset All Pipeline State", use_container_width=True):
                reset_all()
                st.rerun()

            # st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Admin debug panel
    # -----------------------------------------------------------------------
    show_debug_panel = st.toggle("Show admin/debug panel", value=False)

    if show_debug_panel:
        st.markdown("## Admin / Debug")

        with st.expander("Step 1 debug", expanded=False):
            if st.session_state.extraction_result_layer1 is not None:
                st.code(
                    f"Layer 1 parts extracted: {len(st.session_state.extraction_result_layer1.data)}\n"
                    f"Layer 2 rows extracted:  {len(st.session_state.extraction_result_layer2.data)}"
                )
                st.write("Layer 1 object:", st.session_state.extraction_result_layer1)
                st.write("Layer 2 object:", st.session_state.extraction_result_layer2)
            else:
                st.info("Step 1 has not been run yet.")

        with st.expander("Step 2 debug", expanded=False):
            if st.session_state.mapped_parts is not None:
                st.code(
                    f"Mapped parts: {len(st.session_state.mapped_parts.parts)}\n"
                    f"Orphaned rows: {st.session_state.mapped_parts.orphaned_count}"
                )
                st.write("Mapped parts object:", st.session_state.mapped_parts)
            else:
                st.info("Step 2 has not been run yet.")

        with st.expander("Step 3 debug", expanded=False):
            if st.session_state.part_nodes is not None:
                st.code(
                    f"Parent nodes: {len(st.session_state.part_nodes)}\n"
                    f"Child nodes: {len(st.session_state.row_nodes)}\n"
                    f"All nodes: {len(st.session_state.all_nodes)}"
                )
                st.write("Part nodes:", st.session_state.part_nodes[:3])
                st.write("Row nodes:", st.session_state.row_nodes[:3])
            else:
                st.info("Step 3 has not been run yet.")

        with st.expander("Step 4 debug", expanded=False):
            st.write("Basic index:", st.session_state.basic_index)
            st.write("Hybrid index:", st.session_state.hybrid_index)

        with st.expander("Step 5 debug", expanded=False):
            st.write("Property graph index:", st.session_state.property_graph_index)

        with st.expander("Step 6 debug", expanded=False):
            st.write("Vector retriever:", st.session_state.vector_retriever)
            st.write("Hybrid retriever:", st.session_state.hybrid_retriever)
            st.write("KG retriever:", st.session_state.kg_retriever)

    # -----------------------------------------------------------------------
    # Developer Console
    # -----------------------------------------------------------------------

    dev_console = st.toggle("Show dev-console", value=False)

    if dev_console:
        st.markdown("## Developer Console")

        with st.container(border=True):
            # st.markdown("### Developer Console")
            st.caption("Run arbitrary Python against the current pipeline/session state.")

            code = st.text_area(
                "Python code",
                value=st.session_state.dev_console_code,
                height=220,
                key="dev_console_code",
            )

            c1, c2 = st.columns([1, 5])

            with c1:
                run_code_clicked = st.button("Run code", use_container_width=True)

            with c2:
                clear_output_clicked = st.button("Clear output", use_container_width=True)

            if run_code_clicked:
                st.session_state.dev_console_output = run_dev_console(code)

            if clear_output_clicked:
                st.session_state.dev_console_output = ""

            st.markdown("### Output")
            if st.session_state.dev_console_output:
                st.code(st.session_state.dev_console_output)
            else:
                st.info("No output yet.")

# -----------------------------------------------------------------------
# EVALUATION TAB
# -----------------------------------------------------------------------
elif st.session_state.active_tab == "Evaluation":
    st.markdown('<div class="section-title">Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">This tab is reserved for benchmark questions, retrieval metrics, and answer quality analysis.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="placeholder-card">
            Evaluation UI placeholder.<br><br>
            Later, we can add:
            <br>• test question sets
            <br>• retrieved context inspection
            <br>• answer comparison
            <br>• latency / accuracy metrics
        </div>
        """,
        unsafe_allow_html=True,
    )