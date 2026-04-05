import asyncio
import logging
import sys
import os
import io
from typing import Optional

import streamlit as st
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.postprocessor.cohere_rerank import CohereRerank
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
from pipeline.indexing import build_pgvector_store, index_nodes_with_store
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
from config.settings import (
    DEEPSEEK_KEY, OPENAI_KEY,
    SYSTEM_PROMPT_L1, SYSTEM_PROMPT_L2,
    PDF_PATH,
)

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="GearBot — Mechanical Parts Catalog",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #0f0f0f; color: #e8e8e8; }

section[data-testid="stSidebar"] {
    background-color: #111;
    border-right: 1px solid #222;
}

.gear-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #f0c040;
    letter-spacing: 0.05em;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #222;
    margin-bottom: 1rem;
}

.sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #555;
    margin: 1.2rem 0 0.4rem 0;
}

/* Step card */
.step-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
}
.step-card.done   { border-left: 3px solid #4caf50; }
.step-card.running{ border-left: 3px solid #f0c040; }
.step-card.error  { border-left: 3px solid #f44336; }
.step-card.idle   { border-left: 3px solid #333; }

.step-label { color: #888; font-size: 0.62rem; letter-spacing: 0.1em; }
.step-name  { color: #e8e8e8; font-size: 0.78rem; margin-top: 1px; }
.step-stat  { color: #555; font-size: 0.65rem; margin-top: 2px; }

/* Log box */
.log-box {
    background: #0a0a0a;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #6abf6a;
    max-height: 160px;
    overflow-y: auto;
    white-space: pre-wrap;
    margin-top: 0.3rem;
}

/* Chat messages */
.stChatMessage {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    margin-bottom: 0.5rem !important;
}
.stChatMessage[data-testid="chat-message-user"]      { border-left: 3px solid #f0c040 !important; }
.stChatMessage[data-testid="chat-message-assistant"] { border-left: 3px solid #4a9eff !important; }

.source-pill {
    display: inline-block;
    background: #1e2a1e;
    border: 1px solid #2d4a2d;
    color: #6abf6a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin: 2px 3px;
}

.mode-badge {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    color: #4a9eff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 10px;
    border-radius: 3px;
    margin-bottom: 0.75rem;
}

.chat-header    { font-family: 'IBM Plex Mono', monospace; font-size: 1.7rem; font-weight: 600; color: #e8e8e8; }
.chat-subheader { font-size: 0.85rem; color: #555; margin-bottom: 1.5rem; }

.stSelectbox > div > div {
    background-color: #1a1a1a !important;
    border-color: #333 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    background: #1a1a1a;
    border: 1px solid #333;
    color: #e8e8e8;
    border-radius: 4px;
}
.stButton > button:hover { border-color: #f0c040; color: #f0c040; }
</style>
""", unsafe_allow_html=True)


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
        # pipeline artifacts
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
        # step statuses: "idle" | "done" | "error"
        **{k: "idle" for k in STEP_KEYS},
        # step logs
        **{f"{k}_log": "" for k in STEP_KEYS},
        # step stats (short summary line)
        **{f"{k}_stat": "" for k in STEP_KEYS},
        # chat
        "chat_history": [],
        "memory": ChatMemoryBuffer.from_defaults(token_limit=4096),
        # settings
        "retrieval_mode": "Hybrid + Metadata Filtering",
        "use_reranker": False,
        "llm_choice": "DeepSeek (deepseek-chat)",
        "embed_choice": "OpenAI (text-embedding-3-small)",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# -----------------------------------------------------------------------
# Model setup
# -----------------------------------------------------------------------
def setup_models():
    llm_choice   = st.session_state.llm_choice
    embed_choice = st.session_state.embed_choice

    if llm_choice == "DeepSeek (deepseek-chat)":
        Settings.llm = DeepSeek(model="deepseek-chat", api_key=DEEPSEEK_KEY, temperature=0.1)
    elif llm_choice == "OpenAI (gpt-4o-mini)":
        Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY, temperature=0.1)
    elif llm_choice == "OpenAI (gpt-4o)":
        Settings.llm = OpenAI(model="gpt-4o", api_key=OPENAI_KEY, temperature=0.1)

    if embed_choice == "OpenAI (text-embedding-3-small)":
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_KEY)
    elif embed_choice == "OpenAI (text-embedding-3-large)":
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=OPENAI_KEY)


# -----------------------------------------------------------------------
# Pipeline steps
# -----------------------------------------------------------------------
def step_icon(status: str) -> str:
    return {"idle": "○", "running": "◌", "done": "●", "error": "✕"}.get(status, "○")

def step_css_class(status: str) -> str:
    return {"idle": "idle", "done": "done", "error": "error"}.get(status, "idle")


def run_step1():
    setup_models()
    handler = capture_logs()
    st.session_state.step1_extraction = "running"
    try:
        r1 = asyncio.run(extract_layer1_fields(PDF_PATH, SYSTEM_PROMPT_L1, PartSchema))
        r2 = asyncio.run(extract_layer2_fields(PDF_PATH, SYSTEM_PROMPT_L2, TableRowSchema))
        st.session_state.extraction_result_layer1 = r1
        st.session_state.extraction_result_layer2 = r2
        st.session_state.step1_extraction = "done"
        st.session_state.step1_extraction_stat = (
            f"Layer 1: {len(r1.data)} record(s)  ·  Layer 2: {len(r2.data)} row(s)"
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
        rows  = attach_part_id_to_layer2_rows(st.session_state.extraction_result_layer2)
        mapped = map_parts_and_rows(parts, rows)
        log_mapping_diagnostics(mapped)
        st.session_state.mapped_parts = mapped
        st.session_state.step2_mapping = "done"
        st.session_state.step2_mapping_stat = (
            f"{len(mapped.parts)} part(s) joined  ·  {mapped.orphaned_count} orphaned row(s)"
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
        st.session_state.row_nodes  = row_nodes
        st.session_state.all_nodes  = part_nodes + row_nodes
        st.session_state.step3_nodes = "done"
        st.session_state.step3_nodes_stat = (
            f"{len(part_nodes)} parent node(s)  ·  {len(row_nodes)} child node(s)"
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
        basic_store  = build_pgvector_store("basic")
        hybrid_store = build_pgvector_store("hybrid")
        basic_index  = index_nodes_with_store(st.session_state.all_nodes, basic_store)
        hybrid_index = index_nodes_with_store(st.session_state.all_nodes, hybrid_store)
        st.session_state.basic_index  = basic_index
        st.session_state.hybrid_index = hybrid_index
        st.session_state.step4_indexing = "done"
        st.session_state.step4_indexing_stat = (
            f"{len(st.session_state.all_nodes)} node(s) indexed into pgvector (basic + hybrid)"
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
        st.session_state.kg_retriever = build_kg_retriever(
            property_graph_index=st.session_state.property_graph_index,
            similarity_top_k=8,
            path_depth=3,
            include_text=True,
        )
        st.session_state.step6_retrievers = "done"
        st.session_state.step6_retrievers_stat = "Vector · Hybrid · KG retrievers ready"
    except Exception as e:
        st.session_state.step6_retrievers = "error"
        st.session_state.step6_retrievers_stat = str(e)
    finally:
        release_logs(handler)
        st.session_state.step6_retrievers_log = handler.get_logs()


# -----------------------------------------------------------------------
# Step definitions
# -----------------------------------------------------------------------
STEPS = [
    {
        "key":      "step1_extraction",
        "label":    "STEP 1",
        "name":     "Two-Layer Extraction",
        "run":      run_step1,
        "requires": None,
    },
    {
        "key":      "step2_mapping",
        "label":    "STEP 2",
        "name":     "Layer Mapping & Join",
        "run":      run_step2,
        "requires": "step1_extraction",
    },
    {
        "key":      "step3_nodes",
        "label":    "STEP 3",
        "name":     "Node Construction",
        "run":      run_step3,
        "requires": "step2_mapping",
    },
    {
        "key":      "step4_indexing",
        "label":    "STEP 4",
        "name":     "pgvector Indexing",
        "run":      run_step4,
        "requires": "step3_nodes",
    },
    {
        "key":      "step5_kg",
        "label":    "STEP 5",
        "name":     "Knowledge Graph",
        "run":      run_step5,
        "requires": "step3_nodes",
    },
    {
        "key":      "step6_retrievers",
        "label":    "STEP 6",
        "name":     "Build Retrievers",
        "run":      run_step6,
        "requires": "step4_indexing",   # also needs step5, checked in button logic
    },
]

pipeline_ready = st.session_state.step6_retrievers == "done"


# -----------------------------------------------------------------------
# Source formatting
# -----------------------------------------------------------------------
def format_sources(nodes) -> str:
    seen, pills = set(), []
    for node in nodes:
        meta     = node.node.metadata
        page     = meta.get("page_number")
        art_nr   = meta.get("art_nr")
        part_id  = meta.get("part_id", "")
        node_type = meta.get("node_type", "")

        label = ""
        if art_nr and art_nr not in seen:
            label = f"Art.{art_nr}"; seen.add(art_nr)
        elif part_id and part_id not in seen:
            label = part_id.replace("_", " "); seen.add(part_id)

        if label:
            page_str = f" · p.{page}" if page else ""
            pills.append(
                f'<span class="source-pill">{label}{page_str} [{node_type}]</span>'
            )
    return "".join(pills)


# -----------------------------------------------------------------------
# Query with memory
# -----------------------------------------------------------------------
def run_query_with_memory(user_query: str) -> tuple[str, str]:
    mode = st.session_state.retrieval_mode

    if mode == "Vector Only":
        retriever = build_basic_vector_retriever(basic_index=st.session_state.basic_index)
    elif mode == "Hybrid Only":
        retriever = build_basic_hybrid_retriever(hybrid_index=st.session_state.hybrid_index)
    elif mode == "Hybrid + Metadata Filtering":
        intent = extract_query_intent(user_query, Settings.llm)
        retriever = build_filtered_retriever(
            intent=intent,
            basic_index=st.session_state.basic_index,
            hybrid_index=st.session_state.hybrid_index,
            use_hybrid=True,
        )
    else:  # KG + Hybrid + Metadata Filtering
        retriever = build_custom_retriever(
            query=user_query,
            basic_index=st.session_state.basic_index,
            hybrid_index=st.session_state.hybrid_index,
            kg_retriever=st.session_state.kg_retriever,
        )

    node_postprocessors = []
    if st.session_state.use_reranker:
        try:
            # reranker = CohereRerank(top_n=5)
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                top_n=5,
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

    response     = chat_engine.chat(user_query)
    source_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
    return str(response), format_sources(source_nodes)


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="gear-title">⚙️ GearBot</div>', unsafe_allow_html=True)

    # --- Model selection ---
    st.markdown('<div class="sidebar-section">Language Model</div>', unsafe_allow_html=True)
    llm_choice = st.selectbox(
        "LLM",
        ["DeepSeek (deepseek-chat)", "OpenAI (gpt-4o-mini)", "OpenAI (gpt-4o)"],
        index=["DeepSeek (deepseek-chat)", "OpenAI (gpt-4o-mini)", "OpenAI (gpt-4o)"].index(
            st.session_state.llm_choice
        ),
        label_visibility="collapsed",
    )
    if llm_choice != st.session_state.llm_choice:
        st.session_state.llm_choice = llm_choice
        setup_models()

    st.markdown('<div class="sidebar-section">Embedding Model</div>', unsafe_allow_html=True)
    embed_choice = st.selectbox(
        "Embeddings",
        ["OpenAI (text-embedding-3-small)", "OpenAI (text-embedding-3-large)"],
        index=["OpenAI (text-embedding-3-small)", "OpenAI (text-embedding-3-large)"].index(
            st.session_state.embed_choice
        ),
        label_visibility="collapsed",
    )
    if embed_choice != st.session_state.embed_choice:
        st.session_state.embed_choice = embed_choice
        setup_models()

    # --- Pipeline steps ---
    st.markdown('<div class="sidebar-section">Pipeline</div>', unsafe_allow_html=True)

    for step in STEPS:
        key     = step["key"]
        status  = st.session_state[key]
        stat    = st.session_state.get(f"{key}_stat", "")
        log     = st.session_state.get(f"{key}_log", "")
        req     = step["requires"]
        blocked = bool(req and st.session_state[req] != "done")

        # Extra check for step 6: also needs step5
        if key == "step6_retrievers":
            blocked = bool(
            st.session_state["step4_indexing"] != "done" or
            st.session_state["step5_kg"] != "done"
        )

        css = step_css_class(status)
        icon = step_icon(status)

        st.markdown(f"""
        <div class="step-card {css}">
            <div class="step-label">{icon} {step['label']}</div>
            <div class="step-name">{step['name']}</div>
            {"<div class='step-stat'>" + stat + "</div>" if stat else ""}
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            btn_label = "▶ Run" if status == "idle" else ("↺ Re-run" if status == "done" else "▶ Retry")
            if st.button(btn_label, key=f"btn_{key}", disabled=blocked):
                with st.spinner(f"Running {step['name']}..."):
                    step["run"]()
                st.rerun()

        with col2:
            if log:
                show_log = st.toggle("Logs", key=f"log_toggle_{key}", value=False)

        if log and st.session_state.get(f"log_toggle_{key}", False):
            st.markdown(f'<div class="log-box">{log}</div>', unsafe_allow_html=True)

    # --- Reset ---
    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    if st.button("↺ Reset All", use_container_width=True):
        for k in STEP_KEYS:
            st.session_state[k] = "idle"
            st.session_state[f"{k}_log"] = ""
            st.session_state[f"{k}_stat"] = ""
        st.session_state.chat_history = []
        st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        st.rerun()

    # --- Retrieval mode (only when pipeline ready) ---
    if pipeline_ready:
        st.markdown('<div class="sidebar-section">Retrieval Mode</div>', unsafe_allow_html=True)
        st.session_state.retrieval_mode = st.selectbox(
            "Mode",
            [
                "Vector Only",
                "Hybrid Only",
                "Hybrid + Metadata Filtering",
                "KG + Hybrid + Metadata Filtering",
            ],
            index=[
                "Vector Only",
                "Hybrid Only",
                "Hybrid + Metadata Filtering",
                "KG + Hybrid + Metadata Filtering",
            ].index(st.session_state.retrieval_mode),
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-section">Reranker</div>', unsafe_allow_html=True)
        st.session_state.use_reranker = st.toggle(
            "Enable Cohere Reranker",
            value=st.session_state.use_reranker,
        )

        st.markdown('<div class="sidebar-section">Chat</div>', unsafe_allow_html=True)
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
            st.rerun()


# -----------------------------------------------------------------------
# Main chat area
# -----------------------------------------------------------------------
st.markdown('<div class="chat-header">Mechanical Parts Catalog</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="chat-subheader">Ask anything about spur gears and bevel gears — '
    'dimensions, materials, torque, article numbers.</div>',
    unsafe_allow_html=True,
)

if pipeline_ready:
    mode_colors = {
        "Vector Only":                      "#4a9eff",
        "Hybrid Only":                      "#9b59b6",
        "Hybrid + Metadata Filtering":      "#f0c040",
        "KG + Hybrid + Metadata Filtering": "#4caf50",
    }
    c = mode_colors.get(st.session_state.retrieval_mode, "#4a9eff")
    st.markdown(
        f'<div class="mode-badge" style="border-color:{c};color:{c};">'
        f'MODE: {st.session_state.retrieval_mode.upper()}'
        f'{"  ·  RERANKER ON" if st.session_state.use_reranker else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input(
    "e.g. 'Find steel spur gears with module 1.0 and torque > 200 Ncm'",
    disabled=not pipeline_ready,
):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching catalog..."):
            try:
                response_text, sources_html = run_query_with_memory(prompt)
            except Exception as e:
                response_text = f"⚠️ Retrieval error: {e}"
                sources_html  = ""
                logging.error(f"Query failed: {e}")

        st.markdown(response_text)
        if sources_html:
            st.markdown(
                f'<div style="margin-top:0.6rem;">'
                f'<span style="font-size:0.68rem;color:#444;font-family:\'IBM Plex Mono\',monospace;">SOURCES </span>'
                f'{sources_html}</div>',
                unsafe_allow_html=True,
            )

    full = response_text + (f"\n\n{sources_html}" if sources_html else "")
    st.session_state.chat_history.append({"role": "assistant", "content": full})

# Empty state
if not pipeline_ready and not st.session_state.chat_history:
    st.markdown("""
    <div style="
        text-align:center; padding:4rem 2rem; color:#333;
        font-family:'IBM Plex Mono',monospace; font-size:0.82rem;
        border:1px dashed #222; border-radius:8px; margin-top:2rem;
    ">
        <div style="font-size:2.5rem;margin-bottom:1rem;">⚙️</div>
        Run pipeline steps from the sidebar to activate the chat.
    </div>
    """, unsafe_allow_html=True)