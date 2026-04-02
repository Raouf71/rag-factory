import asyncio
import logging
import sys
import os
from typing import Optional

import streamlit as st
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SentenceTransformerRerank

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pipeline.run_pipeline import run_pipeline
from retrieval.retriever import (
    build_basic_vector_retriever,
    build_basic_hybrid_retriever,
    build_kg_retriever,
    build_custom_retriever,
)
from retrieval.filters import build_filtered_retriever
from retrieval.intent import extract_query_intent
from config.settings import CATALOG_QA_PROMPT, DEEPSEEK_KEY, OPENAI_KEY

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #0f0f0f;
    color: #e8e8e8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161616;
    border-right: 1px solid #2a2a2a;
}

/* Title */
.gear-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f0c040;
    letter-spacing: 0.05em;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 1rem;
}

/* Section labels */
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin: 1.2rem 0 0.5rem 0;
}

/* Chat messages */
.stChatMessage {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    margin-bottom: 0.5rem !important;
}

/* User message */
.stChatMessage[data-testid="chat-message-user"] {
    border-left: 3px solid #f0c040 !important;
}

/* Assistant message */
.stChatMessage[data-testid="chat-message-assistant"] {
    border-left: 3px solid #4a9eff !important;
}

/* Source reference pill */
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

/* Mode badge */
.mode-badge {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    color: #4a9eff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin-bottom: 0.75rem;
}

/* Input box */
.stChatInputContainer {
    border-top: 1px solid #2a2a2a !important;
    background-color: #0f0f0f !important;
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    background: #1a1a1a;
    border: 1px solid #333;
    color: #e8e8e8;
    border-radius: 4px;
}

.stButton > button:hover {
    border-color: #f0c040;
    color: #f0c040;
}

/* Toggle labels */
label[data-testid="stWidgetLabel"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    color: #b0b0b0 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #1a1a1a !important;
    border-color: #333 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.green { background: #4caf50; }
.status-dot.yellow { background: #f0c040; }
.status-dot.red { background: #f44336; }

/* Chat header */
.chat-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #e8e8e8;
    margin-bottom: 0.25rem;
}
.chat-subheader {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Session state init
# -----------------------------------------------------------------------
def init_session_state():
    defaults = {
        "pipeline": None,
        "chat_history": [],
        "memory": ChatMemoryBuffer.from_defaults(token_limit=4096),
        "pipeline_ready": False,
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
def setup_models(llm_choice: str, embed_choice: str):
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


# -----------------------------------------------------------------------
# Build retriever based on selected mode
# -----------------------------------------------------------------------
def get_retriever(mode: str, query: str):
    pipeline = st.session_state.pipeline
    basic_index  = pipeline["basic_index"]
    hybrid_index = pipeline["hybrid_index"]
    kg_retriever = pipeline["kg_retriever"]

    if mode == "Vector Only":
        return build_basic_vector_retriever(basic_index=basic_index)

    elif mode == "Hybrid Only":
        return build_basic_hybrid_retriever(hybrid_index=hybrid_index)

    elif mode == "Hybrid + Metadata Filtering":
        intent = extract_query_intent(query, Settings.llm)
        return build_filtered_retriever(
            intent=intent,
            basic_index=basic_index,
            hybrid_index=hybrid_index,
            use_hybrid=True,
        )

    elif mode == "KG + Hybrid + Metadata Filtering":
        return build_custom_retriever(
            query=query,
            basic_index=basic_index,
            hybrid_index=hybrid_index,
            kg_retriever=kg_retriever,
        )

    return build_basic_hybrid_retriever(hybrid_index=hybrid_index)


# -----------------------------------------------------------------------
# Format source references
# -----------------------------------------------------------------------
def format_sources(nodes) -> str:
    seen = set()
    pills = []
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
            type_str = f" [{node_type}]" if node_type else ""
            pills.append(f'<span class="source-pill">{label}{page_str}{type_str}</span>')

    return "".join(pills) if pills else ""


# -----------------------------------------------------------------------
# Query and respond
# -----------------------------------------------------------------------
def run_query_with_memory(user_query: str) -> tuple[str, str]:
    retriever = get_retriever(st.session_state.retrieval_mode, user_query)

    # optionally wrap with reranker
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
            logger.warning(f"Reranker unavailable: {e}")

    # build chat engine with memory
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=st.session_state.memory,
        llm=Settings.llm,
        node_postprocessors=node_postprocessors,
        context_prompt=(
            "You are an expert in mechanical gear catalogs.\n"
            "Answer using only the catalog data provided below.\n"
            "If the answer is not in the context, say so explicitly.\n\n"
            "Context:\n{context_str}\n\n"
            "Chat history:\n{chat_history}"
        ),
    )

    response = chat_engine.chat(user_query)
    source_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
    sources_html = format_sources(source_nodes)

    return str(response), sources_html


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="gear-title">⚙️ GearBot</div>', unsafe_allow_html=True)

    # --- Pipeline status ---
    st.markdown('<div class="sidebar-section">Pipeline</div>', unsafe_allow_html=True)

    if not st.session_state.pipeline_ready:
        st.markdown(
            '<span class="status-dot yellow"></span> Not initialized',
            unsafe_allow_html=True,
        )
        if st.button("▶ Initialize Pipeline", use_container_width=True):
            with st.spinner("Running pipeline — this may take a few minutes..."):
                try:
                    setup_models(
                        st.session_state.llm_choice,
                        st.session_state.embed_choice,
                    )
                    st.session_state.pipeline = asyncio.run(run_pipeline())
                    st.session_state.pipeline_ready = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
    else:
        st.markdown(
            '<span class="status-dot green"></span> Pipeline ready',
            unsafe_allow_html=True,
        )
        if st.button("↺ Re-initialize", use_container_width=True):
            st.session_state.pipeline = None
            st.session_state.pipeline_ready = False
            st.session_state.chat_history = []
            st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
            st.rerun()

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
        setup_models(llm_choice, st.session_state.embed_choice)

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
        setup_models(st.session_state.llm_choice, embed_choice)

    # --- Retrieval mode ---
    st.markdown('<div class="sidebar-section">Retrieval Mode</div>', unsafe_allow_html=True)
    retrieval_mode = st.selectbox(
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
    st.session_state.retrieval_mode = retrieval_mode

    # --- Reranker ---
    st.markdown('<div class="sidebar-section">Reranker</div>', unsafe_allow_html=True)
    st.session_state.use_reranker = st.toggle(
        "Enable Cohere Reranker",
        value=st.session_state.use_reranker,
    )

    # --- Clear chat ---
    st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        st.rerun()


# -----------------------------------------------------------------------
# Main chat area
# -----------------------------------------------------------------------
st.markdown('<div class="chat-header">Mechanical Parts Catalog</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="chat-subheader">Ask anything about spur gears and bevel gears — dimensions, materials, torque, article numbers.</div>',
    unsafe_allow_html=True,
)

# Active mode badge
mode_colors = {
    "Vector Only": "#4a9eff",
    "Hybrid Only": "#9b59b6",
    "Hybrid + Metadata Filtering": "#f0c040",
    "KG + Hybrid + Metadata Filtering": "#4caf50",
}
badge_color = mode_colors.get(st.session_state.retrieval_mode, "#4a9eff")
st.markdown(
    f'<div class="mode-badge" style="border-color:{badge_color};color:{badge_color};">'
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
    disabled=not st.session_state.pipeline_ready,
):
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching catalog..."):
            if not st.session_state.pipeline_ready:
                response_text = "⚠️ Pipeline not initialized. Click **▶ Initialize Pipeline** in the sidebar."
                sources_html = ""
            else:
                try:
                    response_text, sources_html = run_query_with_memory(prompt)
                except Exception as e:
                    response_text = f"⚠️ Error during retrieval: {e}"
                    sources_html = ""
                    logger.error(f"Query failed: {e}")

        st.markdown(response_text)

        if sources_html:
            st.markdown(
                f'<div style="margin-top:0.75rem;"><span style="font-size:0.7rem;color:#555;font-family:\'IBM Plex Mono\',monospace;">SOURCES </span>{sources_html}</div>',
                unsafe_allow_html=True,
            )

    full_response = response_text
    if sources_html:
        full_response += f"\n\n{sources_html}"
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# Placeholder when pipeline not ready
if not st.session_state.pipeline_ready and not st.session_state.chat_history:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 4rem 2rem;
        color: #444;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        border: 1px dashed #2a2a2a;
        border-radius: 8px;
        margin-top: 2rem;
    ">
        <div style="font-size:2.5rem;margin-bottom:1rem;">⚙️</div>
        Initialize the pipeline from the sidebar to start querying the catalog.
    </div>
    """, unsafe_allow_html=True)