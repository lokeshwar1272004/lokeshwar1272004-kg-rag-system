"""
ui/streamlit_app.py
Complete Streamlit web application for the Hybrid KG+RAG system.

Features:
- Document upload & ingestion
- Manual text input
- Query interface with type selection
- Knowledge graph visualization
- System health dashboard
- Session history
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="KG + RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --bg-primary: #0E1117;
        --bg-secondary: #1A1D27;
        --accent-blue: #4ECDC4;
        --accent-red: #FF6B6B;
        --accent-yellow: #FFEAA7;
        --text-primary: #FAFAFA;
        --text-muted: #8B8FA8;
        --border: #2D3250;
    }

    .main { background-color: var(--bg-primary); }

    .stApp { font-family: 'Inter', sans-serif; }

    .answer-card {
        background: linear-gradient(135deg, #1A1D27 0%, #252A3A 100%);
        border: 1px solid #4ECDC4;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(78, 205, 196, 0.15);
    }

    .metric-card {
        background: #1A1D27;
        border: 1px solid #2D3250;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-graph { background: #FF6B6B33; color: #FF6B6B; border: 1px solid #FF6B6B; }
    .badge-vector { background: #4ECDC433; color: #4ECDC4; border: 1px solid #4ECDC4; }
    .badge-hybrid { background: #FFEAA733; color: #FFEAA7; border: 1px solid #FFEAA7; }
    .badge-high { background: #82E0AA33; color: #82E0AA; border: 1px solid #82E0AA; }
    .badge-medium { background: #F8C47133; color: #F8C471; border: 1px solid #F8C471; }
    .badge-low { background: #FF6B6B33; color: #FF6B6B; border: 1px solid #FF6B6B; }

    .stat-box {
        background: #1A1D27;
        border-left: 3px solid #4ECDC4;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }

    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; }
    code { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialization ─────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "ingestion_stats" not in st.session_state:
    st.session_state.ingestion_stats = {}
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None


# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load query pipeline (cached across reruns)."""
    from query.pipeline import query_pipeline
    return query_pipeline


@st.cache_resource
def load_ingestion_pipeline():
    """Load ingestion pipeline (cached across reruns)."""
    from ingestion.pipeline import ingestion_pipeline
    return ingestion_pipeline


@st.cache_resource
def load_graph_client():
    from graph.neo4j_client import graph_client
    return graph_client


@st.cache_resource
def load_vector_store():
    from rag.vector_store import vector_store
    return vector_store


def get_system_health():
    """Check health of all system components."""
    health = {}
    try:
        from llm.llm_router import llm_router
        llm_health = llm_router.health_check()
        health["Groq"] = "✅" if llm_health.get("groq") else "❌"
        health["Ollama"] = "✅" if llm_health.get("ollama") else "❌"
        health["Active LLM"] = llm_health.get("active", "None")
    except Exception as e:
        health["LLM"] = f"❌ {e}"

    try:
        vs = load_vector_store()
        stats = vs.get_stats()
        health["ChromaDB"] = f"✅ ({stats['total_chunks']} chunks)"
    except Exception as e:
        health["ChromaDB"] = f"❌ {e}"

    try:
        gc = load_graph_client()
        stats = gc.get_stats()
        health["Graph DB"] = f"✅ {stats['backend']} ({stats['nodes']}N / {stats['edges']}E)"
    except Exception as e:
        health["Graph DB"] = f"❌ {e}"

    return health


def render_query_badge(query_type: str) -> str:
    css_class = f"badge-{query_type.lower()}"
    return f'<span class="badge {css_class}">{query_type}</span>'


def render_confidence_badge(confidence: str) -> str:
    css_class = f"badge-{confidence.lower()}"
    return f'<span class="badge {css_class}">Confidence: {confidence.upper()}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 KG + RAG System")
    st.markdown("*Hybrid Knowledge Graph + RAG*")
    st.divider()

    # System Health
    st.markdown("### 🔍 System Status")
    health = get_system_health()
    for component, status in health.items():
        st.markdown(f"**{component}:** {status}")

    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["💬 Query", "📥 Ingest Documents", "🕸️ Knowledge Graph", "📊 Analytics"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### ⚙️ Query Settings")
    force_type = st.selectbox(
        "Force Query Type",
        ["Auto (Recommended)", "GRAPH", "VECTOR", "HYBRID"],
        help="Override automatic query classification"
    )
    top_k = st.slider("Top-K Results", 1, 10, 5)
    use_llm_relations = st.toggle("LLM Relation Extraction", value=False,
                                   help="Slower but more accurate graph building")

    st.divider()
    if st.button("🗑️ Clear Cache", use_container_width=True):
        from utils.cache import cache_manager
        cache_manager.clear()
        st.success("Cache cleared!")


# ── Main Content ──────────────────────────────────────────────────────────────

# ── PAGE: Query ───────────────────────────────────────────────────────────────
if page == "💬 Query":
    st.markdown("# 💬 Ask a Question")
    st.markdown("Query the knowledge base using natural language.")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="e.g. Who founded OpenAI? What is machine learning? Tell me about Elon Musk.",
            label_visibility="collapsed",
        )
    with col2:
        ask_btn = st.button("🔍 Ask", use_container_width=True, type="primary")

    # Example questions
    st.markdown("**Quick examples:**")
    ex_cols = st.columns(4)
    examples = [
        "Who works at Tesla?",
        "Explain transformers",
        "Everything about OpenAI",
        "How is Musk related to SpaceX?",
    ]
    for i, (col, ex) in enumerate(zip(ex_cols, examples)):
        with col:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                question = ex
                ask_btn = True

    if ask_btn and question:
        pipeline = load_pipeline()

        # Resolve force_type
        forced = None
        if force_type != "Auto (Recommended)":
            from query.classifier import QueryType
            forced = QueryType(force_type)

        with st.spinner("🤔 Thinking..."):
            response = pipeline.run(
                question=question,
                force_query_type=forced,
                top_k=top_k,
            )

        # Store in history
        st.session_state.query_history.insert(0, response)
        st.session_state.last_retrieval = response.retrieval

        # ── Display Answer ──
        st.markdown("---")

        # Metadata row
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        with meta_col1:
            st.markdown(
                f"**Query Type:** {render_query_badge(response.query_type)}",
                unsafe_allow_html=True
            )
        with meta_col2:
            st.markdown(
                f"**Confidence:** {render_confidence_badge(response.confidence)}",
                unsafe_allow_html=True
            )
        with meta_col3:
            st.metric("📄 Doc Sources", response.sources_used)
        with meta_col4:
            st.metric("🕸️ Graph Facts", response.graph_facts_used)

        # Answer card
        st.markdown(
            f'<div class="answer-card"><strong>Answer:</strong><br><br>{response.answer}</div>',
            unsafe_allow_html=True,
        )

        # Show retrieved context
        if response.retrieval:
            with st.expander("🔍 View Retrieved Context"):
                if response.retrieval.graph_results:
                    st.markdown("**Knowledge Graph Facts:**")
                    for r in response.retrieval.graph_results[:10]:
                        subj = r.get("subject", "")
                        pred = r.get("predicate", r.get("relation", ""))
                        obj = r.get("object", "")
                        st.markdown(f"- `{subj}` **—[{pred}]→** `{obj}`")

                if response.retrieval.vector_results:
                    st.markdown("**Semantic Search Results:**")
                    for r in response.retrieval.vector_results[:3]:
                        with st.container():
                            st.markdown(
                                f"<div class='stat-box'><small>Score: {r.score:.3f} | Doc: {r.doc_id}</small><br>{r.content[:300]}...</div>",
                                unsafe_allow_html=True
                            )

        st.info(f"⚡ Answered by **{response.llm_provider}**", icon="🤖")

    # Query History
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 📜 Query History")
        for i, hist in enumerate(st.session_state.query_history[:5]):
            with st.expander(f"Q: {hist.question[:60]}...", expanded=(i == 0)):
                st.markdown(f"**Type:** `{hist.query_type}` | **Confidence:** `{hist.confidence}`")
                st.markdown(hist.answer)


# ── PAGE: Ingest ──────────────────────────────────────────────────────────────
elif page == "📥 Ingest Documents":
    st.markdown("# 📥 Ingest Documents")
    st.markdown("Add documents to the knowledge base (vector store + knowledge graph).")

    tab1, tab2, tab3 = st.tabs(["📎 Upload File", "✏️ Paste Text", "📁 Sample Data"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT file",
            type=["pdf", "txt", "md", "docx"],
            help="Supported: PDF, TXT, MD, DOCX"
        )

        col1, col2 = st.columns(2)
        with col1:
            build_graph_file = st.toggle("Build Knowledge Graph", value=True, key="bg_file")
        with col2:
            use_llm_rel_file = st.toggle("LLM Relations", value=False, key="llm_rel_file",
                                          help="Better relations but slower")

        if uploaded_file and st.button("🚀 Ingest File", type="primary", use_container_width=True):
            ip = load_ingestion_pipeline()
            with st.spinner(f"Processing {uploaded_file.name}..."):
                result = ip.ingest_bytes(
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name,
                    build_graph=build_graph_file,
                    use_llm_relations=use_llm_rel_file,
                )

            if result.get("success"):
                st.success(f"✅ Successfully ingested **{uploaded_file.name}**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📑 Chunks", result.get("chunks", 0))
                c2.metric("🔢 Vectors", result.get("vectors_indexed", 0))
                c3.metric("🔵 Entities", result.get("entities_added", 0))
                c4.metric("🔗 Relations", result.get("relations_added", 0))
                st.session_state.ingestion_stats[uploaded_file.name] = result
            else:
                st.error(f"❌ Ingestion failed: {result.get('error')}")

    with tab2:
        raw_text = st.text_area(
            "Paste your text here",
            height=250,
            placeholder="Paste any text content here — articles, Wikipedia excerpts, company info, etc.",
        )
        source_name = st.text_input("Source name", value="custom_text")

        col1, col2 = st.columns(2)
        with col1:
            build_graph_text = st.toggle("Build Knowledge Graph", value=True, key="bg_text")
        with col2:
            use_llm_rel_text = st.toggle("LLM Relations", value=False, key="llm_rel_text")

        if raw_text and st.button("🚀 Ingest Text", type="primary", use_container_width=True):
            ip = load_ingestion_pipeline()
            with st.spinner("Processing text..."):
                result = ip.ingest_text(
                    text=raw_text,
                    source_name=source_name,
                    build_graph=build_graph_text,
                    use_llm_relations=use_llm_rel_text,
                )

            if result.get("success"):
                st.success("✅ Text ingested successfully!")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📑 Chunks", result.get("chunks", 0))
                c2.metric("🔢 Vectors", result.get("vectors_indexed", 0))
                c3.metric("🔵 Entities", result.get("entities_added", 0))
                c4.metric("🔗 Relations", result.get("relations_added", 0))
            else:
                st.error(f"❌ {result.get('error')}")

    with tab3:
        st.markdown("### 📚 Load Sample Dataset")
        st.markdown("Load pre-built sample data for testing:")

        sample_text = """
        OpenAI is an AI research company founded in December 2015 by Sam Altman, Elon Musk,
        Greg Brockman, Ilya Sutskever, and others. The company is headquartered in San Francisco.
        OpenAI created ChatGPT and the GPT series of large language models.
        Sam Altman serves as the CEO of OpenAI.

        Elon Musk is the CEO of Tesla and SpaceX. He co-founded PayPal and acquired Twitter,
        renaming it to X Corp. Tesla is headquartered in Austin, Texas. SpaceX is located in
        Hawthorne, California. Musk attended Stanford University but left after two days.

        Google was founded by Larry Page and Sergey Brin while they were PhD students at
        Stanford University. Google is a subsidiary of Alphabet Inc. Sundar Pichai is the
        CEO of Google and Alphabet. Google developed TensorFlow and the BERT language model.

        Microsoft was founded by Bill Gates and Paul Allen in 1975. Satya Nadella is the
        current CEO of Microsoft. Microsoft acquired LinkedIn in 2016 and GitHub in 2018.
        Microsoft partnered with OpenAI and invested $10 billion in the company.
        Azure is Microsoft's cloud computing platform.

        Meta Platforms, formerly Facebook, was founded by Mark Zuckerberg at Harvard University.
        Meta owns Instagram, WhatsApp, and Oculus. Mark Zuckerberg is the CEO of Meta.
        Meta is headquartered in Menlo Park, California.
        """

        if st.button("📥 Load Sample Data", type="primary", use_container_width=True):
            ip = load_ingestion_pipeline()
            with st.spinner("Loading sample dataset..."):
                result = ip.ingest_text(
                    text=sample_text,
                    source_name="tech_companies_sample",
                    build_graph=True,
                    use_llm_relations=False,
                )
            if result.get("success"):
                st.success("✅ Sample data loaded! Try asking: 'Who founded OpenAI?' or 'Who is the CEO of Google?'")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📑 Chunks", result.get("chunks", 0))
                c2.metric("🔢 Vectors", result.get("vectors_indexed", 0))
                c3.metric("🔵 Entities", result.get("entities_added", 0))
                c4.metric("🔗 Relations", result.get("relations_added", 0))
            else:
                st.error(f"Failed: {result.get('error')}")

    # Ingestion history
    if st.session_state.ingestion_stats:
        st.divider()
        st.markdown("### 📋 Ingestion History")
        for name, stats in st.session_state.ingestion_stats.items():
            st.markdown(f"- **{name}**: {stats.get('chunks', 0)} chunks, {stats.get('vectors_indexed', 0)} vectors")


# ── PAGE: Knowledge Graph ──────────────────────────────────────────────────────
elif page == "🕸️ Knowledge Graph":
    st.markdown("# 🕸️ Knowledge Graph Explorer")

    gc = load_graph_client()
    stats = gc.get_stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("🔵 Total Nodes", stats["nodes"])
    c2.metric("🔗 Total Edges", stats["edges"])
    c3.metric("🗄️ Backend", stats["backend"])

    if stats["nodes"] == 0:
        st.warning("⚠️ Knowledge Graph is empty. Please ingest documents first.")
    else:
        tab1, tab2, tab3 = st.tabs(["🌐 Full Graph", "🔍 Entity Explorer", "📋 All Triples"])

        with tab1:
            st.markdown("### Full Knowledge Graph (limited to 100 nodes)")
            max_nodes = st.slider("Max nodes to display", 10, 100, 50)

            with st.spinner("Rendering graph..."):
                from graph.visualizer import graph_visualizer
                triples = gc.get_all_triples()

                if triples:
                    html = graph_visualizer.create_from_triples(triples, max_nodes=max_nodes)
                    if html:
                        st.components.v1.html(html, height=550, scrolling=False)
                    else:
                        st.error("Graph rendering failed. Check pyvis installation.")
                else:
                    st.info("No graph triples found.")

        with tab2:
            st.markdown("### Entity-Centric Explorer")
            entity_query = st.text_input("Search for an entity:", placeholder="e.g. OpenAI, Elon Musk")

            if entity_query:
                with st.spinner("Fetching relations..."):
                    relations = gc.query_relations(entity_query)

                if relations:
                    st.markdown(f"Found **{len(relations)}** relations for `{entity_query}`:")
                    from graph.visualizer import graph_visualizer
                    html = graph_visualizer.create_ego_graph(entity_query, relations)
                    if html:
                        st.components.v1.html(html, height=450, scrolling=False)

                    st.markdown("**Relation Table:**")
                    import pandas as pd
                    df = pd.DataFrame(relations)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning(f"No relations found for '{entity_query}'")

        with tab3:
            st.markdown("### All Graph Triples")
            triples = gc.get_all_triples()
            if triples:
                import pandas as pd
                df = pd.DataFrame(triples)
                st.dataframe(df, use_container_width=True, height=400)
                st.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False),
                    file_name="knowledge_graph_triples.csv",
                    mime="text/csv",
                )


# ── PAGE: Analytics ───────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.markdown("# 📊 System Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔢 Vector Store")
        vs = load_vector_store()
        vs_stats = vs.get_stats()
        st.markdown(f"""
        <div class='stat-box'>📦 Total Chunks: <strong>{vs_stats['total_chunks']}</strong></div>
        <div class='stat-box'>🗂️ Collection: <strong>{vs_stats['collection']}</strong></div>
        <div class='stat-box'>💾 Storage: <strong>{vs_stats['persist_dir']}</strong></div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🕸️ Knowledge Graph")
        gc = load_graph_client()
        kg_stats = gc.get_stats()
        st.markdown(f"""
        <div class='stat-box'>🔵 Nodes: <strong>{kg_stats['nodes']}</strong></div>
        <div class='stat-box'>🔗 Edges: <strong>{kg_stats['edges']}</strong></div>
        <div class='stat-box'>🗄️ Backend: <strong>{kg_stats['backend']}</strong></div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🤖 LLM Status")
    from llm.llm_router import llm_router
    health = llm_router.health_check()
    c1, c2, c3 = st.columns(3)
    c1.metric("Groq", "Online" if health["groq"] else "Offline")
    c2.metric("Ollama", "Online" if health["ollama"] else "Offline")
    c3.metric("Active", health["active"])

    st.divider()
    st.markdown("### 📜 Query History")
    if st.session_state.query_history:
        import pandas as pd
        history_data = [
            {
                "Question": h.question[:60],
                "Type": h.query_type,
                "Confidence": h.confidence,
                "Sources": h.sources_used,
                "Graph Facts": h.graph_facts_used,
                "LLM": h.llm_provider,
            }
            for h in st.session_state.query_history
        ]
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
    else:
        st.info("No queries yet.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Vector Store", use_container_width=True):
            vs.clear()
            st.success("Vector store cleared!")
    with col2:
        if st.button("🗑️ Clear Knowledge Graph", use_container_width=True):
            gc.clear()
            st.success("Knowledge graph cleared!")
