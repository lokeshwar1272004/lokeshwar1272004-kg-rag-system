"""
main.py
Command-line entry point for the Hybrid KG+RAG system.
Supports: ingest, query, serve (Streamlit), and demo modes.

Usage:
    python main.py demo                      # Run with sample data
    python main.py ingest --file data.pdf    # Ingest a file
    python main.py ingest --text "..."       # Ingest raw text
    python main.py query "Who founded OpenAI?"
    python main.py serve                     # Launch Streamlit UI
    python main.py health                    # Check system health
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import logger
from config.settings import settings


def cmd_health():
    """Check health of all system components."""
    print("\n" + "=" * 50)
    print("  SYSTEM HEALTH CHECK")
    print("=" * 50)

    # LLM
    from llm.llm_router import llm_router
    health = llm_router.health_check()
    print(f"\n🤖 LLM Providers:")
    print(f"   Groq:   {'✅ Online' if health['groq'] else '❌ Offline'}")
    print(f"   Ollama: {'✅ Online' if health['ollama'] else '❌ Offline'}")
    print(f"   Active: {health['active']}")

    # Vector Store
    from rag.vector_store import vector_store
    vs_stats = vector_store.get_stats()
    print(f"\n🔢 Vector Store (ChromaDB):")
    print(f"   Chunks indexed: {vs_stats['total_chunks']}")
    print(f"   Collection: {vs_stats['collection']}")

    # Knowledge Graph
    from graph.neo4j_client import graph_client
    kg_stats = graph_client.get_stats()
    print(f"\n🕸️  Knowledge Graph ({kg_stats['backend']}):")
    print(f"   Nodes: {kg_stats['nodes']}")
    print(f"   Edges: {kg_stats['edges']}")

    # NER
    from extraction.ner import ner_extractor
    ner_status = "✅ spaCy loaded" if ner_extractor.nlp else "⚠️  Fallback (no spaCy)"
    print(f"\n🏷️  NER: {ner_status}")

    # Embedding
    from rag.embedder import embedding_model
    emb_status = f"✅ {embedding_model.model_name} (dim={embedding_model.dimension})" if embedding_model.model else "❌ Not loaded"
    print(f"\n🔣 Embeddings: {emb_status}")

    print("\n" + "=" * 50 + "\n")


def cmd_ingest(args):
    """Ingest a file or text into the knowledge base."""
    from ingestion.pipeline import ingestion_pipeline

    print(f"\n📥 Ingesting...")

    if args.file:
        result = ingestion_pipeline.ingest_file(
            args.file,
            build_graph=not args.no_graph,
            use_llm_relations=args.llm_relations,
        )
    elif args.text:
        result = ingestion_pipeline.ingest_text(
            args.text,
            source_name=args.source_name or "cli_input",
            build_graph=not args.no_graph,
            use_llm_relations=args.llm_relations,
        )
    elif args.directory:
        result = ingestion_pipeline.ingest_directory(
            args.directory,
            build_graph=not args.no_graph,
        )
    else:
        print("❌ Please provide --file, --text, or --directory")
        return

    if result.get("success"):
        print(f"✅ Ingestion complete!")
        print(f"   📑 Chunks:      {result.get('chunks', 0)}")
        print(f"   🔢 Vectors:     {result.get('vectors_indexed', 0)}")
        print(f"   🔵 Entities:    {result.get('entities_added', 0)}")
        print(f"   🔗 Relations:   {result.get('relations_added', 0)}")
    else:
        print(f"❌ Failed: {result.get('error')}")


def cmd_query(args):
    """Run a single query and print the answer."""
    from query.pipeline import query_pipeline, QueryPipeline
    from query.classifier import QueryType

    question = args.question
    print(f"\n❓ Question: {question}")
    print("⏳ Processing...\n")

    forced = None
    if args.type:
        try:
            forced = QueryType(args.type.upper())
        except ValueError:
            print(f"Invalid type. Use: GRAPH, VECTOR, HYBRID")

    response = query_pipeline.run(
        question=question,
        force_query_type=forced,
        top_k=args.top_k,
    )

    print(f"{'=' * 60}")
    print(f"🏷️  Query Type:   {response.query_type}")
    print(f"📊 Confidence:   {response.confidence}")
    print(f"📄 Doc Sources:  {response.sources_used}")
    print(f"🕸️  Graph Facts:  {response.graph_facts_used}")
    print(f"🤖 LLM:          {response.llm_provider}")
    print(f"{'=' * 60}")
    print(f"\n💡 Answer:\n{response.answer}\n")

    if args.show_context and response.retrieval:
        if response.retrieval.graph_results:
            print("📊 Graph Results:")
            for r in response.retrieval.graph_results[:5]:
                print(f"  {r.get('subject')} --[{r.get('predicate', r.get('relation'))}]--> {r.get('object')}")
        if response.retrieval.vector_results:
            print("\n📄 Top Vector Result:")
            print(f"  {response.retrieval.vector_results[0].content[:200]}...")


def cmd_demo():
    """Load sample data and run demo queries."""
    from ingestion.pipeline import ingestion_pipeline
    from query.pipeline import query_pipeline

    sample_text = """
    OpenAI is an AI research company founded in December 2015 by Sam Altman, Elon Musk,
    Greg Brockman, and Ilya Sutskever. OpenAI is headquartered in San Francisco.
    Sam Altman is the CEO of OpenAI. OpenAI created ChatGPT and the GPT-4 language model.

    Elon Musk is the CEO of Tesla and SpaceX. He co-founded PayPal and OpenAI.
    Tesla is headquartered in Austin, Texas and makes electric vehicles.
    SpaceX is a rocket company located in Hawthorne, California.

    Google was founded by Larry Page and Sergey Brin at Stanford University.
    Sundar Pichai is the CEO of Google and its parent company Alphabet Inc.
    Google developed TensorFlow, BERT, and the Gemini language model.

    Microsoft was founded by Bill Gates and Paul Allen. Satya Nadella is the CEO.
    Microsoft invested $10 billion in OpenAI and partnered with them on Azure OpenAI Service.
    Microsoft acquired LinkedIn in 2016 and GitHub in 2018.
    """

    print("\n🚀 DEMO MODE - Loading sample data...\n")
    result = ingestion_pipeline.ingest_text(
        sample_text, source_name="demo_tech", build_graph=True, use_llm_relations=False
    )
    print(f"✅ Data loaded: {result.get('chunks')} chunks, {result.get('entities_added')} entities\n")

    demo_queries = [
        ("Who founded OpenAI?", "GRAPH"),
        ("What is Tesla?", "VECTOR"),
        ("Tell me everything about Elon Musk.", "HYBRID"),
        ("Who is the CEO of Google?", "GRAPH"),
        ("Explain the relationship between Microsoft and OpenAI.", "HYBRID"),
    ]

    print("=" * 60)
    for question, expected_type in demo_queries:
        print(f"\n❓ Q: {question}")
        response = query_pipeline.run(question, top_k=3)
        print(f"🏷️  Type: {response.query_type} (expected: {expected_type})")
        print(f"💡 A: {response.answer[:300]}...")
        print(f"📊 Confidence: {response.confidence} | LLM: {response.llm_provider}")
        print("-" * 60)


def cmd_serve():
    """Launch the Streamlit UI."""
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    print(f"\n🌐 Launching Streamlit at http://localhost:8501\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(ui_path),
        "--server.port", "8501",
        "--server.headless", "false",
    ])


# ── CLI Parser ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid KG + RAG Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py health
  python main.py demo
  python main.py ingest --file report.pdf
  python main.py ingest --text "OpenAI was founded by Sam Altman..."
  python main.py query "Who founded OpenAI?"
  python main.py query "Explain transformers" --type VECTOR
  python main.py serve
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # health
    subparsers.add_parser("health", help="Check system health")

    # demo
    subparsers.add_parser("demo", help="Load sample data and run demo queries")

    # serve
    subparsers.add_parser("serve", help="Launch Streamlit UI")

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_p.add_argument("--file", type=str, help="Path to PDF/TXT file")
    ingest_p.add_argument("--text", type=str, help="Raw text to ingest")
    ingest_p.add_argument("--directory", type=str, help="Directory to ingest")
    ingest_p.add_argument("--source-name", type=str, default="cli_input")
    ingest_p.add_argument("--no-graph", action="store_true", help="Skip KG building")
    ingest_p.add_argument("--llm-relations", action="store_true", help="Use LLM for relation extraction")

    # query
    query_p = subparsers.add_parser("query", help="Ask a question")
    query_p.add_argument("question", type=str, help="Your question")
    query_p.add_argument("--type", choices=["GRAPH", "VECTOR", "HYBRID"], help="Force query type")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    query_p.add_argument("--show-context", action="store_true", help="Show retrieved context")

    args = parser.parse_args()

    if args.command == "health":
        cmd_health()
    elif args.command == "demo":
        cmd_demo()
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "serve":
        cmd_serve()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
