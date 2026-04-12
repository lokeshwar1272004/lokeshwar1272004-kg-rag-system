# 🚀 Setup Guide — Hybrid KG + RAG System

## 1. Prerequisites

- Python 3.10 or higher
- pip or conda
- (Optional) Neo4j Desktop or Docker
- (Optional) Ollama installed locally

---

## 2. Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

---

## 4. Environment Configuration

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your credentials
nano .env   # or use any text editor
```

### Key variables to set:

```env
# Required: Get from https://console.groq.com
GROQ_API_KEY=gsk_your_key_here

# If Neo4j is not available, use NetworkX fallback
USE_NETWORKX_FALLBACK=true

# Optional: Ollama local model
OLLAMA_MODEL=llama3
```

---

## 5. Neo4j Setup (Optional — skip if using NetworkX fallback)

### Option A: Docker (Recommended)
```bash
docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5
```

### Option B: Neo4j Desktop
1. Download from https://neo4j.com/download/
2. Create a new database
3. Start the database
4. Set your password in `.env`

### Option C: Skip Neo4j (Use NetworkX)
```env
USE_NETWORKX_FALLBACK=true
```
NetworkX stores the graph in memory (resets on restart, but works for demos).

---

## 6. Ollama Setup (Optional Fallback LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Start Ollama server
ollama serve
```

---

## 7. Run the System

### Option A: Streamlit UI (Recommended)
```bash
streamlit run ui/streamlit_app.py
# or
python main.py serve
```
Open http://localhost:8501

### Option B: Command Line

```bash
# Check system health
python main.py health

# Run demo with sample data
python main.py demo

# Ingest a file
python main.py ingest --file data/sample_documents/tech_companies.txt

# Ask a question
python main.py query "Who founded OpenAI?"
python main.py query "What is machine learning?" --type VECTOR
python main.py query "Tell me everything about Elon Musk" --type HYBRID

# Show retrieved context
python main.py query "Who is the CEO of Google?" --show-context
```

---

## 8. Quick Start (5 minutes)

```bash
# 1. Setup (assuming .env is configured)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Launch UI
streamlit run ui/streamlit_app.py

# 3. In the UI:
#    - Go to "📥 Ingest Documents" → "📚 Sample Data" → Click "Load Sample Data"
#    - Go to "💬 Query" → Ask: "Who founded OpenAI?"
#    - Go to "🕸️ Knowledge Graph" to see the visual graph
```

---

## 9. Common Errors & Fixes

### ❌ `ModuleNotFoundError: No module named 'spacy'`
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### ❌ `OSError: [E050] Can't find model 'en_core_web_sm'`
```bash
python -m spacy download en_core_web_sm
```

### ❌ Groq API authentication error
- Verify your `GROQ_API_KEY` in `.env`
- Get a key from https://console.groq.com/keys
- System will auto-fallback to Ollama

### ❌ Neo4j connection refused
- Either start Neo4j or set `USE_NETWORKX_FALLBACK=true`
- Verify `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` in `.env`

### ❌ ChromaDB errors / corrupted DB
```bash
rm -rf data/chroma_db/
# Restart and re-ingest
```

### ❌ Ollama model not found
```bash
ollama pull llama3
ollama list  # Verify model is available
```

### ❌ `sentence_transformers` slow first run
- Normal behavior — model downloads on first use (~90MB for all-MiniLM-L6-v2)
- Cached after first download

### ❌ Out of memory on embedding
- Reduce `CHUNK_SIZE` in `.env` (e.g., 256 instead of 512)
- Process fewer documents at once

---

## 10. Project Structure

```
kg_rag_system/
├── config/          # Settings & environment management
├── ingestion/       # Document loading, preprocessing, pipeline
├── extraction/      # NER (spaCy) and relation extraction
├── graph/           # Neo4j client, graph builder, Cypher gen, visualizer
├── rag/             # Embeddings (sentence-transformers) + ChromaDB
├── query/           # Classifier, retriever, answer generator, pipeline
├── llm/             # Groq + Ollama router and prompts
├── ui/              # Streamlit application
├── utils/           # Logger, cache, text utilities
├── data/            # ChromaDB, cache, sample docs
├── main.py          # CLI entry point
├── requirements.txt
└── .env.example
```

---

## 11. Architecture Flow

```
User Question
     │
     ▼
[Query Classifier] ──── LLM (Groq→Ollama) ──→ GRAPH / VECTOR / HYBRID
     │
     ▼
[Hybrid Retriever]
  ├── GRAPH: Cypher Generator → Neo4j/NetworkX → Triples
  └── VECTOR: Embedding Query → ChromaDB → Top-K Chunks
     │
     ▼
[Answer Generator] ──── LLM (Groq→Ollama) ──→ Final Answer
     │
     ▼
Streamlit UI / CLI Output
```

---

## 12. Interview-Ready Key Points

| Concept | Explanation |
|---------|-------------|
| **RAG** | Retrieves relevant text chunks from vector DB, feeds to LLM as context |
| **Knowledge Graph** | Stores entity relationships as nodes/edges for structured reasoning |
| **Hybrid** | Uses both: graph for "who/how/what connects", vector for "explain/describe" |
| **Groq→Ollama** | Cloud LLM with local fallback for reliability |
| **spaCy NER** | Identifies entities (Person, Org, Location) to become KG nodes |
| **ChromaDB** | Vector database storing text embeddings for semantic search |
| **Cypher** | Graph query language for Neo4j (like SQL for graphs) |
| **sentence-transformers** | Converts text to dense vectors for similarity search |
