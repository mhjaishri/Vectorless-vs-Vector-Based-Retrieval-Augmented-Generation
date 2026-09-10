# Legal AI — Command Reference

All commands are run from the project root (`/home/archer/Code/LegalAI`) unless a different directory is specified.

---

## Table of Contents

1. [Infrastructure — Docker](#1-infrastructure--docker)
2. [legalAgent — PageIndex RAG (Constitution + BNS + BNSS)](#2-legalagent--pageindex-rag)
3. [legalAgent2 — Vector + Graph RAG (Constitution only)](#3-legalagent2--vector--graph-rag)
4. [Evaluation Pipeline](#4-evaluation-pipeline)
5. [Environment Variables](#5-environment-variables)

---

## 1. Infrastructure — Docker

Both RAG systems require Qdrant (vector store) and Neo4j (graph store) running locally.

```bash
# Start both services in the background
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f

# Stop services
docker compose down
```

**Service endpoints:**

| Service | URL |
|---|---|
| Qdrant (REST) | http://localhost:6333 |
| Qdrant (Dashboard) | http://localhost:6333/dashboard |
| Neo4j (Browser) | http://localhost:7474 |
| Neo4j (Bolt) | bolt://localhost:7687 |

> Neo4j credentials: `neo4j` / `reform-william-center-vibrate-press-5829`

---

## 2. legalAgent — PageIndex RAG

**Documents:** Constitution of India, BNS (Bharatiya Nyaya Sanhita), BNSS (Bharatiya Nagarik Suraksha Sanhita)
**Tech stack:** TypeScript, tsx, OpenAI gpt-4.1-mini (navigation) + gpt-4.1 (answers), HyDE query enrichment

### 2.1 Install dependencies

```bash
cd legalAgent
npm install
```

### 2.2 Configure environment

```bash
cd legalAgent
cp .env.example .env
# Edit .env — set LLM_API_KEY (required), optionally TAVILY_API_KEY
```

**Key `.env` variables:**

```
LLM_API_KEY=<gemini-api-key>
LLM_BASE_URL=https://generativelanguage.googleapis.com/openai/
NAV_MODEL=gemini-2.0-flash     # model for tree navigation + HyDE
ANSWER_MODEL=gemini-2.0-flash  # model for final answer generation
HYDE_PROVIDER=internal         # tavily | serper | internal (no web search)
TAVILY_API_KEY=                # only needed when HYDE_PROVIDER=tavily
MAX_HOPS=3                     # retrieval hops per query
```

### 2.3 Generate document indexes (one-time)

Required before first use. Indexes are saved to `legalAgent/indexes/` as `.tree.json` files.

```bash
# Clone PageIndex library (one-time)
git clone https://github.com/VectifyAI/PageIndex /tmp/pageindex
pip install -r /tmp/pageindex/requirements.txt

# Generate all three indexes (constitution, bns, bnss)
cd legalAgent
python3 scripts/generate_index.py
```

> If indexes already exist (`indexes/constitution.tree.json`, `indexes/bns.tree.json`, `indexes/bnss.tree.json`), skip this step.

### 2.4 Run the interactive chatbot

```bash
cd legalAgent
npm start
# or equivalently:
npx tsx src/index.ts
```

Type your question at the `Legal AI >` prompt. Type `exit` to quit.

**Example session:**
```
Legal AI > What does Article 21 of the Constitution guarantee?
Legal AI > Can police arrest someone without a warrant under BNS?
Legal AI > exit
```

### 2.5 Run a single query via adapter (for testing)

```bash
cd legalAgent
npx tsx adapter.ts --query "What does Article 21 of the Constitution guarantee?"
```

Outputs JSON: `{ "chunks": [...], "answer": "..." }`

---

## 3. legalAgent2 — Vector + Graph RAG

**Documents:** Constitution of India only
**Tech stack:** TypeScript, tsx, Qdrant (vector search), Neo4j (graph traversal), OpenAI text-embedding-3-small + gpt-4o-mini

> **Requires Docker services running** (Qdrant + Neo4j) before any step below.

### 3.1 Install dependencies

```bash
cd legalAgent2
npm install
```

### 3.2 Configure environment

```bash
# Edit legalAgent2/.env — OPENAI_API_KEY is already set
```

**Key `.env` variables:**

```
OPENAI_API_KEY=<openai-api-key>
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=reform-william-center-vibrate-press-5829
QDRANT_URL=http://localhost:6333
```

### 3.3 Ingest data into Qdrant + Neo4j (one-time)

Reads `legalAgent2/constitution_data.json`, creates embeddings, and populates both stores. Takes ~5–10 minutes (501 articles, ~1677 chunks).

```bash
cd legalAgent2
npx tsx ingest2.ts
```

> Re-running is safe — Qdrant collection is recreated from scratch each time.

### 3.4 Run the interactive chatbot

```bash
cd legalAgent2
npx tsx retrieve.ts
```

Type your question at the `💬 Your question:` prompt. Type `exit` or `quit` to quit.

**Example session:**
```
💬 Your question: What does Article 21 of the Constitution guarantee?
💬 Your question: What are the fundamental rights under Part III?
💬 Your question: exit
```

### 3.5 Run a single query via adapter (for testing)

```bash
cd legalAgent2
npx tsx adapter.ts --query "What does Article 21 of the Constitution guarantee?"
```

Outputs JSON: `{ "chunks": [...], "answer": "..." }`

### 3.6 Diagnostic check (verify Qdrant + Neo4j connectivity)

```bash
cd legalAgent2
npx tsx diagnostic.ts
```

---

## 4. Evaluation Pipeline

The eval pipeline runs both systems against a QnA JSON file and computes four metrics: **context precision**, **context recall**, **faithfulness**, and **answer correctness**.

**Eval directory:** `eval/`
**Python venv:** `eval/.venv/` (already set up)

### 4.1 QnA file format

```json
[
  {
    "id": "q001",
    "question": "What does Article 21 of the Constitution guarantee?",
    "ground_truth_answer": "Article 21 guarantees the right to life and personal liberty.",
    "ground_truth_sources": ["Article 21"],
    "type": "direct_lookup"
  }
]
```

- Place the custom dataset at `eval/qna_100.json` (or any path, pass with `--qna`)
- `type` is one of: `direct_lookup`, `conceptual`, `multi_hop`

### 4.2 Run evaluation on the demo dataset (5 questions)

```bash
cd eval
/home/archer/Code/LegalAI/eval/.venv/bin/python runner.py \
    --qna demo_qna.json \
    --output results/
```

### 4.3 Run evaluation on a custom dataset

```bash
cd eval
/home/archer/Code/LegalAI/eval/.venv/bin/python runner.py \
    --qna qna_100.json \
    --output results_100/
```

### 4.4 Re-run with fresh LLM judge cache

The LLM judge results are cached to avoid redundant API calls. Delete the cache to force re-evaluation:

```bash
rm eval/results/llm_cache.json        # for default results/ directory
# or
rm eval/results_100/llm_cache.json    # for a named output directory

cd eval
/home/archer/Code/LegalAI/eval/.venv/bin/python runner.py \
    --qna qna_100.json \
    --output results_100/
```

### 4.5 Output files

After a successful run, the output directory contains:

| File | Description |
|---|---|
| `results.json` | Per-question + averaged scores for both systems |
| `llm_cache.json` | Cached LLM judge calls (do not delete unless re-running) |
| `comparison_bar.png` | Grouped bar chart — 4 metrics × 2 systems |
| `comparison_radar.png` | Radar/spider chart — shape-of-performance view |
| `per_question_context_precision.png` | Per-question line chart |
| `per_question_context_recall.png` | Per-question line chart |
| `per_question_faithfulness.png` | Per-question line chart |
| `per_question_answer_correctness.png` | Per-question line chart |

### 4.6 Smoke test both adapters individually

```bash
# legalAgent adapter
cd legalAgent
npx tsx adapter.ts --query "What does Article 21 guarantee?"

# legalAgent2 adapter
cd legalAgent2
npx tsx adapter.ts --query "What does Article 21 guarantee?"
```

Both should print a JSON object with non-empty `chunks` and `answer` fields.

---

## 5. Environment Variables

### legalAgent (`legalAgent/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEY` | Yes | — | Gemini API key |
| `LLM_BASE_URL` | Yes | — | `https://generativelanguage.googleapis.com/openai/` |
| `NAV_MODEL` | No | `gemini-2.0-flash` | Model for tree navigation and HyDE |
| `ANSWER_MODEL` | No | `gemini-2.0-flash` | Model for final answer generation |
| `HYDE_PROVIDER` | No | `tavily` | `tavily` \| `serper` \| `internal` |
| `TAVILY_API_KEY` | If HYDE_PROVIDER=tavily | — | Tavily web search key |
| `SERPER_API_KEY` | If HYDE_PROVIDER=serper | — | Google search via Serper |
| `MAX_HOPS` | No | `3` | Max retrieval hops per query |

### legalAgent2 (`legalAgent2/.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Gemini API key (used as OpenAI-compat key) |
| `OPENAI_BASE_URL` | Yes | — | `https://generativelanguage.googleapis.com/openai/` |
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | No | `reform-william-center-vibrate-press-5829` | Neo4j password |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant REST endpoint |
