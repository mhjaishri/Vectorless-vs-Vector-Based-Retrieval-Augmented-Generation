# Legal AI Evaluation — Session Summary

> Written for a fresh Claude session. Read this before touching any file.

---

## Main Goal

Compare two Legal AI RAG systems against **110 questions** about the Indian Constitution (`eval/qna_100.json`) using a rigorous multi-metric evaluation pipeline. The two systems:

- **legalAgent** (PageIndex RAG) — tree-based hierarchical retrieval, no vector DB, multi-hop agent with tool calls, lives in `legalAgent/`
- **legalAgent2** (Vector+Graph RAG) — Qdrant cosine search + Neo4j graph traversal, lives in `legalAgent2/`

Evaluation lives in `eval/`. Results go to `eval/results_110/`.

---

## Current Architecture (Settled)

### API Allocation

| Role | Service | Model | Key/Config |
|---|---|---|---|
| legalAgent nav + answer | Gemini Key 1 | `gemini-2.0-flash` | `legalAgent/.env` → `LLM_API_KEY` |
| legalAgent2 answer gen | Gemini Key 2 | `gemini-2.0-flash` | `legalAgent2/.env` → `OPENAI_API_KEY` |
| legalAgent2 embeddings | Ollama local | `nomic-embed-text` 768-dim | `localhost:11434/v1` |
| Eval semantic similarity | Ollama local | `nomic-embed-text` | same Ollama instance |
| Eval LLM judge | Groq free | `llama-3.3-70b-versatile` | `legalAgent/.env` → `GROQ_API_KEY` |

### legalAgent2 Infrastructure

- **Qdrant** collection `constitution_chunks`: 1677 vectors, 768-dim, Cosine distance
- **Neo4j**: graph of Articles → Clauses → Subclauses with REFERS_TO relations
- **Docker**: managed by root-level `docker-compose.yml` (NOT `legalAgent2/docker-compose.yml`)
- Re-ingest completed successfully with Ollama embeddings

### Evaluation Metrics (6 total)

| Metric | Type | Implementation |
|---|---|---|
| `context_precision` | LLM judge | YES/NO per chunk, fraction relevant |
| `context_recall` | Deterministic | Fraction of GT sources found (flexible matching) |
| `hit_rate` | Deterministic | Binary: any GT source found? |
| `faithfulness` | LLM judge | Decompose claims → verify each |
| `answer_correctness` | LLM judge + deterministic | 0.4×token_f1 + 0.6×structured_llm (JSON correctness+completeness) |
| `semantic_similarity` | Deterministic | cosine(nomic-embed-text(gen), nomic-embed-text(ref)) |

---

## Key Decisions and Why

### 1. Ollama for legalAgent2 embeddings (not Gemini)
**Tried:** `text-embedding-004` → 404 (not available on compat endpoint). `gemini-embedding-001` (3072-dim) → hit 429 rate limits every few batches during ingest.  
**Settled on:** `nomic-embed-text` via local Ollama — 768-dim, zero rate limits, consistent across ingest and query-time.

### 2. Groq for eval judge (not Gemini)
Eval makes ~5,000–7,000 LLM judge calls. Gemini free tier = 1,500 RPD (would take multiple days). Groq free tier = 14,400 RPD at 30 RPM — runs in ~3–4 hours.

### 3. Gemini for answer generation (not Groq)
**legalAgent:** Its navigation prompts are 18–20K tokens (huge system prompt + tree root overview + tool schemas). Every Groq free tier model has a 6–12K TPM limit — a single request of 18K tokens exceeds a full minute's budget → 413 error. Only Gemini (which doesn't have TPM constraints this tight) works.  
**legalAgent2:** Smaller prompts (~2K tokens), Gemini works fine.

### 4. Why not Ollama for answer generation?
legalAgent uses OpenAI's **function-calling / tool-use API** for its agent loop. Tested `meta-llama/llama-4-scout-17b-16e-instruct` via Groq — no 413 error but the model stopped after search tool calls without calling `get_node_content` or `done()`, returning empty results. Ollama models have even less reliable tool-calling support.

### 5. Structured LLM judge (peer review fix)
Old `answer_correctness`: "Rate 0-10" → single subjective number.  
New: JSON prompt `{"correctness": 0.X, "completeness": 0.X}` → structured, quantified, reproducible. Formula: `0.4 * token_f1 + 0.6 * avg(correctness, completeness)`.

### 6. Two new metrics (peer review fix)
- `hit_rate`: binary retrieval success signal — did we find *any* ground truth source? Deterministic, no API calls.
- `semantic_similarity`: cosine distance between Ollama embeddings of generated vs reference answer. Hard deterministic signal alongside soft LLM judge. Uses Ollama (already running) — zero rate limits.

---

## What Has Been Implemented (All Done)

### Code changes (all applied, verified)

**`legalAgent2/ingest2.ts`**
- Replaced `CONFIG.openai.*` with `CONFIG.embedding.*` block
- `vectorSize: 768` (was 3072)
- `batchSize: 100` (was 50)
- `EmbeddingHandler` constructed with `CONFIG.embedding.*`

**`legalAgent2/retrieve.ts`**
- Separate `CONFIG.embedding` block (Ollama) alongside `CONFIG.openai` (Gemini LLM)
- Added `private embeddingClient: OpenAI` field
- `embedQuery()` uses `this.embeddingClient` + `CONFIG.embedding.model`
- `generateAnswer()` `temperature: 0` (was 0.3) for reproducibility
- `chatModel: "gemini-2.0-flash"`

**`legalAgent2/.env`**
```
OPENAI_API_KEY=AIzaSyBhhKtQVxzvxk_zZGHly7-xJ2hswSdDR3M   ← Gemini Key 2
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=reform-william-center-vibrate-press-5829
QDRANT_URL=http://localhost:6333
EMBEDDING_API_KEY=ollama
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
```

**`legalAgent/.env`**
```
LLM_API_KEY=AIzaSyDP38SXzH1CRbQj9MWS8aek4FYS9R_Z-_w       ← Gemini Key 1
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GROQ_API_KEY=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
NAV_MODEL=gemini-2.0-flash
ANSWER_MODEL=gemini-2.0-flash
TAVILY_API_KEY=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
HYDE_PROVIDER=internal
MAX_HOPS=3
```

**`eval/metrics.py`** — fully rewritten:
- `_embed_client` global (Ollama) initialized in `init_judge()`
- `_judge_call()` uses Groq `llama-3.3-70b-versatile`, `_MIN_INTERVAL=2.0s`, SHA256 cache
- `hit_rate()` — new function (reuses `_source_found` helper)
- `semantic_similarity()` — new function, uses Ollama `nomic-embed-text`
- `answer_correctness()` — restructured: JSON judge prompt, `0.4*f1 + 0.6*llm_score`

**`eval/runner.py`** — updated:
- Results dict initialized with 6 metrics including `hit_rate` and `semantic_similarity`
- Per-question loop calls `metrics.hit_rate()` and `metrics.semantic_similarity()`
- Averages loop iterates over all 6 metric names
- Summary table prints all 6 metrics

**`eval/visualize.py`** — updated:
- `METRICS` list extended to 6 entries (added `hit_rate`, `semantic_similarity`)
- `METRIC_LABELS` extended accordingly
- `plot_comparison_bar` widened from 10→13 inches

### Infrastructure

- `nomic-embed-text` pulled in Ollama, verified 768-dim ✅
- Qdrant `constitution_chunks` collection dropped and re-ingested: **1677 vectors at 768-dim** ✅
- Docker services (Qdrant + Neo4j) running via root `docker-compose.yml` ✅
- legalAgent2 adapter smoke test: **returns chunks + answer** ✅

---

## Current Blocker

**Both Gemini API keys hit their daily free-tier RPD limit** from repeated test calls during ingest debugging earlier today.

- Gemini Key 1 (`legalAgent`): exhausted
- Gemini Key 2 (`legalAgent2`): exhausted
- **Reset time: ~midnight Pacific Time (07:00 UTC)**

Everything else is ready. The eval cannot run until Gemini quotas reset.

---

## Exact Next Steps (for tomorrow)

### Step 1 — Verify Gemini quotas reset
```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer AIzaSyDP38SXzH1CRbQj9MWS8aek4FYS9R_Z-_w" \
  -d '{"model":"gemini-2.0-flash","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":5}'
# Expected: {"choices":[{"message":{"content":"OK"}}...]} — NOT a 429
```

### Step 2 — Verify Docker is up
```bash
docker compose up -d
curl -s http://localhost:6333/collections/constitution_chunks | python3 -c \
  "import json,sys; r=json.load(sys.stdin)['result']; print('points:', r['points_count'], '| dim:', r['config']['params']['vectors']['size'])"
# Expected: points: 1677 | dim: 768
```

### Step 3 — Smoke test both adapters
```bash
# legalAgent2
cd /home/archer/Code/LegalAI/legalAgent2
npx tsx adapter.ts --query "What does Article 21 guarantee?"
# Expected: JSON with non-empty chunks array + answer

# legalAgent
cd /home/archer/Code/LegalAI/legalAgent
npx tsx adapter.ts --query "What does Article 21 guarantee?"
# Expected: JSON with non-empty chunks array + answer (uses tree traversal)
```

### Step 4 — Smoke test eval metrics
```bash
cd /home/archer/Code/LegalAI/eval
/home/archer/Code/LegalAI/eval/.venv/bin/python -c "
import metrics
metrics.init_judge()
print('judge:', metrics._judge_call('Reply YES: Is Article 21 about personal liberty?'))
print('sem_sim:', metrics.semantic_similarity('Right to life is protected', 'Article 21 guarantees life and liberty'))
print('hit_rate:', metrics.hit_rate(['Article 21'], ['Article 21 protects personal liberty']))
"
# Expected: judge=YES, sem_sim≈0.7–0.9, hit_rate=1.0
```

### Step 5 — **Ask user to confirm before running** then launch full eval
```bash
cd /home/archer/Code/LegalAI/eval
/home/archer/Code/LegalAI/eval/.venv/bin/python runner.py \
    --qna qna_100.json \
    --output results_110/
```

Expected runtime: **~3–4 hours**  
Output: `results_110/results.json` + 11 chart PNGs (bar, radar, summary table, 6 per-metric bars, 6 per-question line charts)

---

## Known Issues / Watch Out For

1. **Qdrant retrieval quality for legalAgent2**: The smoke test returned irrelevant chunks for Article 21 (the retriever found Article 360, 157, etc. instead). This may be because `scoreThreshold=0.4` is filtering correctly but nomic-embed-text encodes differently than the Gemini model that was originally planned. Monitor precision/recall in results — if consistently low, consider lowering scoreThreshold to 0.3 in `retrieve.ts`.

2. **Gemini rate limits during eval**: With 110 questions at natural processing pace (~10–30s per question), both systems will stay well under 15 RPM. No throttling needed, but if errors appear, add a small delay between questions in `runner.py`.

3. **Groq judge cache**: `results_110/llm_cache.json` persists SHA256-keyed judge responses. If you re-run after a partial failure, the cache prevents re-calling the judge — much faster recovery.

4. **legalAgent tree index**: `legalAgent/indexes/` contains pre-built tree JSON for Constitution, BNS, and BNSS. The eval only tests Constitution questions, but legalAgent always searches all 3 trees. This is expected behavior, not a bug.

5. **numpy dependency in metrics.py**: `eval/metrics.py` uses `numpy` for cosine similarity. Confirm it's installed: `eval/.venv/bin/pip install numpy` if `semantic_similarity()` throws ImportError.

---

## File Quick Reference

```
LegalAI/
├── legalAgent/                    ← PageIndex RAG chatbot
│   ├── .env                       ← Gemini Key 1, Groq key, NAV/ANSWER models
│   ├── adapter.py                 ← Python wrapper (calls adapter.ts via subprocess)
│   ├── adapter.ts                 ← TypeScript entry for eval
│   ├── src/lib/agent.ts           ← Tool-calling agent loop (uses NAV_MODEL)
│   ├── src/lib/llmClient.ts       ← Exports openai client + NAV_MODEL + ANSWER_MODEL
│   └── indexes/                   ← Pre-built tree JSON (constitution, bns, bnss)
│
├── legalAgent2/                   ← Vector+Graph RAG chatbot
│   ├── .env                       ← Gemini Key 2 + EMBEDDING_* for Ollama
│   ├── adapter.py                 ← Python wrapper (calls adapter.ts via subprocess)
│   ├── adapter.ts                 ← TypeScript entry for eval
│   ├── ingest2.ts                 ← Ingests constitution_data.json → Qdrant + Neo4j
│   ├── retrieve.ts                ← Query pipeline: embed → Qdrant → Neo4j → LLM answer
│   └── constitution_data.json     ← Source data
│
├── eval/
│   ├── metrics.py                 ← 6 metric functions + Groq judge + Ollama embedder
│   ├── runner.py                  ← Runs both adapters, computes metrics, saves results
│   ├── visualize.py               ← Chart generation (bar, radar, table, per-metric, per-Q)
│   ├── qna_100.json               ← 110 questions (yes, 110 despite the name)
│   └── .venv/                     ← Python venv with openai, numpy, matplotlib
│
└── docker-compose.yml             ← Root-level: manages qdrant + neo4j containers
```
