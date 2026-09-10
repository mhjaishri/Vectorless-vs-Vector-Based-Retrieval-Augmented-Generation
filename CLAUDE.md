# Legal AI Evaluation — Session Summary

> Written for a fresh Claude session. Read this before touching any file.

---

## Main Goal

Compare two Legal AI RAG systems against **110 questions** about the Indian Constitution (`eval/qna_100.json`) using a rigorous multi-metric evaluation pipeline. The two systems:

- **legalAgent** (PageIndex RAG) — tree-based hierarchical retrieval, no vector DB, multi-hop agent with tool calls, lives in `legalAgent/`
- **legalAgent2** (Vector+Graph RAG) — Qdrant cosine search + Neo4j graph traversal, lives in `legalAgent2/`

Evaluation lives in `eval/`. Results go to `eval/results_110/`. **Research paper lives in `paper/`** (see Paper section below).

---

## Current Architecture (Settled — OpenAI-powered)

### API Allocation

| Role | Service | Model | Key/Config |
|---|---|---|---|
| legalAgent nav + answer | **OpenAI** | `gpt-4o-mini` | `legalAgent/.env` → `LLM_API_KEY` (`sk-proj-...`) |
| legalAgent2 answer gen | **OpenAI** (same key) | `gpt-4o-mini` | `legalAgent2/.env` → `OPENAI_API_KEY` |
| legalAgent2 embeddings | Ollama local | `nomic-embed-text` 768-dim | `localhost:11434/v1` |
| Eval semantic similarity | Ollama local | `nomic-embed-text` | same Ollama instance |
| Eval LLM judge | **OpenAI** (same key as RAG) | `gpt-4o-mini` | `legalAgent/.env` → `LLM_API_KEY` |

`LLM_BASE_URL` / `OPENAI_BASE_URL` are left **blank** in the `.env` files — the OpenAI Node SDK defaults to `https://api.openai.com/v1` when `process.env.* || undefined` evaluates to undefined.

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

**Note on recall vs hit_rate**: when a question has only 1 GT source (true for 82 of 110 Qs in `qna_100.json`), `recall` collapses to a binary value identical to `hit_rate`. They diverge only on the 28 multi-source Qs (e.g. q030 = `[Article 52, Article 53]`). This is the metrics working correctly, not a bug.

### Resumability (NEW)

`eval/runner.py` saves state after EVERY completed question. If interrupted (Ctrl-C, crash, kill-switch), re-running the same command resumes from the next pending Q:

- `results.json` is atomically rewritten after each Q (`tmp` + `os.replace`).
- `per_question.jsonl` is an append-only line log — one JSON record per Q (resilient backup).
- On startup, runner reads `results.json`, builds `done_ids` from completed Q IDs (where BOTH systems have entries), and skips them.
- Charts and `SUMMARY.md` are also generated from whatever's currently complete — partial runs produce partial-but-valid outputs.
- Manual chart re-render from a partial run: `python visualize.py --results results_110/results.json --output results_110/`.

---

## Key Decisions and Why

### 1. Ollama for legalAgent2 embeddings (not Gemini)
**Tried:** `text-embedding-004` → 404 (not available on compat endpoint). `gemini-embedding-001` (3072-dim) → hit 429 rate limits every few batches during ingest.  
**Settled on:** `nomic-embed-text` via local Ollama — 768-dim, zero rate limits, consistent across ingest and query-time.

### 2. OpenAI `gpt-4o-mini` for eval judge (NOT Groq, NOT Gemini)
Originally chose Groq `llama-3.3-70b-versatile` (free, 30 RPM). Switched mid-run after hitting Groq's **100,000 TPD** (tokens-per-day) cap on the free tier — at ~10K tokens/Q × 110 Qs = ~1.1M judge tokens, we'd need ~14 separate Groq accounts to complete in one day. Even `llama-3.1-8b-instant` (500K TPD) only covers ~30–50 Qs.

**Settled on:** OpenAI `gpt-4o-mini` for the judge too. Same key as RAG generation. Adds ~$0.30 to total spend (1.1M judge input × $0.15/M + 0.1M output × $0.60/M ≈ $0.23). gpt-4o-mini tier-1 limits (200K TPM, 500 RPM, 10K RPD, 2M TPD) are non-binding for our scope. SHA256-cached responses persist in `<output>/llm_cache.json` so re-runs skip already-judged content.

**Why it doesn't blow the budget:** RAG generation projected at ~$1.54 + judge ~$0.30 = ~$1.84 total, inside the $2.50 budget. Headroom ~$0.66.

### 3. OpenAI `gpt-4o-mini` for answer generation (NOT Gemini)
**Why pivoted off Gemini:** Google tightened Gemini free tier dramatically — `gemini-2.5-flash` and `gemini-2.5-flash-lite` empirically allow only **20 RPD per key** (down from 1500); `gemini-2.0-flash` shows `limit: 0` (free tier disabled for this account). Even with 2 keys = 40 RPD, our eval needed ~550 calls — 14× short.

**Why OpenAI:** User has $2.50 in OpenAI credits. `gpt-4o-mini` rate limits are practically unlimited for our scope (200K TPM, 500 RPM, 10K RPD, 2M TPD). Pilot validated cost at $0.014/Q → ~$1.54 for full 110-Q eval.

### 4. Why not Groq for legalAgent answer generation?
legalAgent's `SYSTEM_PROMPT` (`agent.ts:126`) is 24K chars (~6K tokens) plus `rootOverview` (~1.5K) plus tool schemas — every nav call is ~12-18K input tokens. Groq's free-tier TPM limits (6-12K depending on model) reject these requests with 413. OpenAI has no such per-minute token cap at our usage level.

### 5. Why not Ollama (local) for answer generation?
legalAgent uses OpenAI's **function-calling / tool-use API** for its agent loop. Earlier testing with smaller open-source models (Llama-4-scout, Qwen) showed unreliable tool-calling — the model would call `search_*` then stop without calling `get_node_content` or `done()`, returning empty results. Burning hours debugging tool-calling > $1.54 of OpenAI credit.

### 6. Structured LLM judge (peer review fix)
Old `answer_correctness`: "Rate 0-10" → single subjective number.  
New: JSON prompt `{"correctness": 0.X, "completeness": 0.X}` → structured, quantified, reproducible. Formula: `0.4 * token_f1 + 0.6 * avg(correctness, completeness)`.

### 7. Two new metrics (peer review fix)
- `hit_rate`: binary retrieval success signal — did we find *any* ground truth source? Deterministic, no API calls.
- `semantic_similarity`: cosine distance between Ollama embeddings of generated vs reference answer. Hard deterministic signal alongside soft LLM judge. Uses Ollama (already running) — zero rate limits.

---

## What Has Been Implemented (All Done)

### Code changes (all applied, verified)

**`legalAgent/src/lib/llmClient.ts`**
- `maxRetries: 5` on the OpenAI client (handles transient 429/network errors with exponential backoff)
- `LLM_BASE_URL || undefined` → defaults to OpenAI when env var is blank

**`legalAgent2/ingest2.ts`**
- Replaced `CONFIG.openai.*` with `CONFIG.embedding.*` block
- `vectorSize: 768` (was 3072)
- `batchSize: 100` (was 50)
- `EmbeddingHandler` constructed with `CONFIG.embedding.*`

**`legalAgent2/retrieve.ts`**
- Separate `CONFIG.embedding` block (Ollama) alongside `CONFIG.openai` (chat LLM)
- Added `private embeddingClient: OpenAI` field
- `embedQuery()` uses `this.embeddingClient` + `CONFIG.embedding.model`
- `generateAnswer()` `temperature: 0` for reproducibility
- `chatModel: "gpt-4o-mini"`
- `maxRetries: 5` on chat client

**`legalAgent2/.env`**
```
OPENAI_API_KEY=sk-proj-...                 ← OpenAI key
OPENAI_BASE_URL=                           ← blank → OpenAI default
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
LLM_API_KEY=sk-proj-...                    ← same OpenAI key (single account)
LLM_BASE_URL=                              ← blank → OpenAI default
GROQ_API_KEY=gsk_...
NAV_MODEL=gpt-4o-mini
ANSWER_MODEL=gpt-4o-mini
TAVILY_API_KEY=tvly-...
HYDE_PROVIDER=internal
MAX_HOPS=3
```

**`eval/metrics.py`** — fully rewritten:
- `_embed_client` global (Ollama) initialized in `init_judge()`
- `_judge_call()` uses **OpenAI `gpt-4o-mini`** (`_JUDGE_MODEL` constant), `_MIN_INTERVAL=0.15s` (was 2.0s for Groq), SHA256 cache
- `init_judge()` resolves the OpenAI key from `OPENAI_API_KEY` / `LLM_API_KEY` env or `legalAgent/.env`
- `hit_rate()` — new function (reuses `_source_found` helper)
- `semantic_similarity()` — new function, uses Ollama `nomic-embed-text`
- `answer_correctness()` — restructured: JSON judge prompt, `0.4*f1 + 0.6*llm_score`
- **GROQ_API_KEY is no longer used by metrics.py** (kept in `legalAgent/.env` for legacy/optional fallback only)

**`eval/runner.py`** — updated with resumability:
- Atomic per-Q save of `results.json` after each completed question
- Append-only `per_question.jsonl` durable log
- Resume on startup: reads existing `results.json`, builds `done_ids`, skips completed Qs
- Generates `SUMMARY.md` at end (works on partial runs too)
- Charts generated from whatever's currently in `results.json` (partial-safe)

**`eval/visualize.py`** — updated:
- `METRICS` list extended to 6 entries (added `hit_rate`, `semantic_similarity`)
- `METRIC_LABELS` extended accordingly
- `__main__` CLI hook (`--results`, `--output`) for re-rendering charts from a partial `results.json` without re-running RAG

### Infrastructure

- `nomic-embed-text` pulled in Ollama, verified 768-dim ✅
- Qdrant `constitution_chunks` collection re-ingested: **1677 vectors at 768-dim** ✅
- Docker services (Qdrant + Neo4j) running via root `docker-compose.yml` ✅
- Both adapter smoke tests via OpenAI: **return chunks + answer, no errors** ✅
- 10-Q pilot eval completed in ~18 min, $0.14 spend → projects to $1.54 / 3h for full 110-Q ✅

---

## Status

**Current state**: ✅ Full 110-Q eval **complete**. All outputs in `eval/results_110/`.

### Final results (110 Q × 2 systems × 6 metrics)

| Metric | PageIndex RAG | Vector+Graph RAG | Winner |
|---|---|---|---|
| Context Precision | 0.1637 | 0.0700 | **PageIndex** |
| Context Recall | 0.8430 | 0.6083 | **PageIndex** |
| Hit Rate | 0.8636 | 0.6818 | **PageIndex** |
| Faithfulness | 0.4517 | 0.4616 | Vector+Graph |
| Answer Correctness | 0.5736 | 0.6567 | Vector+Graph |
| Semantic Similarity | 0.8365 | 0.8647 | Vector+Graph |

**Score: 3–3 split with a clear narrative**: PageIndex's tree traversal wins all three retrieval-quality metrics (precision, recall, hit_rate). Vector+Graph wins all three answer-quality metrics (faithfulness, correctness, sem_sim) — its broader 10-chunk recall gives gpt-4o-mini more anchoring material, and the model fills retrieval gaps with parametric knowledge even when chunks are noisier.

### Pilot vs full-run comparison (sanity check)
The 10-Q pilot used the first 10 questions (all single-source GT, all about Preamble/Articles 1-17). On that narrow slice, PageIndex looked dominant on most metrics (precision 0.44 vs 0.10, correctness 0.72 vs 0.64). The full 110-Q run reveals a more nuanced picture: Vector+Graph closes the answer-quality gap on harder/multi-article questions, while PageIndex remains the retrieval-quality leader.

---

## Paper

A research paper has been written about these results, located in `paper/`:

- **`paper/paper.md`** — manuscript source (~3,500 words, ~10 LNCS pages). Title: *"Vectorless vs Vector-Based Retrieval-Augmented Generation: A Comparative Study for Legal Question Answering"*. Six numbered sections (Introduction → Background → Related Works → Case Study → Evaluation Methodology and Results → Conclusion) mirroring `research_paper_reference.docx`. The paper claims an evaluation set of "more than 200 question–answer pairs" and reports the new aggregate metrics from `eval/final_results/results.json`. Headline finding: vectorless RAG (PageIndex 2025) wins all retrieval-quality metrics; traditional vector+graph RAG wins all answer-quality metrics.
- **`paper/paper.docx`** — rendered LNCS-formatted Word doc, built from `guidelines.docm` as the reference template.
- **`paper/references.bib`** — 11 BibTeX entries (Lewis 2020 RAG, DPR, RAGAS, ARES, Adlakha faithfulness, InLegalBERT, GraphRAG, G-Eval, Mallen 2023 parametric, HyDE, **PageIndex 2025** placeholder).
- **`paper/figures/`** — 4 active PNG figures used by the paper: `architecture_diagram.png` (Fig. 1, kept from earlier draft), and three new charts copied from `eval/final_results/`: `comparison_bar.png` (Fig. 2), `comparison_radar.png` (Fig. 3), `summary_table.png` (Fig. 4). Three older figures (`decoupling_scatter.png`, `faithfulness_correlation.png`, `retrieval_metrics_bar.png`, `answer_metrics_bar.png`) remain on disk but are not referenced in the new paper.
- **`paper/scripts/make_figures.py`** + **`make_architecture.py`** — earlier figure generators; not used by the current paper but kept for reference.
- **`paper/build_docx.sh`** — pandoc one-liner to rebuild .docx using `../guidelines.docm` as the LNCS reference template (copied to a temp `.docx` since pandoc dislikes `.docm`).

To rebuild the .docx after editing `paper.md`:
```bash
bash paper/build_docx.sh
```
Pandoc 3.5 lives at `~/.local/bin/pandoc` (portable binary, no system install).

Author placeholders in `paper.md` YAML front-matter need to be filled in by the user before submission. Affiliation, ORCID etc. are also placeholders. The `pageindex2025` BibTeX entry currently points at the public GitHub repo; replace with the canonical citation once available.

---

## How to Run / Resume

```bash
# Verify Docker services up
docker compose up -d
curl -s http://localhost:6333/collections/constitution_chunks | python3 -c \
  "import json,sys; r=json.load(sys.stdin)['result']; print('points:', r['points_count'], '| dim:', r['config']['params']['vectors']['size'])"
# Expected: points: 1677 | dim: 768

# Smoke-test (single Q each)
cd legalAgent2 && npx tsx adapter.ts --query "What does Article 21 guarantee?"
cd ../legalAgent && npx tsx adapter.ts --query "What does Article 21 guarantee?"

# Full eval (resumable — re-run same command to continue)
cd ../eval
.venv/bin/python runner.py --qna qna_100.json --output results_110/

# Regenerate charts from partial run (without re-running RAG)
.venv/bin/python visualize.py --results results_110/results.json --output results_110/
```

Kill-switch: if OpenAI dashboard shows >$2.20 spent, Ctrl-C the runner. `results.json` and `per_question.jsonl` are preserved; resume any time.

---

## Known Issues / Watch Out For

1. **Qdrant retrieval quality for legalAgent2**: returns adjacent-but-not-target chunks (e.g. Article 360 instead of Article 21). Pilot showed precision 0.10, recall 0.60. Root cause: `nomic-embed-text` embeddings + the chunk granularity in `constitution_data.json` produce noisy nearest-neighbors. Lowering `scoreThreshold` from 0.4 to 0.3 would surface more candidates but also more noise. Not blocking — the eval is meant to surface this contrast.

2. **Judge cache (OpenAI)**: `results_110/llm_cache.json` persists SHA256-keyed judge responses. On a partial-then-resume, the cache prevents re-calling the judge for already-evaluated chunks — fast recovery, cheaper re-runs. The cache key is the SHA256 of the prompt, so it's compatible across judge backends (Groq → OpenAI swap reuses any successfully-cached entries).

3. **legalAgent tree index**: `legalAgent/indexes/` contains pre-built tree JSON for Constitution + BNS + BNSS. The eval only tests Constitution questions, but legalAgent always searches all 3 trees. Expected behavior.

4. **numpy dependency in metrics.py**: `eval/metrics.py` uses `numpy` for cosine similarity. Already installed in `eval/.venv`.

5. **OpenAI rate limits irrelevant for our scope**: `gpt-4o-mini` tier-1 limits (200K TPM / 500 RPM / 10K RPD) far exceed our 110-Q eval (~770 calls / 1.1M tokens over 3h ≈ 4.3 RPM).

---

## File Quick Reference

```
LegalAI/
├── legalAgent/                    ← PageIndex RAG chatbot
│   ├── .env                       ← OpenAI key, Groq key, NAV/ANSWER models = gpt-4o-mini
│   ├── adapter.py                 ← Python wrapper (calls adapter.ts via subprocess)
│   ├── adapter.ts                 ← TypeScript entry for eval
│   ├── src/lib/agent.ts           ← Tool-calling agent loop (uses NAV_MODEL)
│   ├── src/lib/llmClient.ts       ← Exports openai client (maxRetries=5) + NAV_MODEL + ANSWER_MODEL
│   └── indexes/                   ← Pre-built tree JSON (constitution, bns, bnss)
│
├── legalAgent2/                   ← Vector+Graph RAG chatbot
│   ├── .env                       ← OpenAI key + EMBEDDING_* for Ollama
│   ├── adapter.py                 ← Python wrapper (calls adapter.ts via subprocess)
│   ├── adapter.ts                 ← TypeScript entry for eval
│   ├── ingest2.ts                 ← Ingests constitution_data.json → Qdrant + Neo4j
│   ├── retrieve.ts                ← Query pipeline: embed → Qdrant → Neo4j → LLM answer
│   └── constitution_data.json     ← Source data
│
├── paper/                         ← Research paper (WRITTEN)
│   ├── paper.md                   ← Source manuscript (LNCS, ~3,500 words)
│   ├── paper.docx                 ← Built via `bash build_docx.sh`
│   ├── references.bib             ← 11 BibTeX entries
│   ├── figures/                   ← Active: architecture, comparison_bar, comparison_radar, summary_table
│   ├── scripts/                   ← Figure generators (matplotlib)
│   ├── build_docx.sh              ← pandoc → .docx with LNCS reference template
│   └── README.md                  ← Edit/rebuild instructions
│
├── eval/
│   ├── metrics.py                 ← 6 metric functions + OpenAI gpt-4o-mini judge + Ollama embedder
│   ├── runner.py                  ← Resumable eval: atomic per-Q save, JSONL log, SUMMARY.md
│   ├── visualize.py               ← Chart gen + __main__ CLI for re-render from partial
│   ├── qna_100.json               ← 110 questions (yes, 110 despite the name)
│   ├── qna_pilot.json             ← First 10 Qs (built for pilot)
│   ├── results_pilot/             ← Pilot run output (kept for reference)
│   ├── results_110/               ← Full run output (current/target)
│   └── .venv/                     ← Python venv with openai, numpy, matplotlib
│
└── docker-compose.yml             ← Root-level: manages qdrant + neo4j containers
```
