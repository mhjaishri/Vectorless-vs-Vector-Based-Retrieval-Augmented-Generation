# Legal AI RAG Evaluation — Setup & Metric Testing Instructions

> **Scope:** Get both RAG systems running, set up the evaluation pipeline, and validate it with demo QnA. QnA dataset creation is out of scope for now and will be handled separately.

---

## Repo Structure (Expected)

```
project-root/
├── docker-compose.yml          # Qdrant + Neo4j local instances
├── INSTRUCTIONS.md             # This file
├── eval/
│   ├── runner.py               # Main evaluation runner
│   ├── metrics.py              # Precision, Recall, Faithfulness, Correctness
│   ├── visualize.py            # Bar/radar chart generation per metric
│   ├── demo_qna.json           # Demo QnA dataset (replace later)
│   └── results/                # Output JSONs + chart images
├── rag-vector-graph/           # Existing Vector + GraphRAG chatbot (Constitution only)
│   └── cli.py                  # CLI entrypoint (do not modify core logic)
└── rag-pageindex/              # Existing PageIndex chatbot (Constitution + BNS + BNSS)
    └── cli.py                  # CLI entrypoint (do not modify core logic)
```

---

## 1. Infrastructure — Docker

Both systems may need Qdrant (vector store) and Neo4j (graph store). Start them before running either chatbot.

```bash
docker compose up -d
```

### docker-compose.yml

Already placed at project root. Services:
- **Qdrant** on `localhost:6333`
- **Neo4j** on `localhost:7474` (HTTP) and `localhost:7687` (Bolt)

Default Neo4j credentials: `neo4j / password` (configurable via `.env`).

> If either chatbot does not use one of these services, simply ignore that service — it won't interfere.

---

## 2. Getting Both Chatbots Running

### Step 1 — Check dependencies

For each chatbot directory, read its existing `requirements.txt` or `package.json` and install dependencies. Do not modify core RAG logic.

Only permitted modifications to existing chatbots:
- Add a **retrieval adapter** — a thin wrapper that exposes two functions: `retrieve(query) -> list[str]` and `answer(query) -> str`. Wire this to the existing internals without changing them.
- Add environment variable support (`.env`) for service URLs if hardcoded values exist.

### Step 2 — Smoke test each chatbot

Run a single hardcoded query against each CLI and confirm a response is returned:

```bash
# Vector + GraphRAG
cd rag-vector-graph
python cli.py --query "What does Article 21 of the Constitution guarantee?"

# PageIndex
cd rag-pageindex
python cli.py --query "What does Article 21 of the Constitution guarantee?"
```

Both must return a non-empty answer before proceeding to evaluation.

---

## 3. Evaluation Dataset Format

Use **JSON**. Simple, human-readable, easy to replace later.

### Schema — `eval/demo_qna.json`

```json
[
  {
    "id": "q001",
    "question": "What does Article 21 of the Constitution guarantee?",
    "ground_truth_answer": "Article 21 guarantees the right to life and personal liberty to all persons.",
    "ground_truth_sources": ["Article 21"],
    "type": "direct_lookup"
  },
  {
    "id": "q002",
    "question": "What are the reasonable restrictions on freedom of speech under the Constitution?",
    "ground_truth_answer": "Article 19(2) permits restrictions on freedom of speech on grounds including sovereignty, security, public order, decency, and contempt of court.",
    "ground_truth_sources": ["Article 19", "Article 19(2)"],
    "type": "conceptual"
  },
  {
    "id": "q003",
    "question": "How do Articles 14, 19, and 21 form the golden triangle?",
    "ground_truth_answer": "Articles 14, 19, and 21 together protect equality, fundamental freedoms, and right to life. Courts have held that any law affecting one must satisfy the others.",
    "ground_truth_sources": ["Article 14", "Article 19", "Article 21"],
    "type": "multi_hop"
  }
]
```

**Field definitions:**

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique question ID |
| `question` | string | The query sent to both chatbots |
| `ground_truth_answer` | string | Reference answer for correctness scoring |
| `ground_truth_sources` | list[string] | Article/section labels that must be in retrieved context |
| `type` | string | `direct_lookup`, `conceptual`, or `multi_hop` |

> When replacing with the real dataset later, keep this exact schema. Only the contents change.

---

## 4. Retrieval Adapter Interface

Each chatbot needs to expose this interface so the eval runner can call them uniformly. Add this as a thin wrapper file in each chatbot's directory — do not touch core logic.

```python
# rag-vector-graph/adapter.py  (and same pattern for rag-pageindex/adapter.py)

def retrieve(query: str) -> list[str]:
    """
    Call the chatbot's internal retrieval and return
    a list of retrieved text chunks/passages.
    """
    # Wire to existing internals here
    raise NotImplementedError

def answer(query: str) -> str:
    """
    Return the chatbot's final generated answer for a query.
    """
    # Wire to existing internals here
    raise NotImplementedError
```

If the existing chatbot is not Python, expose the same interface via subprocess or a lightweight HTTP call — the eval runner will handle either pattern.

---

## 5. Metrics Implementation

All four metrics are computed in `eval/metrics.py`. Use an LLM-as-judge for faithfulness and correctness — call whichever LLM API is already in use by the project. If none, default to the Anthropic API.

### 5.1 Context Precision

```
Precision = relevant_chunks_retrieved / total_chunks_retrieved
```

For each query, ask the LLM judge:
> "Given this query: {query}, is the following passage relevant? Reply YES or NO only."

```python
def context_precision(query, retrieved_chunks, llm_judge) -> float:
    relevant = sum(1 for c in retrieved_chunks if llm_judge(query, c) == "YES")
    return relevant / len(retrieved_chunks) if retrieved_chunks else 0.0
```

### 5.2 Context Recall

```
Recall = ground_truth_sources_found / total_ground_truth_sources
```

Check whether each `ground_truth_source` label from the QnA entry appears in the retrieved chunk texts (substring or fuzzy match).

```python
def context_recall(ground_truth_sources, retrieved_chunks) -> float:
    retrieved_text = " ".join(retrieved_chunks).lower()
    found = sum(1 for src in ground_truth_sources if src.lower() in retrieved_text)
    return found / len(ground_truth_sources) if ground_truth_sources else 0.0
```

### 5.3 Faithfulness

```
Faithfulness = supported_claims / total_claims_in_answer
```

Step 1 — Decompose the answer into atomic claims via LLM:
> "Break this answer into individual factual claims, one per line: {answer}"

Step 2 — Verify each claim against retrieved context via LLM:
> "Is this claim: '{claim}' supported by this context: '{context}'? Reply YES or NO only."

```python
def faithfulness(answer, retrieved_chunks, llm_judge) -> float:
    context = "\n".join(retrieved_chunks)
    claims = llm_judge(f"Break into atomic claims, one per line:\n{answer}")
    claims = [c.strip() for c in claims.split("\n") if c.strip()]
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if llm_judge(c, context) == "YES")
    return supported / len(claims)
```

### 5.4 Answer Correctness

```
Correctness = 0.5 × Token_F1 + 0.5 × LLM_judge_score
```

Token F1:

```python
def token_f1(generated: str, reference: str) -> float:
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    common = gen_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
```

LLM judge score (0.0–1.0):
> "Score how well the generated answer matches the reference answer on a scale of 0 to 10. Reply with the number only.\nGenerated: {generated}\nReference: {reference}"

```python
def answer_correctness(generated, reference, llm_judge) -> float:
    f1 = token_f1(generated, reference)
    score_str = llm_judge(generated, reference)  # returns "0"–"10"
    llm_score = int(score_str.strip()) / 10
    return 0.5 * f1 + 0.5 * llm_score
```

---

## 6. Evaluation Runner

`eval/runner.py` orchestrates everything.

**What it does:**
1. Loads `demo_qna.json`
2. For each question, calls both chatbot adapters
3. Computes all 4 metrics for each system per question
4. Averages scores across all questions
5. Saves per-question results to `eval/results/results.json`
6. Calls `visualize.py` to generate charts

**Run:**
```bash
cd eval
python runner.py --qna demo_qna.json --output results/
```

**Results JSON schema:**
```json
{
  "vector_graph_rag": {
    "context_precision": 0.0,
    "context_recall": 0.0,
    "faithfulness": 0.0,
    "answer_correctness": 0.0,
    "per_question": []
  },
  "pageindex_rag": {
    "context_precision": 0.0,
    "context_recall": 0.0,
    "faithfulness": 0.0,
    "answer_correctness": 0.0,
    "per_question": []
  }
}
```

---

## 7. Graphical Comparison

`eval/visualize.py` generates two charts saved to `eval/results/`:

### Chart 1 — Grouped Bar Chart (Overall)
Compares both systems across all 4 metrics side by side.

- X-axis: Metric names
- Y-axis: Score (0.0 – 1.0)
- Two bars per metric: Vector+GraphRAG (blue), PageIndex (orange)
- Save as: `results/comparison_bar.png`

### Chart 2 — Radar Chart
Gives a holistic shape-of-performance view across all 4 metrics.

- 4 axes: Precision, Recall, Faithfulness, Correctness
- Two overlapping polygons: one per system
- Save as: `results/comparison_radar.png`

### Chart 3 — Per-Question Line Chart (optional but useful)
Shows metric score per question for each system to reveal where one wins/loses.

- X-axis: Question IDs
- Y-axis: Score per metric (one chart per metric, 4 charts total)
- Save as: `results/per_question_{metric}.png`

Use `matplotlib` for all charts. No external chart libraries needed.

---

## 8. Validation Checklist

Before running the real dataset, confirm:

- [ ] `docker compose up -d` starts both Qdrant and Neo4j without errors
- [ ] Both chatbot CLIs return answers for the smoke test query
- [ ] Both adapters (`retrieve()` and `answer()`) return non-empty results
- [ ] `runner.py` completes on `demo_qna.json` (3–5 questions) without errors
- [ ] `results/results.json` is written with scores for both systems
- [ ] `results/comparison_bar.png` and `results/comparison_radar.png` are generated
- [ ] Scores are in range `[0.0, 1.0]` for all metrics

Once all boxes are checked, the pipeline is ready. Replace `demo_qna.json` with the real dataset when ready.

---

## 9. Key Constraints & Reminders

- **Evaluation is Constitution-only** — both systems are tested only on Constitution of India questions since that is the only document the Vector+GraphRAG system covers.
- **Do not modify core RAG logic** of either chatbot — only add the adapter wrapper and `.env` support.
- **No chatbot UI work** — both systems have existing CLIs; evaluation calls them programmatically.
- **LLM judge calls** — batch them where possible to reduce API cost. Cache results to disk so re-runs don't re-call the API.
- **Tech stack for eval** — no constraint. Use whatever is simplest. Python + matplotlib is the default.
- **QnA format is JSON** — schema is fixed. Do not change it. The real dataset will use the same schema.
