"""
RAG evaluation metrics for Legal AI comparison.

Metrics:
  1. context_precision    — relevant_chunks / total_chunks (LLM judge)
  2. context_recall       — ground_truth_sources_found / total_gt_sources
  3. hit_rate             — 1.0 if any GT source found, else 0.0 (deterministic)
  4. faithfulness         — supported_claims / total_claims (LLM judge)
  5. answer_correctness   — 0.4 * token_f1 + 0.6 * structured_llm_score
  6. semantic_similarity  — cosine(nomic-embed-text(gen), nomic-embed-text(ref))

LLM judge uses OpenAI gpt-4o-mini (same key used for RAG generation).
Semantic similarity uses local Ollama nomic-embed-text (768-dim, no rate limits).
Results are cached to avoid re-calling on reruns.

Rate limiting: gpt-4o-mini free/tier-1 = 500 RPM, plenty of headroom.
Calls are throttled to one per _MIN_INTERVAL seconds (defensive, not binding).
SHA256-cached responses persist in <output>/llm_cache.json so re-runs are cheap.
"""
import json
import os
import hashlib
import re
import time

import numpy as np
from openai import OpenAI

_client: OpenAI | None = None
_embed_client: OpenAI | None = None
_cache: dict = {}
_cache_file: str = ""
_last_call_time: float = 0.0

# OpenAI gpt-4o-mini: 500 RPM → minimum ~0.12 s between calls; keep at 0.15 for safety
_MIN_INTERVAL: float = 0.15
_MAX_RETRIES: int = 5
_JUDGE_MODEL: str = "gpt-4o-mini"


def init_judge(cache_path: str = "results/llm_cache.json") -> None:
    """Initialize the OpenAI judge client, Ollama embed client, and load the on-disk cache."""
    global _client, _embed_client, _cache, _cache_file

    # Resolve OpenAI key: env var first, otherwise read from legalAgent/.env (LLM_API_KEY)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "legalAgent", ".env"
        )
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("LLM_API_KEY="):
                        api_key = line.split("=", 1)[1]
                        break

    if not api_key:
        raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY or LLM_API_KEY (in legalAgent/.env).")

    _client = OpenAI(api_key=api_key)  # default base URL = OpenAI

    _embed_client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    _cache_file = cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _cache = json.load(f)


def _save_cache() -> None:
    if _cache_file:
        with open(_cache_file, "w") as f:
            json.dump(_cache, f, indent=2)


def _judge_call(prompt: str, max_tokens: int = 16) -> str:
    """
    Call gpt-4o-mini with caching, rate limiting, and retry on 429.
    Returns raw model response text.
    """
    global _last_call_time

    key = hashlib.sha256(prompt.encode()).hexdigest()
    if key in _cache:
        return _cache[key]

    for attempt in range(_MAX_RETRIES):
        elapsed = time.monotonic() - _last_call_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        try:
            _last_call_time = time.monotonic()
            response = _client.chat.completions.create(
                model=_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            result = response.choices[0].message.content.strip()
            _cache[key] = result
            _save_cache()
            return result

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str
            is_server_err = "500" in err_str or "503" in err_str or "overloaded" in err_str
            if (is_rate_limit or is_server_err) and attempt < _MAX_RETRIES - 1:
                backoff = _MIN_INTERVAL * (2 ** attempt)
                print(f"\n  [judge] rate limit / server error — retry {attempt + 1}/{_MAX_RETRIES - 1} in {backoff:.0f}s")
                time.sleep(backoff)
                continue
            print(f"\n  [judge] non-retryable error: {e}")
            return ""

    return ""


# ── Metric 1: Context Precision ───────────────────────────────────────────────

def context_precision(query: str, retrieved_chunks: list) -> float:
    """
    Fraction of retrieved chunks that are relevant to the query.
    Uses LLM judge: YES / NO per chunk.
    """
    if not retrieved_chunks:
        return 0.0

    relevant = 0
    for chunk in retrieved_chunks:
        prompt = (
            f"Query: {query}\n\n"
            f"Passage: {chunk[:800]}\n\n"
            "Is this passage relevant to answering the query? Reply YES or NO only."
        )
        answer = _judge_call(prompt).upper()
        if "YES" in answer:
            relevant += 1

    return relevant / len(retrieved_chunks)


# ── Metric 2: Context Recall ──────────────────────────────────────────────────

def _source_found(src: str, text: str) -> bool:
    """
    Check whether a ground-truth source label appears in the retrieved text.
    Accepts:
      • exact substring match (e.g. "article 21")
      • bare number at a word boundary (e.g. "21" matches "21. Protection..." or "Article 21")
    """
    if src.lower() in text:
        return True
    m = re.search(r'\b(\d+[A-Za-z]*)\b', src)
    if m:
        num = m.group(1)
        if re.search(rf'\b{re.escape(num)}\b', text):
            return True
    return False


def context_recall(ground_truth_sources: list, retrieved_chunks: list) -> float:
    """
    Fraction of ground-truth source labels found in the retrieved text.
    Uses flexible matching: exact substring OR bare article/section number.
    """
    if not ground_truth_sources:
        return 0.0

    retrieved_text = " ".join(retrieved_chunks).lower()
    found = sum(1 for src in ground_truth_sources if _source_found(src, retrieved_text))
    return found / len(ground_truth_sources)


# ── Metric 3: Hit Rate ────────────────────────────────────────────────────────

def hit_rate(ground_truth_sources: list, retrieved_chunks: list) -> float:
    """1.0 if any ground-truth source appears in retrieved chunks, else 0.0."""
    if not ground_truth_sources or not retrieved_chunks:
        return 0.0
    text = " ".join(retrieved_chunks).lower()
    return 1.0 if any(_source_found(src, text) for src in ground_truth_sources) else 0.0


# ── Metric 4: Faithfulness ────────────────────────────────────────────────────

def faithfulness(answer_text: str, retrieved_chunks: list) -> float:
    """
    Fraction of atomic claims in the answer that are supported by retrieved context.
    Two LLM judge calls: decompose claims, then verify each.
    """
    if not answer_text or not retrieved_chunks:
        return 0.0

    decompose_prompt = (
        "Break the following answer into individual factual claims, one per line. "
        "Each line should be a single, standalone factual statement. "
        "Do not add numbering or bullet points.\n\n"
        f"Answer: {answer_text[:1500]}"
    )
    claims_raw = _judge_call(decompose_prompt, max_tokens=512)
    claims = [c.strip() for c in claims_raw.split("\n") if c.strip()]

    if not claims:
        return 0.0

    context = "\n\n".join(c[:600] for c in retrieved_chunks[:10])

    supported = 0
    for claim in claims[:20]:
        verify_prompt = (
            f"Context:\n{context[:2000]}\n\n"
            f"Claim: {claim}\n\n"
            "Is this claim supported by the context above? Reply YES or NO only."
        )
        verdict = _judge_call(verify_prompt).upper()
        if "YES" in verdict:
            supported += 1

    return supported / len(claims[:20])


# ── Metric 5: Answer Correctness ─────────────────────────────────────────────

def token_f1(generated: str, reference: str) -> float:
    """Token-level F1 between generated and reference answers."""
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    common = gen_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_correctness(generated: str, reference: str) -> float:
    """
    0.4 * token_f1 + 0.6 * structured_llm_score.
    LLM judge returns JSON {"correctness": 0-1, "completeness": 0-1}.
    structured_llm_score = avg(correctness, completeness).
    """
    f1 = token_f1(generated, reference)

    score_prompt = (
        "Compare the generated answer to the reference answer.\n"
        'Reply ONLY with JSON: {"correctness": 0.X, "completeness": 0.X}\n'
        "correctness: Are the stated facts accurate relative to the reference? (0.0-1.0)\n"
        "completeness: Are all key points from the reference covered? (0.0-1.0)\n\n"
        f"Generated: {generated[:600]}\n\nReference: {reference[:600]}"
    )
    raw = _judge_call(score_prompt, max_tokens=64)
    try:
        data = json.loads(raw)
        correctness = float(data.get("correctness", 0.0))
        completeness = float(data.get("completeness", 0.0))
        llm_score = max(0.0, min(1.0, (correctness + completeness) / 2.0))
    except (json.JSONDecodeError, KeyError, ValueError):
        nums = re.findall(r'\d+\.?\d*', raw)
        if nums:
            val = float(nums[0])
            llm_score = val / 10.0 if val > 1.0 else val
            llm_score = max(0.0, min(1.0, llm_score))
        else:
            llm_score = 0.0

    return 0.4 * f1 + 0.6 * llm_score


# ── Metric 6: Semantic Similarity ─────────────────────────────────────────────

def semantic_similarity(generated: str, reference: str) -> float:
    """Cosine similarity between nomic-embed-text embeddings of generated vs reference answer."""
    response = _embed_client.embeddings.create(
        model="nomic-embed-text",
        input=[generated[:512], reference[:512]],
    )
    a = np.array(response.data[0].embedding)
    b = np.array(response.data[1].embedding)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(max(0.0, np.dot(a, b) / norm))
