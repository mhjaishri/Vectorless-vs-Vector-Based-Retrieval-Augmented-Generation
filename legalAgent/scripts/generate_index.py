"""
One-time indexing script: generates PageIndex tree JSON files for each Indian law PDF.

Setup:
    git clone https://github.com/VectifyAI/PageIndex /tmp/pageindex
    pip install -r /tmp/pageindex/requirements.txt
    cp .env.example .env  (set OPENAI_API_KEY or GEMINI_API_KEY)

Run:
    python3 scripts/generate_index.py

Output:
    indexes/constitution.tree.json
    indexes/bns.tree.json
    indexes/bnss.tree.json
"""

import os
import sys
import json
import time
from collections import deque

# PageIndex has no pyproject.toml — add the cloned repo to sys.path directly
PAGEINDEX_REPO = os.environ.get("PAGEINDEX_REPO", "/tmp/pageindex")
if PAGEINDEX_REPO not in sys.path:
    sys.path.insert(0, PAGEINDEX_REPO)

from dotenv import load_dotenv

load_dotenv()

# Map LLM_API_KEY → the provider-specific env var litellm expects.
# litellm routes by model prefix: "gemini/..." needs GEMINI_API_KEY, etc.
_llm_api_key = os.environ.get("LLM_API_KEY", "")
_nav_model = os.environ.get("NAV_MODEL", "gpt-4o-mini")
if _llm_api_key:
    if _nav_model.startswith("gemini/") or _nav_model.startswith("google/"):
        os.environ.setdefault("GEMINI_API_KEY", _llm_api_key)
    elif _nav_model.startswith("anthropic/") or "claude" in _nav_model:
        os.environ.setdefault("ANTHROPIC_API_KEY", _llm_api_key)
    else:
        os.environ.setdefault("OPENAI_API_KEY", _llm_api_key)

# Rate limiting strategy:
#   pageindex/__init__.py does `from .page_index import *` at import time, and
#   page_index.py does `from .utils import *` — so by the time we import pageindex,
#   page_index.py already has its OWN local bindings for llm_completion/llm_acompletion.
#   Patching only pi_utils has no effect on those. We must patch pi_main directly.
#
#   Also: asyncio.Lock can't safely be reused across multiple asyncio.run() calls
#   (each call creates a new event loop). Instead we use asyncio's single-threaded
#   nature: check + append with no `await` between them is atomic, so no lock needed.
#
# Limit math (gpt-4.1-mini tier-1): 200k TPM
#   With max_page_num_each_node=5: ~5-8k tokens/call → safe at 20 calls/min
#   Add 2s minimum gap between calls to prevent bursting.

import asyncio
import pageindex.utils as pi_utils
import pageindex.page_index as pi_main

_RATE_LIMIT = 20       # max calls per 60s window
_MIN_GAP    = 2.0      # minimum seconds between any two calls (burst prevention)
_call_times: deque = deque()
_last_call: float = 0.0

def _sync_wait() -> None:
    global _last_call
    # Enforce minimum gap
    gap = time.time() - _last_call
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)
    # Enforce per-minute cap
    while True:
        now = time.time()
        while _call_times and now - _call_times[0] > 60:
            _call_times.popleft()
        if len(_call_times) < _RATE_LIMIT:
            _call_times.append(now)
            _last_call = now
            break
        sleep_for = 60 - (now - _call_times[0]) + 1.0
        print(f"  [rate limiter] sleeping {sleep_for:.1f}s ({len(_call_times)} calls in last 60s)")
        time.sleep(sleep_for)

async def _async_wait() -> None:
    global _last_call
    # Enforce minimum gap
    gap = time.time() - _last_call
    if gap < _MIN_GAP:
        await asyncio.sleep(_MIN_GAP - gap)
    # Enforce per-minute cap — check+append are atomic in asyncio (no await between them)
    while True:
        now = time.time()
        while _call_times and now - _call_times[0] > 60:
            _call_times.popleft()
        if len(_call_times) < _RATE_LIMIT:
            _call_times.append(now)  # claim slot atomically before any await
            _last_call = now
            break
        sleep_for = 60 - (now - _call_times[0]) + 1.0
        print(f"  [rate limiter] sleeping {sleep_for:.1f}s ({len(_call_times)} calls in last 60s)")
        await asyncio.sleep(sleep_for)

_orig_completion   = pi_utils.llm_completion
_orig_acompletion  = pi_utils.llm_acompletion

def _rate_limited_completion(model, prompt, **kwargs):
    _sync_wait()
    return _orig_completion(model, prompt, **kwargs)

async def _rate_limited_acompletion(model, prompt, **kwargs):
    await _async_wait()
    return await _orig_acompletion(model, prompt, **kwargs)

# Patch in BOTH modules: utils (for any direct callers) and page_index (its own namespace)
pi_utils.llm_completion   = _rate_limited_completion
pi_utils.llm_acompletion  = _rate_limited_acompletion
pi_main.llm_completion    = _rate_limited_completion
pi_main.llm_acompletion   = _rate_limited_acompletion

# Now safe to import the main function
from pageindex import page_index_main
from pageindex.utils import ConfigLoader


DOCS = [
    ("constitution", "constitution.pdf"),
    ("bns", "bns.pdf"),
    ("bnss", "bnss.pdf"),
]

OUTPUT_DIR = "indexes"
INTER_DOC_PAUSE = 30  # seconds between documents to drain rate limit window


def index_document(doc_name: str, pdf_path: str) -> None:
    output_path = os.path.join(OUTPUT_DIR, f"{doc_name}.tree.json")

    if os.path.exists(output_path):
        print(f"[{doc_name}] Already indexed at {output_path} — skipping.")
        return

    if not os.path.exists(pdf_path):
        print(f"[{doc_name}] PDF not found at {pdf_path} — skipping.")
        return

    print(f"\n[{doc_name}] Starting indexing of {pdf_path} ...")

    # Determine model — supports litellm provider prefixes:
    #   OpenAI:  'gpt-4o-mini'
    #   Gemini:  'gemini/gemini-2.0-flash'  (set GEMINI_API_KEY)
    model = os.getenv("NAV_MODEL", "gpt-4o-mini")

    opt = ConfigLoader().load({
        "if_add_node_text": "yes",        # CRITICAL: default is "no" — text absent if omitted
        "if_add_node_id": "yes",
        "model": model,
        "max_page_num_each_node": 5,      # reduce tokens/call (default 10 → ~20k tokens; 5 → ~8k)
    })

    result = page_index_main(pdf_path, opt)
    # result shape: { "doc_name": "bns.pdf", "structure": [...] }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    node_count = _count_nodes(result.get("structure", []))
    print(f"[{doc_name}] Done. {node_count} nodes saved to {output_path}")


def _count_nodes(nodes: list) -> int:
    count = 0
    for node in nodes:
        count += 1
        if node.get("nodes"):
            count += _count_nodes(node["nodes"])
    return count


def main() -> None:
    print("Legal AI — PageIndex tree generation")
    print(f"Processing {len(DOCS)} documents sequentially.\n")

    for i, (doc_name, pdf_filename) in enumerate(DOCS):
        index_document(doc_name, pdf_filename)

        if i < len(DOCS) - 1:
            print(f"\nPausing {INTER_DOC_PAUSE}s between documents to drain rate limit window...")
            time.sleep(INTER_DOC_PAUSE)

    print("\nAll documents processed.")


if __name__ == "__main__":
    main()
