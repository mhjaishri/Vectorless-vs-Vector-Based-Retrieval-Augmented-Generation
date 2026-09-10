"""
Retrieval adapter for legalAgent2 (Vector+GraphRAG — Constitution of India).
Exposes retrieve(query) -> list[str] and answer(query) -> str.
"""
import json
import os
import subprocess

_DIR = os.path.dirname(os.path.abspath(__file__))
_ADAPTER_TS = os.path.join(_DIR, "adapter.ts")
_cache: dict = {}


def _run(query: str) -> dict:
    if query in _cache:
        return _cache[query]
    result = subprocess.run(
        ["npx", "tsx", _ADAPTER_TS, "--query", query],
        capture_output=True,
        text=True,
        cwd=_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"legalAgent2 adapter failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    # Strip any leading noise before the JSON object
    stdout = result.stdout.strip()
    json_start = stdout.rfind("{")
    if json_start == -1:
        raise RuntimeError(f"No JSON found in legalAgent2 output:\n{stdout}")
    data = json.loads(stdout[json_start:])
    _cache[query] = data
    return data


def retrieve(query: str) -> list:
    """Return list of retrieved text chunks for the query."""
    return _run(query).get("chunks", [])


def answer(query: str) -> str:
    """Return the generated answer for the query."""
    return _run(query).get("answer", "")
