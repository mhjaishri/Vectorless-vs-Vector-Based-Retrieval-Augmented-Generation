"""
Evaluation runner for Legal AI RAG systems.

Usage:
    cd eval/
    python runner.py --qna qna_100.json --output results_110/

Loads both RAG system adapters, runs all QnA questions through them,
computes 6 metrics per question per system, writes results.json after
each completed question (atomic), appends to per_question.jsonl, and
generates comparison charts plus SUMMARY.md at the end.

Resumable: re-running with the same --output reuses already-completed
question entries from results.json and skips them.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_adapter(name: str, path: str):
    """Load a Python module by absolute file path under a unique name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pageindex_adapter = _load_adapter(
    "pageindex_adapter", os.path.join(REPO_ROOT, "legalAgent", "adapter.py")
)
vector_graph_adapter = _load_adapter(
    "vector_graph_adapter", os.path.join(REPO_ROOT, "legalAgent2", "adapter.py")
)

import metrics


SYSTEMS = {
    "pageindex_rag": pageindex_adapter,
    "vector_graph_rag": vector_graph_adapter,
}

METRIC_NAMES = [
    "context_precision", "context_recall", "hit_rate",
    "faithfulness", "answer_correctness", "semantic_similarity",
]


def _atomic_save(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _empty_results() -> dict:
    return {
        sn: {**{m: 0.0 for m in METRIC_NAMES}, "per_question": []}
        for sn in SYSTEMS
    }


def _compute_averages(results: dict) -> None:
    for sn in SYSTEMS:
        per_q = results[sn]["per_question"]
        if not per_q:
            continue
        n = len(per_q)
        for m in METRIC_NAMES:
            results[sn][m] = round(sum(q.get(m, 0.0) for q in per_q) / n, 4)


def _write_summary_md(results: dict, output_dir: str, total_questions: int) -> str:
    """Write a SUMMARY.md with averages, per-metric winner, and partial-run note."""
    completed = min(len(results[sn]["per_question"]) for sn in SYSTEMS)
    label = {"pageindex_rag": "PageIndex RAG", "vector_graph_rag": "Vector+Graph RAG"}

    lines = ["# Legal AI RAG Evaluation — Summary", ""]
    lines.append(f"- Questions evaluated: **{completed} / {total_questions}**")
    if completed < total_questions:
        lines.append(f"- ⚠️ Partial run — {total_questions - completed} question(s) not yet processed.")
    lines.append("")
    lines.append("## Metric averages")
    lines.append("")
    lines.append("| Metric | PageIndex RAG | Vector+Graph RAG | Winner |")
    lines.append("|---|---|---|---|")
    for m in METRIC_NAMES:
        pi = results["pageindex_rag"].get(m, 0.0)
        vg = results["vector_graph_rag"].get(m, 0.0)
        if pi > vg:
            winner = "PageIndex"
        elif vg > pi:
            winner = "Vector+Graph"
        else:
            winner = "tie"
        lines.append(f"| {m.replace('_',' ').title()} | {pi:.4f} | {vg:.4f} | {winner} |")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `results.json` — full per-question results + averages")
    lines.append("- `per_question.jsonl` — append-only line log of each completed Q")
    lines.append("- `comparison_bar.png`, `comparison_radar.png`, `summary_table.png`")
    lines.append("- `metric_<name>.png` × 6, `per_question_<name>.png` × 6")

    path = os.path.join(output_dir, "SUMMARY.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run_evaluation(qna_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "llm_cache.json")
    results_path = os.path.join(output_dir, "results.json")
    jsonl_path = os.path.join(output_dir, "per_question.jsonl")

    metrics.init_judge(cache_path)

    with open(qna_path) as f:
        qna = json.load(f)

    # ── Resume support ────────────────────────────────────────────
    done_ids: set[str] = set()
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                prior = json.load(f)
            sys_done = {sn: {pq["id"] for pq in prior.get(sn, {}).get("per_question", [])} for sn in SYSTEMS}
            done_ids = set.intersection(*sys_done.values()) if all(sys_done.values()) else set()
            results = prior
            # Ensure all systems / metric keys exist
            for sn in SYSTEMS:
                results.setdefault(sn, {**{m: 0.0 for m in METRIC_NAMES}, "per_question": []})
                for m in METRIC_NAMES:
                    results[sn].setdefault(m, 0.0)
                results[sn].setdefault("per_question", [])
            print(f"[resume] Found existing results.json; {len(done_ids)} questions already completed — skipping them\n")
        except Exception as e:
            print(f"[resume] Could not load existing results.json ({e}); starting fresh\n")
            results = _empty_results()
    else:
        results = _empty_results()

    total_questions = len(qna)
    print(f"Running evaluation on {total_questions} questions across {len(SYSTEMS)} systems...")
    print("Note: Each question calls both RAG systems (may take several minutes each).\n")

    for q_idx, item in enumerate(qna):
        q_id = item["id"]
        question = item["question"]
        gt_answer = item["ground_truth_answer"]
        gt_sources = item["ground_truth_sources"]

        if q_id in done_ids:
            print(f"[{q_idx + 1}/{total_questions}] {q_id}: SKIPPED (resume)")
            continue

        print(f"[{q_idx + 1}/{total_questions}] {q_id}: {question[:70]}...")

        for sys_name, adapter in SYSTEMS.items():
            print(f"  → {sys_name} ...", end=" ", flush=True)
            t0 = time.time()

            try:
                chunks = adapter.retrieve(question)
                gen_answer = adapter.answer(question)
            except Exception as e:
                print(f"ERROR: {e}")
                chunks = []
                gen_answer = ""

            elapsed = time.time() - t0
            print(f"({elapsed:.1f}s, {len(chunks)} chunks)")

            cp = metrics.context_precision(question, chunks)
            cr = metrics.context_recall(gt_sources, chunks)
            hr = metrics.hit_rate(gt_sources, chunks)
            faith = metrics.faithfulness(gen_answer, chunks)
            corr = metrics.answer_correctness(gen_answer, gt_answer)
            sem = metrics.semantic_similarity(gen_answer, gt_answer)

            print(f"     precision={cp:.2f}  recall={cr:.2f}  hit_rate={hr:.2f}"
                  f"  faithfulness={faith:.2f}  correctness={corr:.2f}  sem_sim={sem:.2f}")

            results[sys_name]["per_question"].append({
                "id": q_id,
                "question": question,
                "generated_answer": gen_answer,
                "chunks_count": len(chunks),
                "context_precision": cp,
                "context_recall": cr,
                "hit_rate": hr,
                "faithfulness": faith,
                "answer_correctness": corr,
                "semantic_similarity": sem,
            })

        # Compute running averages so a partial results.json is meaningful
        _compute_averages(results)
        _atomic_save(results_path, results)

        # Append durable per-Q line log
        try:
            jsonl_record = {
                "q_id": q_id,
                "timestamp": time.time(),
                **{sn: results[sn]["per_question"][-1] for sn in SYSTEMS},
            }
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(jsonl_record) + "\n")
        except Exception as e:
            print(f"  [warn] could not append to per_question.jsonl: {e}")

        print()

    # Final averages (already current, but recompute to be safe)
    _compute_averages(results)
    _atomic_save(results_path, results)
    print(f"Results saved to {results_path}\n")

    # Print summary table
    print("=" * 72)
    print(f"{'Metric':<28} {'PageIndex':>14} {'Vec+Graph':>14}")
    print("=" * 72)
    for m in METRIC_NAMES:
        pi = results["pageindex_rag"][m]
        vg = results["vector_graph_rag"][m]
        print(f"  {m:<26} {pi:>14.4f} {vg:>14.4f}")
    print("=" * 72)
    print()

    # Write SUMMARY.md
    summary_path = _write_summary_md(results, output_dir, total_questions)
    print(f"Wrote {summary_path}")

    # Generate charts (robust to partial: only generates if at least one Q completed)
    if any(results[sn]["per_question"] for sn in SYSTEMS):
        print("Generating charts...")
        try:
            from visualize import generate_all_charts
            generate_all_charts(results_path, output_dir)
        except Exception as e:
            print(f"  [warn] chart generation failed: {e}")
            print(f"  Run manually: python visualize.py --results {results_path} --output {output_dir}")
    else:
        print("[warn] no completed questions yet — skipping chart generation")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Legal AI RAG systems")
    parser.add_argument("--qna", default="qna_100.json", help="Path to QnA JSON file")
    parser.add_argument("--output", default="results", help="Output directory for results")
    args = parser.parse_args()

    run_evaluation(args.qna, args.output)
