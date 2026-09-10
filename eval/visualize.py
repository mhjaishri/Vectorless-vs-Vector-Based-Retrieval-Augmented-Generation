"""
Chart generation for Legal AI RAG evaluation results.
Produces three chart types saved to the results/ directory.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SYSTEM_LABELS = {
    "pageindex_rag": "PageIndex RAG",
    "vector_graph_rag": "Vector+Graph RAG",
}
COLORS = {
    "pageindex_rag": "#2196F3",   # blue
    "vector_graph_rag": "#FF9800", # orange
}
METRICS = [
    "context_precision", "context_recall", "hit_rate",
    "faithfulness", "answer_correctness", "semantic_similarity",
]
METRIC_LABELS = [
    "Context\nPrecision", "Context\nRecall", "Hit\nRate",
    "Faithfulness", "Answer\nCorrectness", "Semantic\nSimilarity",
]


def plot_comparison_bar(results: dict, output_dir: str) -> str:
    """Grouped bar chart comparing both systems across all 4 metrics."""
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(METRICS))
    width = 0.35
    systems = list(results.keys())

    for i, system in enumerate(systems):
        scores = [results[system].get(m, 0.0) for m in METRICS]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=SYSTEM_LABELS.get(system, system),
                      color=COLORS.get(system, "#999"), alpha=0.85)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.set_title("Legal AI RAG System Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "comparison_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_radar(results: dict, output_dir: str) -> str:
    """Radar/spider chart showing shape-of-performance for both systems."""
    categories = METRIC_LABELS
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for system in results:
        scores = [results[system].get(m, 0.0) for m in METRICS]
        scores += scores[:1]
        ax.plot(angles, scores, linewidth=2, label=SYSTEM_LABELS.get(system, system),
                color=COLORS.get(system, "#999"))
        ax.fill(angles, scores, alpha=0.1, color=COLORS.get(system, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7)
    ax.set_title("RAG System Performance Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()

    path = os.path.join(output_dir, "comparison_radar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_question(results: dict, output_dir: str) -> list:
    """Per-question line charts — one chart per metric."""
    paths = []
    systems = list(results.keys())

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for system in systems:
            per_q = results[system].get("per_question", [])
            if not per_q:
                continue
            q_ids = [q["id"] for q in per_q]
            scores = [q.get(metric, 0.0) for q in per_q]
            ax.plot(q_ids, scores, marker="o", linewidth=2,
                    label=SYSTEM_LABELS.get(system, system),
                    color=COLORS.get(system, "#999"))

        ax.set_xlabel("Question ID")
        ax.set_ylabel("Score (0.0 – 1.0)")
        ax.set_title(f"Per-Question: {metric.replace('_', ' ').title()}")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        path = os.path.join(output_dir, f"per_question_{metric}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_summary_table(results: dict, output_dir: str) -> str:
    """Save overall metrics as a formatted table PNG."""
    systems = list(results.keys())
    col_labels = [SYSTEM_LABELS.get(s, s) for s in systems]
    row_labels = [m.replace("_", " ").title() for m in METRICS]

    cell_data = []
    cell_colors = []
    for metric in METRICS:
        row_vals = []
        row_cols = []
        scores = [results[s].get(metric, 0.0) for s in systems]
        best = max(scores)
        for s in systems:
            v = results[s].get(metric, 0.0)
            row_vals.append(f"{v:.4f}")
            row_cols.append("#d4edda" if v == best else "#ffffff")
        cell_data.append(row_vals)
        cell_colors.append(row_cols)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)

    # Style header row and row labels
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Overall Metrics Summary  (green = best per metric)", pad=12, fontsize=12)
    fig.tight_layout()

    path = os.path.join(output_dir, "summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_metric_bar(results: dict, output_dir: str) -> list:
    """One focused bar chart per metric, comparing both systems side by side."""
    paths = []
    systems = list(results.keys())
    x = np.arange(len(systems))
    labels = [SYSTEM_LABELS.get(s, s) for s in systems]
    colors = [COLORS.get(s, "#999") for s in systems]

    for metric in METRICS:
        scores = [results[s].get(metric, 0.0) for s in systems]

        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(x, scores, color=colors, alpha=0.85, width=0.5)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Score (0.0 – 1.0)", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(systems))]
        ax.legend(handles=patches, fontsize=9)

        fig.tight_layout()
        path = os.path.join(output_dir, f"metric_{metric}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def generate_all_charts(results_json_path: str, output_dir: str) -> None:
    """Load results.json and generate all charts."""
    with open(results_json_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    bar_path = plot_comparison_bar(results, output_dir)
    radar_path = plot_radar(results, output_dir)
    table_path = plot_summary_table(results, output_dir)
    per_metric_paths = plot_per_metric_bar(results, output_dir)
    per_q_paths = plot_per_question(results, output_dir)

    print(f"  Bar chart:     {bar_path}")
    print(f"  Radar chart:   {radar_path}")
    print(f"  Summary table: {table_path}")
    for p in per_metric_paths:
        print(f"  Metric chart:  {p}")
    for p in per_q_paths:
        print(f"  Per-question:  {p}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/results.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    generate_all_charts(args.results, args.output)
