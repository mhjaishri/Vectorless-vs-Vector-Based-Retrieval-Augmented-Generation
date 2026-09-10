"""
Generate figures for the paper from eval/results_110/results.json.

Outputs (under paper/figures/):
  decoupling_scatter.png   — Fig. 4: precision vs correctness, both systems, with regression
  retrieval_metrics_bar.png — Fig. 2: precision/recall/hit_rate bar chart
  answer_metrics_bar.png    — Fig. 3: faithfulness/correctness/sem_sim bar chart
  faithfulness_correlation.png — Fig. 5: per-system faithfulness vs precision (shows Vec+Graph grounded, PI decoupled)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "eval", "results_110", "results.json")
FIGS = os.path.join(REPO, "paper", "figures")
os.makedirs(FIGS, exist_ok=True)

with open(RESULTS) as f:
    R = json.load(f)

PI = R["pageindex_rag"]["per_question"]
VG = R["vector_graph_rag"]["per_question"]

PI_LABEL = "PageIndex RAG"
VG_LABEL = "Vector+Graph RAG"
PI_COLOR = "#2196F3"
VG_COLOR = "#FF9800"


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 2 — Retrieval-quality metrics (PageIndex wins)
# ─────────────────────────────────────────────────────────────────────────────
metrics_retr = ["context_precision", "context_recall", "hit_rate"]
labels_retr = ["Context\nPrecision", "Context\nRecall", "Hit Rate"]
pi_vals = [R["pageindex_rag"][m] for m in metrics_retr]
vg_vals = [R["vector_graph_rag"][m] for m in metrics_retr]

fig, ax = plt.subplots(figsize=(7.0, 4.0))
x = np.arange(len(metrics_retr))
w = 0.35
b1 = ax.bar(x - w / 2, pi_vals, w, label=PI_LABEL, color=PI_COLOR, alpha=0.9)
b2 = ax.bar(x + w / 2, vg_vals, w, label=VG_LABEL, color=VG_COLOR, alpha=0.9)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Score (0.0–1.0)")
ax.set_title("Retrieval-Quality Metrics (110 questions)")
ax.set_xticks(x)
ax.set_xticklabels(labels_retr)
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "retrieval_metrics_bar.png"), dpi=200)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Fig. 3 — Answer-quality metrics (Vec+Graph wins)
# ─────────────────────────────────────────────────────────────────────────────
metrics_ans = ["faithfulness", "answer_correctness", "semantic_similarity"]
labels_ans = ["Faithfulness", "Answer\nCorrectness", "Semantic\nSimilarity"]
pi_vals = [R["pageindex_rag"][m] for m in metrics_ans]
vg_vals = [R["vector_graph_rag"][m] for m in metrics_ans]

fig, ax = plt.subplots(figsize=(7.0, 4.0))
b1 = ax.bar(x - w / 2, pi_vals, w, label=PI_LABEL, color=PI_COLOR, alpha=0.9)
b2 = ax.bar(x + w / 2, vg_vals, w, label=VG_LABEL, color=VG_COLOR, alpha=0.9)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Score (0.0–1.0)")
ax.set_title("Answer-Quality Metrics (110 questions)")
ax.set_xticks(x)
ax.set_xticklabels(labels_ans)
ax.set_ylim(0, 1.0)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "answer_metrics_bar.png"), dpi=200)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Fig. 4 — Decoupling scatter: precision vs correctness, both systems
# ─────────────────────────────────────────────────────────────────────────────
def jitter(v, eps=0.012, seed=0):
    rng = np.random.default_rng(seed)
    return v + rng.uniform(-eps, eps, len(v))


fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
for ax, qs, name, color, seed in [
    (axes[0], PI, PI_LABEL, PI_COLOR, 1),
    (axes[1], VG, VG_LABEL, VG_COLOR, 2),
]:
    px = np.array([q["context_precision"] for q in qs])
    py = np.array([q["answer_correctness"] for q in qs])
    ax.scatter(jitter(px, seed=seed), jitter(py, seed=seed + 100), c=color, alpha=0.55, s=24, edgecolor="white", linewidth=0.4)
    # Linear fit
    if px.std() > 0:
        m, b = np.polyfit(px, py, 1)
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, m * xs + b, color=color, linestyle="--", linewidth=2, label=f"slope={m:+.2f}")
        r = float(np.corrcoef(px, py)[0, 1])
        ax.text(0.04, 0.94, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.35"))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Context Precision")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel("Answer Correctness")
fig.suptitle("Retrieval Precision vs Answer Correctness (per question)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "decoupling_scatter.png"), dpi=200)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Fig. 5 — Faithfulness correlation: PI faithfulness decoupled from precision,
#          VG faithfulness strongly tied to precision (the central asymmetry)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
for ax, qs, name, color, seed in [
    (axes[0], PI, PI_LABEL, PI_COLOR, 11),
    (axes[1], VG, VG_LABEL, VG_COLOR, 22),
]:
    px = np.array([q["context_precision"] for q in qs])
    py = np.array([q["faithfulness"] for q in qs])
    ax.scatter(jitter(px, seed=seed), jitter(py, seed=seed + 100), c=color, alpha=0.55, s=24, edgecolor="white", linewidth=0.4)
    if px.std() > 0:
        m, b = np.polyfit(px, py, 1)
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, m * xs + b, color=color, linestyle="--", linewidth=2, label=f"slope={m:+.2f}")
        r = float(np.corrcoef(px, py)[0, 1])
        ax.text(0.04, 0.94, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.35"))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Context Precision")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel("Faithfulness")
fig.suptitle("Faithfulness vs Retrieval Precision — Asymmetric Grounding", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "faithfulness_correlation.png"), dpi=200)
plt.close(fig)

print("Figures written to", FIGS)
for fn in sorted(os.listdir(FIGS)):
    print(" -", fn)
