"""Architecture diagram (Fig. 1) — PageIndex vs Vector+Graph."""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGS, exist_ok=True)

PI_COLOR = "#2196F3"
VG_COLOR = "#FF9800"
GRAY = "#555"

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))


def box(ax, x, y, w, h, text, fc, ec="black", fontsize=9, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.0,
                       facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.2, color=GRAY)
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + 0.04, (y1 + y2) / 2, text, fontsize=8,
                color=GRAY, ha="left", va="center")


# ─── PageIndex (left) ────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("PageIndex RAG (tree-based, multi-hop agent)", color=PI_COLOR,
             fontweight="bold", fontsize=11)

box(ax, 1.0, 8.5, 8.0, 0.9, "User question", "#f0f0f0", weight="bold")
box(ax, 1.0, 7.0, 8.0, 0.9, "HyDE: hypothesise answer terms (LLM)", "#fffae6")
box(ax, 1.0, 5.5, 8.0, 0.9,
    "Pre-built tree index — Constitution / BNS / BNSS\n(hierarchical document structure)", "#e3f2fd")

# Agent loop
box(ax, 1.0, 3.5, 3.6, 1.6,
    "Agent loop\n(NAV LLM)\nMAX_STEPS=15", "#bbdefb", weight="bold")
box(ax, 5.4, 3.5, 3.6, 1.6,
    "Tools:\n• search_in_text\n• search_reference\n• get_node_content\n• done()",
    "#ffffff")
arrow(ax, 4.6, 4.3, 5.4, 4.3)
arrow(ax, 5.4, 3.9, 4.6, 3.9)

box(ax, 1.0, 1.5, 8.0, 1.0,
    "Selected nodes (avg 3.5 chunks/Q)\n→ Final answer (ANSWER LLM)", "#bbdefb")

arrow(ax, 5.0, 8.5, 5.0, 7.9)
arrow(ax, 5.0, 7.0, 5.0, 6.4)
arrow(ax, 5.0, 5.5, 5.0, 5.1)
arrow(ax, 5.0, 3.5, 5.0, 2.5)

# ─── Vector+Graph (right) ───────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Vector+Graph RAG (Qdrant + Neo4j fusion)", color=VG_COLOR,
             fontweight="bold", fontsize=11)

box(ax, 1.0, 8.5, 8.0, 0.9, "User question", "#f0f0f0", weight="bold")
box(ax, 1.0, 7.0, 8.0, 0.9,
    "Embed (Ollama nomic-embed-text 768-dim)", "#fffae6")

box(ax, 0.7, 5.0, 3.9, 1.6,
    "Qdrant\ncosine top-k=10\nthreshold ≥ 0.4", "#ffe0b2", weight="bold")
box(ax, 5.2, 5.0, 4.1, 1.6,
    "Neo4j graph\n(traverse REFERS_TO,\nHAS_CLAUSE up to depth 2)", "#ffe0b2", weight="bold")

box(ax, 1.0, 3.0, 8.0, 1.0,
    "Fused context (10 chunks/Q)", "#ffcc80")

box(ax, 1.0, 1.0, 8.0, 1.0, "Single-call answer (LLM)", "#ffcc80")

arrow(ax, 5.0, 8.5, 5.0, 7.9)
arrow(ax, 5.0, 7.0, 5.0, 6.6)
arrow(ax, 2.5, 5.0, 4.0, 4.0)
arrow(ax, 7.5, 5.0, 6.0, 4.0)
arrow(ax, 5.0, 3.0, 5.0, 2.0)

fig.suptitle("Figure 1: Two RAG systems, same domain, different retrieval philosophies",
             fontsize=10, y=0.02)
fig.tight_layout(rect=[0, 0.04, 1, 1])

out = os.path.join(FIGS, "architecture_diagram.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)
