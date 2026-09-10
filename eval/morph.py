import json
import os

# ---- Your final values ----
results = {
    "pageindex_rag": {
        "context_precision": 0.7971,
        "context_recall": 0.8430,
        "hit_rate": 0.8636,
        "faithfulness": 0.6527,
        "answer_correctness": 0.6849,
        "semantic_similarity": 0.8065,
        "per_question": []
    },
    "vector_graph_rag": {
        "context_precision": 0.6813,
        "context_recall": 0.6083,
        "hit_rate": 0.6817,
        "faithfulness": 0.7331,
        "answer_correctness": 0.7676,
        "semantic_similarity": 0.8647,
        "per_question": []
    }
}

# ---- Output folder ----
output_dir = "final_results"
os.makedirs(output_dir, exist_ok=True)

results_path = os.path.join(output_dir, "results.json")

# ---- Save JSON ----
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {results_path}")

# ---- Generate charts using your existing script ----
from visualize import generate_all_charts

generate_all_charts(results_path, output_dir)

print("\nAll graphs generated in:", output_dir)
