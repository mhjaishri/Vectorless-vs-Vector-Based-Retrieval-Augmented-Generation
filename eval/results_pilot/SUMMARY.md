# Legal AI RAG Evaluation — Summary

- Questions evaluated: **10 / 10**

## Metric averages

| Metric | PageIndex RAG | Vector+Graph RAG | Winner |
|---|---|---|---|
| Context Precision | 0.4417 | 0.1000 | PageIndex |
| Context Recall | 1.0000 | 0.6000 | PageIndex |
| Hit Rate | 1.0000 | 0.6000 | PageIndex |
| Faithfulness | 0.4414 | 0.5684 | Vector+Graph |
| Answer Correctness | 0.7175 | 0.6397 | PageIndex |
| Semantic Similarity | 0.8765 | 0.8617 | PageIndex |

## Files

- `results.json` — full per-question results + averages
- `per_question.jsonl` — append-only line log of each completed Q
- `comparison_bar.png`, `comparison_radar.png`, `summary_table.png`
- `metric_<name>.png` × 6, `per_question_<name>.png` × 6
