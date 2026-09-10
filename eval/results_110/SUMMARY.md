# Legal AI RAG Evaluation — Summary

- Questions evaluated: **110 / 110**

## Metric averages

| Metric | PageIndex RAG | Vector+Graph RAG | Winner |
|---|---|---|---|
| Context Precision | 0.1637 | 0.0700 | PageIndex |
| Context Recall | 0.8430 | 0.6083 | PageIndex |
| Hit Rate | 0.8636 | 0.6818 | PageIndex |
| Faithfulness | 0.4517 | 0.4616 | Vector+Graph |
| Answer Correctness | 0.5736 | 0.6567 | Vector+Graph |
| Semantic Similarity | 0.8365 | 0.8647 | Vector+Graph |

## Files

- `results.json` — full per-question results + averages
- `per_question.jsonl` — append-only line log of each completed Q
- `comparison_bar.png`, `comparison_radar.png`, `summary_table.png`
- `metric_<name>.png` × 6, `per_question_<name>.png` × 6
