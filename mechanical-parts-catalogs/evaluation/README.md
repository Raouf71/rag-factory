## Evaluation — RAGAS

All pipelines are benchmarked using [RAGAS](https://docs.ragas.io/):

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevance** | Does the answer address the question? |
| **Context Precision** | Are retrieved chunks actually useful? |
| **Context Recall** | Did retrieval capture all necessary information? |

Results are logged and compared across patterns and frameworks.