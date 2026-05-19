# Benchmark configs

`scripts/benchmark_ci.py` reads benchmark experiments from JSON config files. The smoke config is intentionally small and runs one public-safe dataset through the preview pipelines.

Add a benchmark by adding a dataset and experiment to a config:

```json
{
  "datasets": {
    "my_dataset": {
      "path": "docs/data/example.csv",
      "sha256": "<sha256>",
      "text_column": "text",
      "data_summary": "Short dataset description"
    }
  },
  "experiments": [
    {
      "name": "redact_my_dataset",
      "pipeline": "redact",
      "dataset": "my_dataset",
      "num_records": 10
    }
  ]
}
```

Datasets can use either `path` or `url`. URL datasets must include `sha256`; the runner downloads them into `.benchmark-data-cache` and verifies the hash before running.
