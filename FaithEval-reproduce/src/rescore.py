#!/usr/bin/env python
"""CLI entry point: `python src/rescore.py <task>_predictions.jsonl --rule <rule> [options]`.

Offline re-scoring of a finished run, no model. See `faitheval.rescore`, or run with `--help`.
"""

from faitheval.rescore import main

if __name__ == "__main__":
    raise SystemExit(main())
