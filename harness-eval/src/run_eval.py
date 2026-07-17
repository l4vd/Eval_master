#!/usr/bin/env python
"""CLI entry point: `python src/run_eval.py --model-id <id> --tasks truthfulqa`.

See `harness_eval.cli` for the full option list, or run with `--help`.
"""

from harness_eval.cli import main

if __name__ == "__main__":
    main()
