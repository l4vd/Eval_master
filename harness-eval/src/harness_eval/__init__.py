"""TruthfulQA evaluation via EleutherAI/lm-evaluation-harness.

A thin wrapper that plugs `lm_eval` into the Eval_master launcher: it translates
the shared `model.id` interface into lm_eval `model_args`, validates task names
against the installed lm_eval, and normalises the results into the repo's
`summary.json` / `samples.jsonl` convention.
"""

__version__ = "0.1.0"
