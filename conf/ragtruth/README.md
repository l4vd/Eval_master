# RAGTruth config (`conf/ragtruth/`)

**What it measures:** hallucination in RAG (retrieval-augmented generation), in **two
stages** — the shared `model` generates responses to grounded prompts (stage 1), then a
separate **detector** model flags hallucinated spans (stage 2). Serverless: no TGI/Docker.
Two modes:

- **default** — your generation model produces responses; the detector flags them; report
  the **hallucination rate**.
- **gold-F1** (`gold_f1=true`) — reproduction mode: skip generation, run the detector on
  the corpus's *original* responses + gold labels, report precision/recall/F1.

**Underlying CLI:** `RAGTruth-reproduce/src/run_eval.py` (→ `ragtruth_eval.cli`), via the
launcher's [`build_ragtruth`](../../run_benchmarks.py). See the parent
[config reference](../README.md) for global keys and the shared `model` block.

## Config keys ([`default.yaml`](default.yaml))

| Key | Type | Default | CLI flag | Meaning |
| --- | --- | --- | --- | --- |
| `enabled` | bool | `true` | — | `false` skips the benchmark. |
| `python` | str \| null | `null` | — | Per-benchmark interpreter; `null` = global. |
| `stage` | str | `all` | `--stage` | `generate` (stage 1 only), `detect` (stage 2 only), or `all`. For many checkpoints: `generate` each, then [detect once](#many-checkpoints-generate-then-detect-once). |
| `split` | str \| null | `test` | `--split` | `test` = the 450 held-out sources. `all` / `null` = all 2,965, including the release's train sources (the CLI warns). Also `train`, `dev`. See [Why `split: test`](#why-split-test). |
| `num_samples` | int \| null | `null` | `--num-samples` | Use only the first N items; `null` = global. |
| `task_types` | list[str] \| null | `null` | `--task-types` | `null` = all, or a subset of `[QA, Summary, Data2txt]`. |
| `gold_f1` | bool | `false` | `--gold-f1` | Reproduction mode (see above); skips stage 1. |
| `detector.id` | str | `llama2_13b_lora_0217` | `--detector-model-id` | The stage-2 detector: Hub id, local path, or LoRA adapter. Override with your own trained detector. |
| `detector.base_model_id` | str \| null | `null` | `--detector-base-model-id` | Base for a LoRA detector adapter. |
| `detector.tokenizer_id` | str \| null | `null` | `--detector-tokenizer-id` | Detector tokenizer, if separate. |
| `batch_size` | int | `8` | `--batch-size` | Stage-1 prompts per `generate()` call (left-padded, longest first, written back in dataset order). Recorded; keep it fixed across compared models. |
| `detector.batch_size` | int | `4` | `--detector-batch-size` | Detector prompts per `generate()` call. Lower it on OOM. Recorded. |
| `detector.seed` | int | `42` | `--detector-seed` | Seed re-applied before each run directory's detection (the detector samples). Recorded. |
| `extra_args` | list[str] | `[]` | (appended raw) | Any flag below not surfaced. |

### The two model roles

| Role | Source | Flags |
| --- | --- | --- |
| **Generator** (stage 1) | shared `model.*` | `--model-id`, `--base-model-id`, `--tokenizer-id` |
| **Detector** (stage 2) | `ragtruth.detector.*` | `--detector-model-id`, `--detector-base-model-id`, `--detector-tokenizer-id` |

Both share `--cache-dir`, `--dtype`, and `--device-map` from the shared `model` block.
`--model-id` is required for `generate`/`all`; `--detector-model-id` for `detect`/`all`.

## Full underlying CLI (for `extra_args`)

Stage-1 decoding controls and a few others aren't surfaced as keys — pass via
`ragtruth.extra_args`:

| Flag | Type / default | Meaning |
| --- | --- | --- |
| `--max-new-tokens INT` | int / `512` | Stage-1 generation length cap. |
| `--do-sample` | flag / off | Sample instead of greedy (stage 1). |
| `--temperature FLOAT` | float / none | Requires `--do-sample`. |
| `--top-p FLOAT` | float / none | Requires `--do-sample`. |
| `--top-k INT` | int / none | Requires `--do-sample`. |
| `--system-prompt STR` | str / none | Optional stage-1 system prompt. |
| `--dataset-dir PATH` | path / `RAGTruth-reproduce/dataset` | Where `source_info.jsonl` (+ `response.jsonl`) live. |
| `--log-level LEVEL` | INFO | Logging verbosity. |

```bash
# your own detector, test split, QA only, sampled generation
./run_all.sh ragtruth.detector.id=/path/to/detector \
    ragtruth.split=test ragtruth.task_types='[QA]' \
    ragtruth.extra_args='[--do-sample,--temperature,0.7,--max-new-tokens,256]'

# reproduce the detector's gold-F1 without running your generator
./run_all.sh ragtruth.gold_f1=true ragtruth.split=test
```

## Why `split: test`

The default used to be `null`, meaning all 2,965 release sources. A model trained on RAG-Truth,
as this project's arms are, has seen 1,845 of those as training pairs and 332 as validation pairs,
so the number mixed held-out and seen sources (`SP-DPO-Base/KNOWN_ISSUES.md` §5). `test` evaluates
the 450 release-test sources, none of which the study trained or selected on. `ragtruth.split=all`
restores the old behaviour, and `run_eval.py` logs a warning whenever it generates for the whole
release.

## Many checkpoints: generate, then detect once

`stage=all` loads the 13B detector for every checkpoint. For a sweep, generate per checkpoint and
detect once:

```bash
# per checkpoint (resumable: a finished run has generation_summary.json)
./analysis/run_eval_checkpoints.sh --checkpoints '<group>/seed_*' --out "$EVAL_ROOT/<arm>" \
    --benchmarks ragtruth --resume --launcher-extra ragtruth.stage=generate
# once, over every run dir (skips dirs that already have summary.json)
./analysis/run_ragtruth_detect.sh --root "$EVAL_ROOT" --dtype bfloat16
```

- **Batching.** Both stages batch and sort by length. Batched greedy generation reproduces batch 1.
- **Seeding.** The detector samples, with the authors' T = 0.05, top_p 0.95 and top_k 40, so it is
  seeded per run directory. Its output depends on `detector.batch_size` and `detector.seed`; both
  are recorded.
- **What `summary.json` records.** Also the split, both decoding configs and `elapsed_s`. Time a
  pilot on one checkpoint before a sweep.

## Output

Writes `generations.jsonl` (plus `generation_summary.json`, Stage 1's completion marker),
`detections.jsonl`, and a nested `summary.json` under `<output_dir>/ragtruth/`. The [analysis layer](../../analysis/README.md) reads
`hallucination_rate` (task `overall`, **lower is better** — the one inverted metric) as
primary, plus optional `gold_precision`/`gold_recall`/`gold_f1` (higher better) in
reproduction mode.
