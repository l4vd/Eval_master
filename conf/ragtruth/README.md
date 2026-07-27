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
| `stage` | str | `all` | `--stage` | `generate` (stage 1 only), `detect` (stage 2 only), or `all`. |
| `split` | str \| null | `null` | `--split` | `null`/all, or `train` \| `dev` \| `test`. |
| `num_samples` | int \| null | `null` | `--num-samples` | Use only the first N items; `null` = global. |
| `task_types` | list[str] \| null | `null` | `--task-types` | `null` = all, or a subset of `[QA, Summary, Data2txt]`. |
| `gold_f1` | bool | `false` | `--gold-f1` | Reproduction mode (see above); skips stage 1. |
| `detector.id` | str | `llama2_13b_lora_0217` | `--detector-model-id` | The stage-2 detector: Hub id, local path, or LoRA adapter. Override with your own trained detector. |
| `detector.base_model_id` | str \| null | `null` | `--detector-base-model-id` | Base for a LoRA detector adapter. |
| `detector.tokenizer_id` | str \| null | `null` | `--detector-tokenizer-id` | Detector tokenizer, if separate. |
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

## Output

Writes `generations.jsonl`, `detections.jsonl`, and a nested `summary.json` under
`<output_dir>/ragtruth/`. The [analysis layer](../../analysis/README.md) reads
`hallucination_rate` (task `overall`, **lower is better** — the one inverted metric) as
primary, plus optional `gold_precision`/`gold_recall`/`gold_f1` (higher better) in
reproduction mode.
