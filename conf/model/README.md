# Shared model config (`conf/model/`)

**One** model target feeds **all five** benchmarks. Selected with `model=<name>` (default
`qwen_tiny`). Both files carry `# @package _global_`, so their `model:` block is merged at
`cfg.model.*` (not `cfg.model.model.*`). The launcher translates each field into whatever
flag the individual benchmark uses — the spellings differ (most use `--model-id`,
TruthfulQA uses `--model_path`, etc.).

| File | Purpose |
| --- | --- |
| [`qwen_tiny.yaml`](qwen_tiny.yaml) | The selected default: `Qwen/Qwen2.5-0.5B-Instruct`, a fast smoke-test model. dtype/device auto-detect hardware. |
| [`_template.yaml`](_template.yaml) | Copy to `conf/model/<name>.yaml`, point `id` at your checkpoint, select with `model=<name>`. |

## Fields

| Key | Type | Default (`qwen_tiny`) | Maps to (per benchmark) | Meaning |
| --- | --- | --- | --- | --- |
| `model.id` | str | `Qwen/Qwen2.5-0.5B-Instruct` | `--model-id` / `--model_path` / `--model-path` | The evaluated model: a Hub id, a local full-model dir, or a PEFT/LoRA adapter checkpoint (e.g. a `final_checkpoint` — auto-detected and merged onto its base). |
| `model.base_model_id` | str \| null | `null` | `--base-model-id` / `--base_model_id` | Base to merge a LoRA adapter onto, when it isn't recorded in the adapter config. |
| `model.tokenizer_id` | str \| null | `null` | `--tokenizer-id` / `--tokenizer_id` | Tokenizer, if not saved alongside the weights. |
| `model.cache_dir` | str \| null | `null` | `--cache-dir` / `--cache_dir` | Hugging Face cache directory. |
| `model.dtype` | str | `${auto_dtype:}` | `--dtype` | `bfloat16` \| `float16` \| `float32`. |
| `model.device_map` | str | `auto` | `--device-map` | HF `device_map` — used by **FaithEval / HaluEval / RAGTruth**. |
| `model.device_index` | int | `${auto_device_index:}` | `--device` | CUDA index — used by **TruthfulQA / harness**. `-1` = CPU, `0` = cuda:0. |

Note the split: the three generation benchmarks take a `device_map`; the two lm_eval-style
ones take a single `device_index`. Both are set for you by the resolvers below; override
whichever the benchmarks you're running actually use.

## The `${auto_*:}` resolvers

`model.dtype` and `model.device_index` (and `harness.batch_size`) default to OmegaConf
resolvers registered in [`run_benchmarks.py`](../../run_benchmarks.py). The launcher's own
venv carries **no torch**, so hardware detection shells out to `nvidia-smi` at startup:

| Resolver | GPU present | No GPU |
| --- | --- | --- |
| `${auto_device_index:}` | `0` (cuda:0) | `-1` (CPU) |
| `${auto_dtype:}` | `bfloat16` if compute capability ≥ 8.0 (Ampere+), else `float16` | `float32` |
| `${auto_batch_size:}` | `"auto"` (lm_eval OOM-probing search) | `"8"` |

So a plain `./run_all.sh` picks safe settings on a laptop and a GPU node alike. Any
explicit value bypasses the resolver.

## Recipes

```bash
# Point at your own checkpoint / LoRA on the CLI (no file edit):
./run_all.sh model.id=/path/to/final_checkpoint
./run_all.sh model.id=/path/to/lora model.base_model_id=meta-llama/Meta-Llama-3.1-8B-Instruct

# Force CPU / a dtype:
./run_all.sh model.device_index=-1 model.device_map=cpu model.dtype=float32

# Use a saved model file instead:
cp conf/model/_template.yaml conf/model/mymodel.yaml   # edit id
./run_all.sh model=mymodel
```
