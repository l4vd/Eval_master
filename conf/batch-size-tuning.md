# Eval batch-size estimates

Scope: how high to set the per-benchmark batch size when evaluating **SmolLM2‑135M,
Qwen2.5‑0.5B(-Instruct) or Qwen2.5‑14B‑Instruct** through this launcher on one GPU. The
primary target is an A100 80 GB; RTX 8000 48 GB and A100 40 GB are covered for the
cases that fit. It covers FaithEval, HaluEval and the harness, the three benchmarks that
take a batch size.

> **The cluster stack decides this, not the model size.** `setup_envs_HPC.sh` installs every
> benchmark from its `pyproject-HPC.toml`, which pins **transformers 4.41**. In 4.41.2,
> `Qwen2ForCausalLM.forward` and `LlamaForCausalLM.forward` run `lm_head` over **every**
> position and then call `logits.float()` (`modeling_qwen2.py:1162-1163`). `num_logits_to_keep`
> does not exist until 4.45. So on the cluster, the prefill of every generation batch
> materializes a `[B, S_prompt, V]` fp32 logit tensor next to its bf16 source: **0.87 MiB per
> prompt token, per sequence**, for either Qwen.
>
> An earlier version of this file assumed the ≥4.45 last-position trimming and recommended
> 128/256. **On the HPC stack those values OOM for every model here.**

---

## TL;DR — recommended values (HPC stack, transformers 4.41)

Peak is the first batch, because both benchmarks sort prompts `desc`, so the B longest
prompts share it. The pick is the largest power of two that keeps ≥15% of the card free.

### A100 80 GB

| Benchmark | Config knob | SmolLM2‑135M | Qwen2.5‑0.5B | Qwen2.5‑14B | Notes |
|---|---|---|---|---|---|
| **FaithEval** | `faitheval.batch_size` | **32** (44.9 GiB) | **16** (63.7 GiB) | **8** (66.1 GiB) | Worst prompt 4,335 tokens (`inconsistent`). One knob covers all three tasks. |
| **HaluEval** | `halueval.batch_size` | **32** (40.4 GiB) | **16** (57.2 GiB) | **8** (62.3 GiB) | Worst prompt 3,875 tokens (`summarization`). One knob covers all three tasks. |
| **harness** (lm‑eval) | `harness.batch_size` | `8` (pinned) | `8` (pinned) | `8` (pinned) | `EVAL_FIXED` pins 8. Fits every model; `auto` would OOM-probe a 14B several times. |
| **TruthfulQA** / **RAGTruth** | — none — | n/a | n/a | n/a | Unbatched (see §1). |

**The committed default of `8` fits all three models on A100 80 GB.** At 14B it is the
ceiling, not a conservative choice. Because batch composition can flip a greedy argmax tie
(see below), keep `8` for any table that compares models across sizes. Raise it only for
throughput within one model family, and hold the raised value across that family's arms.

### Other cards

| Card | Model | FaithEval | HaluEval | harness |
|---|---|---|---|---|
| RTX 8000 48 GB (fp16) | Qwen2.5‑14B | **2** (39.0 GiB, 18%) | **2** (38.1 GiB, 20%) | 8 (est.) |
| RTX 8000 48 GB (fp16) | Qwen2.5‑0.5B | **8** (33.5 GiB) | **8** (30.3 GiB) | 32 (est.) |
| RTX 8000 48 GB (fp16) | SmolLM2‑135M | **16** | **16** | — |
| A100 40 GB | Qwen2.5‑14B | **1** (34.5 GiB, 13%) | **1** (34.0 GiB, 14%) | 4 (est.) |
| A100 40 GB | Qwen2.5‑0.5B | **8** (33.5 GiB, 15%) | **8** (30.3 GiB) | 32 (est.) |

Harness figures assume a ~512-token TruthfulQA MC request (not measured). Everything else
uses the measured worst-case prompts from §2.

**Keep whatever you pick FIXED across all compared arms.** Batch composition and padding
shift the logits slightly, which can flip a greedy argmax tie. Every FaithEval/HaluEval
config comment stresses this, and the value is recorded in each `summary.json`.

---

## 0. Qwen2.5‑14B‑Instruct: the full command

```bash
./run_all.sh run='[faitheval,halueval,harness]' \
    model.id=/gpfs/project/ladan102/models/huggingface/hub/Qwen/Qwen2.5-14B-Instruct \
    model.dtype=bfloat16 \
    faitheval.batch_size=8 halueval.batch_size=8 harness.batch_size=8
```

- **`faitheval.batch_size` / `halueval.batch_size` = 8** equal today's defaults, but pass
  them anyway: the run then documents itself, and a later change to the YAML default cannot
  silently change a 14B row.
- **`model.dtype=bfloat16`** and **`harness.batch_size=8`** are `EVAL_FIXED`
  (`SP-DPO-Base/EXPERIMENT_PROCEDURE.md` §5.2). Without the pin the harness defaults to
  `auto`, which finds its batch by triggering CUDA OOMs, and each retry on a 14B is a full
  forward pass.
- **Allocate exactly one 80 GB A100.** With `model.device_map=auto` (the default) and a
  multi-GPU allocation, accelerate shards the model across cards, which is slow but works. On
  a 24 GB-or-smaller card it **silently offloads layers to CPU** instead of failing, and the
  run crawls.
- **Host RAM ≥ 64 GB.** The harness loads through lm-eval without `device_map` (to CPU first,
  then `.to(cuda)`), so the bf16 weights (27.5 GiB) pass through host memory.
- **Walltime:** estimate **5–9 h**, not measured, so request 16 h. HaluEval dominates with
  ~34 M prefill tokens over 30,000 judgements (summarization alone ~19 M). FaithEval has
  ~3.4 M prefill tokens, but decoding up to 256 tokens per answer is what costs there.
- **Modified protocols** (`halueval.scoring=constrained`, `faitheval.tasks=[counterfactual_mc]`)
  run the same full-logit forward on 4.41 without a KV cache, so the same batch sizes are safe.

A `peft` adapter `model.id` (a 14B `final_checkpoint`) merges onto its base in memory. The
merged model is the same size, so nothing above changes.

---

## 1. Which benchmarks actually take a batch size

From [`run_benchmarks.py`](../run_benchmarks.py) `build_*` and the reproduce CLIs:

| Benchmark | Flag emitted | Consumed by | Batched? |
|---|---|---|---|
| FaithEval | `--batch-size` ← `faitheval.batch_size` | `FaithEval-reproduce/src/faitheval/evaluator.py` (`for start in range(0, n, batch_size)`) | **Yes**, real generation batching |
| HaluEval | `--batch-size` ← `halueval.batch_size` | `HaluEval-reproduce/evaluation/evaluate.py` `_resolve_batch_size` (HF backend only) | **Yes**, real generation batching |
| harness | `--batch-size` ← `harness.batch_size` (default `${auto_batch_size:}` → `auto`) | `lm_eval` | **Yes**; auto-probed unless pinned |
| TruthfulQA | *(no flag)* | `truthfulqa.evaluate`: MC1/MC2/MC3 score answer-choice log-probs one item at a time | No |
| RAGTruth | *(no flag; only `extra_args`)* | `ragtruth_eval.cli`: Stage‑1 generate + Stage‑2 detect, per row | No |

So the only two numbers you choose by hand are `faitheval.batch_size` and
`halueval.batch_size`.

---

## 2. Inputs

### Model facts (from each `config.json`)

| | SmolLM2‑135M | Qwen2.5‑0.5B(-Instruct) | Qwen2.5‑14B‑Instruct |
|---|---|---|---|
| `num_hidden_layers` (L) | 30 | 24 | 48 |
| `hidden_size` (d) | 576 | 896 | 5120 |
| `num_attention_heads` | 9 | 14 | 40 |
| `num_key_value_heads` (h_kv) | 3 | 2 | 8 |
| head dim | 64 | 64 | 128 |
| KV width per layer (h_kv·head_dim) | 192 | 128 | 1024 |
| `intermediate_size` (i) | 1536 | 4864 | 13824 |
| `vocab_size` (V) | 49 152 | 151 936 | 152 064 |
| `tie_word_embeddings` | true | true | **false** |
| parameters | 134.5 M | 494.0 M | 14 770 M |
| bf16 weights | 0.25 GiB | 0.92 GiB | **27.51 GiB** |
| KV cache, bf16 | 22.5 KiB/tok | 12 KiB/tok | **192 KiB/tok** |

`kv_bytes_per_token = 2 · L · (h_kv · head_dim) · dtype_bytes`. Qwen2.5‑0.5B's KV cache is
smaller than SmolLM2's (fewer layers and KV heads). But on the 4.41 stack neither KV term
binds: the logits do (§3), and those scale with V, where the two Qwens are identical.

### Worst-case prompts (measured)

Qwen2.5 tokenizer, rendered through the chat template exactly as the pipelines send them.
The template adds Qwen's default system prompt when none is given.

| Benchmark / task | rows | mean | **longest** | 8th longest |
|---|---|---|---|---|
| FaithEval `unanswerable`   | 2,492 | 528   | 3,663 | 1,524 |
| FaithEval `inconsistent`   | 1,500 | 1,055 | **4,335** | 2,354 |
| FaithEval `counterfactual` | 1,000 | 468   | 767   | 692   |
| HaluEval `summarization` (longer summary side) | 10,000 | 1,928 | **3,875** | 3,563 |
| HaluEval `qa` / `dialogue` | 10,000 each | — | ≤ ~900 (estimate) | — |

`max_new_tokens`: FaithEval 256. HaluEval defaults to 128 on the chat format, which all three
models use. With `sort_by_length: desc`, the first batch pads every row to the single longest
prompt, so **peak memory = B × the longest prompt**, and an OOM surfaces in the first minute.
The earlier version of this file assumed 1,024 / 3,072-token worst cases and "~1 k rows/task".
SmolLM2's tokenizer (49 k vocab) produces somewhat more tokens for the same text; its column
has the headroom to absorb that.

---

## 3. Memory model

```
prefill   = W + O + B·S·kv + B·S²·2
            + max( B·S·act ,  B·S·c·V )          <- 4.41: logits over every prompt position
decode    = W + O + B·(S+N)·kv·(1 + 1/L)         <- DynamicCache grows by torch.cat
peak      = max(prefill, decode)
```

| symbol | meaning | value |
|---|---|---|
| `W` | weights | 2 B/param (bf16 / fp16) |
| `O` | CUDA context, kernels, allocator slack | 2.5 GiB |
| `S`, `N` | longest prompt in the batch, `max_new_tokens` | §2 |
| `kv` | KV bytes per token | §2 |
| `B·S²·2` | the 4-D `[B,1,S,S]` bf16 causal mask 4.41 builds for SDPA with padding | 0.15 GiB at 14B, B=8 |
| `act` | one block's live set, no grad: `2·(4·i + 8·d)` | 21,504 / 53,248 / 192,512 B/tok |
| `c` | bytes per logit element | **6** (bf16 `lm_head` output + fp32 copy inside `.float()`); **10** for lm-eval log-likelihood, which adds an fp32 `log_softmax` |

On **transformers ≥ 4.45** (the local stack), generation keeps only the last position, so
the logit term becomes `B·c·V` and batch sizes grow 2× (14B, where weights and KV then
bind) to 8× (the small models). That is not the cluster.

`act` replaces the old `k·S·d` (k ≈ 3), which undercounts the SwiGLU intermediate
(`i` = 5.4·d for Qwen2.5‑0.5B). It never binds on 4.41 anyway: the logit term is 5–40× larger.

Per sequence at the worst FaithEval prompt (S = 4,335, N = 256), on 4.41:

| model | logits `S·6·V` | KV `(S+N)·kv` | total per sequence |
|---|---|---|---|
| SmolLM2‑135M | 1.19 GiB | 0.10 GiB | ~1.3 GiB |
| Qwen2.5‑0.5B | 3.68 GiB | 0.05 GiB | ~3.7 GiB |
| Qwen2.5‑14B  | 3.68 GiB | 0.84 GiB | ~4.6 GiB, on top of 27.5 GiB of weights |

---

## 4. Empirical cross-check

The committed configs carry `batch_size: 8 #32 rtx 8000 qwen 2.5 0.5b`, recording that 32
once ran on an RTX 8000 with Qwen2.5‑0.5B.

**That cannot hold for today's runs on the HPC stack.** A desc-sorted FaithEval or HaluEval
first batch of 32 needs ~110 GiB of logits alone. It was most likely a run before
`sort_by_length` was added, a `--num-samples` subset, or the local ≥4.45 stack. The model
predicts **8** for that card and model, which matches the value actually committed.

Nothing here has been measured on a 14B yet. The first 14B run's FaithEval log is the
check: its first batch is the peak (predicted 66.1 GiB on A100 80 GB), so an OOM or the
real `nvidia-smi` number shows up within a minute.

---

## 5. Redoing this for another GPU or model

1. Read `L`, `hidden_size`, `intermediate_size`, `num_key_value_heads`, head dim and
   `vocab_size` from the model's `config.json`.
2. Check the transformers version the benchmark env **actually** runs. On < 4.45 the logit
   term is `B·S·6·V`, and it is almost always the binding one.
3. Take `S` as the **longest** chat-templated prompt of the benchmark's worst task (§2), not a
   typical one: `sort_by_length: desc` puts it in the first batch.
4. Evaluate `peak` from §3 for B = 1, 2, 4, …, and keep the largest B whose peak leaves
   ≥ 15% of the card free.
5. **Hold it fixed across every model you compare.** When one value must serve several
   model sizes, the largest model's limit sets it.
