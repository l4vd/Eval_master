# Eval batch-size estimates — A100 80 GB

Scope: how high to set the per-benchmark generation batch size when evaluating a
**small** policy model (SmolLM2‑135M or Qwen2.5‑0.5B) through this launcher on a
single **A100 80 GB**. Covers the four `*-reproduce` benchmarks plus the harness.

The short story: for models this small the batch size is **throughput-bound, not
memory-bound** — the analytic OOM ceiling is in the hundreds-to-thousands, but the
datasets are only ~1 k rows/task (FaithEval) or you cap them anyway for a smoke,
so there is nothing to gain past a few hundred.

---

## TL;DR — recommended values

| Benchmark | Config knob | SmolLM2‑135M | Qwen2.5‑0.5B | Notes |
|---|---|---|---|---|
| **FaithEval** | `faitheval.batch_size` | **128** (≤ 256) | **128** (≤ 256) | ~1 k rows/task, 256 new tokens. >256 → <4 batches, padding waste grows. |
| **HaluEval** | `halueval.batch_size` | **256** (drop to 128 if `summarization` OOMs) | **256** (≤ 512) | 10 k rows/task, 16–128 new tokens. `summarization` (CNN/DM articles) is the limiter. |
| **TruthfulQA** (`metrics: [mc]`) | — none — | n/a | n/a | Sequential log-prob scoring; no `--batch-size` flag. |
| **RAGTruth** | — none — | n/a | n/a | Row-by-row. Memory user is the Stage‑2 Llama2‑13B LoRA detector (~26 GB fp16), not the policy model. |
| **harness** (lm‑eval) | `harness.batch_size` | `auto` | `auto` | lm‑eval OOM-probes for the largest fitting batch itself. Leave it. |

For the `--num-samples 200` **smoke** runs specifically, `64` for both FaithEval
and HaluEval is plenty (200 rows / 64 ≈ 3–4 batches) and matches the known-good
RTX 8000 value with ~2× headroom.

**Keep whatever you pick FIXED across all compared arms** (e.g. all 8 staged-
curriculum smoke checkpoints). Batch composition + right/left padding shift the
logits slightly, which can flip a greedy argmax tie — every FaithEval/HaluEval
config comment stresses this. The value is recorded in each `summary.json`.

---

## 1. Which benchmarks actually take a batch size

From [`run_benchmarks.py`](../run_benchmarks.py) `build_*` and the reproduce CLIs:

| Benchmark | Flag emitted | Consumed by | Batched? |
|---|---|---|---|
| FaithEval | `--batch-size` ← `faitheval.batch_size` | `FaithEval-reproduce/src/faitheval/evaluator.py` (`for start in range(0, n, batch_size)`) | **Yes**, real generation batching |
| HaluEval | `--batch-size` ← `halueval.batch_size` | `HaluEval-reproduce/evaluation/evaluate.py` `_resolve_batch_size` (HF backend only) | **Yes**, real generation batching |
| harness | `--batch-size` ← `harness.batch_size` (default `${auto_batch_size:}` → `auto`) | `lm_eval` | **Yes**, auto-probed |
| TruthfulQA | *(no flag)* | `truthfulqa.evaluate` — MC1/MC2/MC3 score answer-choice log-probs one item at a time | No |
| RAGTruth | *(no flag; only `extra_args`)* | `ragtruth_eval.cli` — Stage‑1 generate + Stage‑2 detect, per row | No |

So the only two numbers you choose by hand are `faitheval.batch_size` and
`halueval.batch_size`.

---

## 2. Model facts (from each `config.json`)

| | SmolLM2‑135M | Qwen2.5‑0.5B(-Instruct) |
|---|---|---|
| `num_hidden_layers` (L) | 30 | 24 |
| `hidden_size` (d) | 576 | 896 |
| `num_attention_heads` | 9 | 14 |
| `num_key_value_heads` (h_kv) | 3 | 2 |
| head dim (d/heads) | 64 | 64 |
| KV width per layer (h_kv·head_dim) | 192 | 128 |
| `intermediate_size` | 1536 | 4864 |
| `vocab_size` (V) | 49 152 | 151 936 |
| `tie_word_embeddings` | true | true |
| native `torch_dtype` | bfloat16 | bfloat16 |

### Parameter count (derivation)

Per transformer block:

- attention = `q_proj + k_proj + v_proj + o_proj`
  `= d·d + d·(h_kv·hd) + d·(h_kv·hd) + d·d`
- MLP (SwiGLU) = `gate + up + down = 3·d·intermediate`

SmolLM2‑135M
- attn/layer = 576·576 + 2·(576·192) + 576·576 = 331 776 + 221 184 + 331 776 = **884 736**
- mlp/layer  = 3·576·1536 = **2 654 208**
- per layer  ≈ 3 538 944 → ×30 = **106.2 M**
- tied embedding = 49 152·576 = **28.3 M**
- **total ≈ 134.5 M** ✓  → bf16 weights ≈ **0.27 GB**, fp32 ≈ 0.54 GB

Qwen2.5‑0.5B (has q/k/v biases, negligible)
- attn/layer ≈ 896·896 + 2·(896·128) + 896·896 = 802 816 + 229 376 + 802 816 = **1 835 008**
- mlp/layer  = 3·896·4864 = **13 074 432**
- per layer  ≈ 14 909 440 → ×24 = **357.8 M**
- tied embedding = 151 936·896 = **136.1 M**
- **total ≈ 493.9 M** ✓  → bf16 weights ≈ **0.99 GB**, fp32 ≈ 1.98 GB

### KV cache per token (all layers)

```
kv_bytes_per_token = 2 (K and V) · L · (h_kv · head_dim) · dtype_bytes
```

| | bf16 (2 B) | fp32 (4 B) |
|---|---|---|
| SmolLM2‑135M | 2·30·192·2 = **23 040 B ≈ 22.5 KiB/tok** | **46 080 B ≈ 45 KiB/tok** |
| Qwen2.5‑0.5B | 2·24·128·2 = **12 288 B ≈ 12 KiB/tok** | **24 576 B ≈ 24 KiB/tok** |

**Key observation:** Qwen2.5‑0.5B is 3.7× the parameters of SmolLM2‑135M but its
KV cache per token is only ~53 % as large (fewer layers: 24 vs 30; fewer KV
heads: 2 vs 3). Qwen's extra size sits in the FFN and the 152 k-row tied
embedding — neither scales with batch. So Qwen2.5‑0.5B batches **at least as
aggressively** as the 135 M model, not less.

---

## 3. Memory model

During generation the GPU holds:

```
M_total ≈ M_weights                              (fixed)
        + M_overhead                             (CUDA context, kernels, allocator
                                                  fragmentation ≈ 2–3 GiB)
        + B · [ S_total · kv_bytes_per_token     (KV cache — grows with batch AND length)
              + k · S_prompt · d · dtype_bytes ] (resident-layer activations + attn
                                                  scratch; k ≈ 3 empirically)
        + B · 1 · V · 4                          (lm_head logits, prefill trimmed to
                                                  the LAST position — see note)
```

- `S_prompt` = longest prompt in the batch (every batch is padded to its own max).
- `S_total`  = `S_prompt + max_new_tokens`.
- **Prefill-logits note:** current `transformers` computes logits for only the
  last prefill position (`num_logits_to_keep=1`). Full-prompt logits would be
  `B·S_prompt·V·4` — for Qwen (V = 152 k), B = 128, S = 2 k that is ~156 GB and
  an instant OOM. Do not disable that trimming.

Solving for the batch ceiling:

```
B_max ≈ (M_gpu − M_weights − M_overhead) / (S_total·kv_bytes + k·S_prompt·d·dtype_bytes)
```

A100 80 GB: nominal 79.6 GiB, take **~78 GiB usable**, `M_overhead ≈ 2.5 GiB`.

---

## 4. Worked scenarios

`k = 3`, A100 80 GB, `usable − overhead = 78 − 2.5 = 75.5 GiB` before weights.

### FaithEval — `S_prompt ≈ 1024`, `max_new_tokens = 256` → `S_total ≈ 1280`

(Contextual passages: typically 200–600 tokens; 1024 is a padded worst case.)

| model / dtype | per-seq KV | per-seq act. `3·1024·d·b` | per-seq total | `M_gpu−weights−oh` | **B_max** |
|---|---|---|---|---|---|
| SmolLM2‑135M bf16 | 1280·23 040 = 29.5 MB | 3·1024·576·2 = 3.5 MB | ≈ 33 MB | 75.2 GiB | **≈ 2 300** |
| SmolLM2‑135M fp32 | 59.0 MB | 7.1 MB | ≈ 66 MB | 75.0 GiB | **≈ 1 160** |
| Qwen2.5‑0.5B bf16 | 1280·12 288 = 15.7 MB | 3·1024·896·2 = 5.5 MB | ≈ 21 MB | 74.5 GiB | **≈ 3 600** |
| Qwen2.5‑0.5B fp32 | 31.5 MB | 11.0 MB | ≈ 42 MB | 73.5 GiB | **≈ 1 800** |

Every ceiling is 1–2 orders of magnitude above the ~1 k rows in a FaithEval task
split. **Memory is irrelevant here.** Recommend **128** (bf16 or fp32, either
model): 8 batches over a 1 k-row split, still tight length buckets. 256 is fine;
beyond that the length spread inside one batch widens and padding waste climbs.

### HaluEval — `summarization`, `S_prompt ≈ 3072`, `max_new_tokens = 128` → `S_total ≈ 3200`

(CNN/DailyMail articles — the widest length spread of the three HaluEval tasks
and the binding one; `qa`/`dialogue` prompts are ~200–400 tokens and would allow
4–8× these numbers.)

| model / dtype | per-seq KV | per-seq act. `3·3072·d·b` | per-seq total | `M_gpu−weights−oh` | **B_max** |
|---|---|---|---|---|---|
| SmolLM2‑135M bf16 | 3200·23 040 = 73.7 MB | 3·3072·576·2 = 10.6 MB | ≈ 84 MB | 75.2 GiB | **≈ 915** |
| SmolLM2‑135M fp32 | 147.5 MB | 21.2 MB | ≈ 169 MB | 75.0 GiB | **≈ 455** |
| Qwen2.5‑0.5B bf16 | 3200·12 288 = 39.3 MB | 3·3072·896·2 = 16.5 MB | ≈ 56 MB | 74.5 GiB | **≈ 1 360** |
| Qwen2.5‑0.5B fp32 | 78.6 MB | 33.0 MB | ≈ 112 MB | 73.5 GiB | **≈ 670** |

Tightest realistic case (SmolLM2‑135M, **fp32**, summarization): ceiling ≈ 455.
Recommend **256** — ~1.8× headroom in that worst case, ~5× in bf16 — and it pays
off across 10 k rows/task. Drop to **128** if `summarization` OOMs (it should not
on 80 GB). `qa`/`dialogue`-only runs can take 512.

> The training runs use `model.torch_dtype=float32`. For eval, **bf16 is
> recommended** (A100-native, ~2× the memory headroom above, faster) and stays
> comparable as long as *all* arms use the same dtype. The fp32 columns are shown
> only for the case you deliberately mirror the training dtype.

---

## 5. Empirical cross-check

The committed configs carry `batch_size: 8 #32 rtx 8000 qwen 2.5 0.5b` — i.e.
**32 ran fine on an RTX 8000 (48 GiB)** with Qwen2.5‑0.5B (fp16; Turing has no
bf16).

- RTX 8000 usable before weights ≈ 48 − 1 (smaller ctx) − 2.5 ≈ 44.5 GiB.
- If 32 was chosen with a typical 2–3× safety margin, the real ceiling there was
  ~65–100.
- Scaling the memory budget to A100: `74.5 / 44.5 ≈ 1.67×` → safe ~110–170,
  ceiling ~300+.

That is consistent with the analytic FaithEval/HaluEval numbers in §4 and is why
the recommendation lands at 128 (safe) / 256 (aggressive) rather than the raw
analytic ceilings.

---

## 6. Redoing this for another GPU or model

1. Read `L`, `hidden_size`, `num_key_value_heads`, head dim, `vocab_size` from
   the model's `config.json`.
2. `kv_bytes_per_token = 2 · L · (h_kv · head_dim) · dtype_bytes`.
3. Pick `S_prompt` for the benchmark's worst task (FaithEval ≈ 1 k; HaluEval
   `summarization` ≈ 3 k; `qa`/`dialogue` ≈ 0.3 k), `S_total = S_prompt +
   max_new_tokens`.
4. `B_max ≈ (VRAM_usable − weights − 2.5 GiB) / (S_total·kv_bytes + 3·S_prompt·d·dtype_bytes)`.
5. Use **¼–½ of `B_max`** as the configured value (padding waste, allocator
   fragmentation, and the wide length spread inside a large batch all eat into
   the analytic ceiling), and **hold it fixed across every model you compare**.
