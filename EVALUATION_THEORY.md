# Evaluation Theory — FaithEval, HaluEval, and the lm-eval Harness

> This document specifies the **methodology** of the evaluation framework in
> `eval/Eval_master`: what each benchmark measures, what the measurement is a function of,
> which design decisions in the code are protocol (and therefore frozen) versus which are
> engineering (and therefore free), and where each number stops being trustworthy.
>
> Scope, as requested: **FaithEval** (§3), **HaluEval** (§4), and the **lm-evaluation-harness
> adapter** (§5). `TruthfulQA-reproduce` and `RAGTruth-reproduce` are referenced only where
> the three in scope depend on them for contrast; they are not specified here.
>
> The section numbering is load-bearing and is cross-referenced from source docstrings.
> Companion documents: `analysis/analysis_theory.md` (what the aggregation layer estimates
> across seeds), `analysis/format_confound.md` (the empirical finding that §7 formalizes),
> and the per-benchmark `ARCHITECTURE.md` files (module-level maps).

---

## 0. The framework in one sentence

The suite measures a checkpoint along **three orthogonal axes of hallucination behaviour** —
generative faithfulness to a supplied context (FaithEval), discriminative recognition of
hallucinated text (HaluEval), and truthfulness under the community-standard log-likelihood
protocol (harness) — through a single shared model interface, so that one `model.id` is
scored identically by three benchmarks whose only intended difference is what they ask.

### Notation

| Symbol | Meaning |
| --- | --- |
| $\mathcal{M}$ | the evaluated checkpoint (Hub id, local full model, or LoRA adapter) |
| $x_i$ | the $i$-th benchmark item (context, question, and where relevant a candidate answer) |
| $g_i = \mathcal{G}(\mathcal{M}, x_i)$ | the model's realized output on item $i$ |
| $\mathcal{G}$ | the **generation operator**: prompt rendering + decoding |
| $\mathcal{P}$ | the **parser**: string $\to$ verdict / normalized answer |
| $\mathcal{S}$ | the **scoring rule**: (parsed output, gold) $\to \{0,1\}$ |
| $\hat\mu$ | the reported benchmark statistic (accuracy, MC1, MC2, …) |
| $B$ | generation batch size |
| $L$ | `max_new_tokens`, the decode budget |

The central structural claim of this document is that every reported number factorizes as

$$\hat\mu \;=\; \frac1n \sum_{i=1}^{n} \mathcal{S}\!\big(\mathcal{P}(\mathcal{G}(\mathcal{M}, x_i)),\ y_i\big)$$

and that **$\mathcal{P}$ is not neutral**. §7 shows that for two of the three benchmarks in
scope, the variance of $\hat\mu$ across this project's arms is dominated by $\mathcal{P}$
rather than by $\mathcal{M}$.

---

## 1. Architecture: one model interface, five protocols

### 1.1 The launcher as a translation layer

`run_benchmarks.py` is a Hydra composition root that owns **no evaluation logic**. It holds
one shared `model` config block and, per benchmark, a config group; `build_<benchmark>`
translates the pair into that benchmark's own CLI, and the benchmark runs as a subprocess in
its own folder (`run_benchmarks.py:184-297`).

Three properties follow, and each is a deliberate methodological choice rather than a
convenience:

1. **The benchmarks stay upstream code.** Each `*-reproduce` folder is a fork of the
   published implementation, modified only where reproduction demanded it. The launcher
   never reimplements a metric, so a number produced here is traceable to the protocol its
   authors defined.
2. **Dependency isolation is per benchmark.** `python: auto` resolves each benchmark's own
   virtualenv (`_resolve_interpreter`), because the five have mutually incompatible stacks —
   most sharply, `harness-eval` needs Python $\ge$ 3.10 and `lm_eval[hf]>=0.4.12` while the
   siblings pin $\ge$ 3.9 with `transformers` 4.41 on the cluster. Sharing one environment
   would silently resolve `lm_eval` back two years (`harness-eval/ARCHITECTURE.md`, "Python
   floor").
3. **Hardware resolution is data, not code.** `${auto_dtype:}`, `${auto_device_index:}` and
   `${auto_batch_size:}` are OmegaConf resolvers backed by `nvidia-smi` probes
   (`run_benchmarks.py:48-99`). The launcher's own venv deliberately carries no torch, so
   detection cannot import it. `bfloat16` requires compute capability $\ge$ 8.0, older GPUs
   get `float16`, CPU gets `float32` — bf16 on a generic CPU is software-emulated and slower
   than fp32, so "newer dtype" would be a pessimization.

### 1.2 The shared model contract

Every benchmark implements the same three-case resolution behind one identifier
(`FaithEval-reproduce/src/faitheval/model.py:110-148`, ported verbatim to
`HaluEval-reproduce/evaluation/hf_local.py`, and re-expressed for lm_eval in
`harness-eval/src/harness_eval/model.py:112-171`):

| Case | Detection | Resolution |
| --- | --- | --- |
| Hub id | default | `from_pretrained(model_id)` |
| Local full model | `_looks_like_local_path` (absolute, `./`, `~`, or backslash) | `from_pretrained(path)`, with `_check_local_path_exists` failing fast |
| LoRA / PEFT adapter | `adapter_config.json` present in the directory | base from the adapter config (or `--base-model-id`), adapter merged with `merge_and_unload()` |

Two subtleties in this contract are load-bearing for cross-benchmark comparability:

**Tokenizer provenance.** The sibling rule is `tokenizer_id or model_id` — i.e. the
*checkpoint's* tokenizer, which a training run from `SP-DPO-Base` always saves. lm_eval's
native `pretrained=<base>,peft=<adapter>` instead loads the tokenizer from the **base**.
`build_model_args` therefore passes `tokenizer=<adapter dir>` explicitly whenever the adapter
carries tokenizer files. Without that, one checkpoint would be scored under two different
tokenizations across the suite, and every cross-benchmark statement would be confounded.

**`model_args` as a dict, not a string.** lm_eval parses its string form with a bare
`split(",")` and no escaping, so a checkpoint path containing a comma corrupts silently. The
dict is passed; `model_args_to_string` exists for provenance only and **raises** on a
comma-bearing value rather than emit a string that misparses on read-back
(`harness-eval/src/harness_eval/model.py:174-191`).

**Fail-fast on missing paths.** `_check_local_path_exists` exists because
`from_pretrained` treats any non-existent path as a Hub repo id, producing a 404-shaped
network error for what is actually a typo. On the cluster, `_reraise_if_offline_cache_miss`
converts the same shape into the actual remedy (`huggingface-cli download …` on a login
node), because compute nodes export `HF_HUB_OFFLINE=1`. Five copies of this helper exist,
one per benchmark, and their docstrings cross-reference each other: a fix in one belongs in
all five.

### 1.3 Prompt rendering, and the base-model fallback

`HFChatGenerator` hands the pipeline a **message list**, which makes transformers apply
**the evaluated model's own chat template**. The wire format is therefore model-dependent
*by design*: each checkpoint is prompted the way it was tuned.

A base (non-instruct) model has no template and would make the pipeline raise. The generator
detects this at load time, warns, and falls back to a `\n\n` concatenation of message
contents. `generator.prompt_format` reports which path ran (`chat_template` / `concat`) and
is recorded in the summary.

> **This is a real comparability seam, not merely a robustness feature.** A base-vs-tuned
> comparison in this suite is partly a comparison of two prompt renderings. HaluEval makes
> the consequence explicit by tying its decode budget to the format (§4.5); the concat
> fallback additionally drops the system preamble that the chat rendering carries
> (`HaluEval-reproduce/ARCHITECTURE.md`).

### 1.4 Batching is throughput, not protocol — with one caveat

All three benchmarks batch generation and sort prompts by token length before cutting
batches. The reasoning is uniform (`evaluator.batch_order`, `evaluate._batch_order`):

- The pipeline pads every batch to *its own* longest prompt. In file order, where length is
  effectively random, one long context drags a whole batch up to its size. Length bucketing
  removes most of that waste at no protocol cost.
- **Descending**, not ascending, so the peak-memory batch runs *first*: a CUDA OOM surfaces
  in the first minute instead of at 90 % completion.
- Decoder-only batched generation requires **left padding** (`tokenizer.padding_side =
  "left"`); with the default right padding, pad tokens sit between prompt and first generated
  token, and short prompts in a batch decode from padding — a silent quality degradation.

The caveat, stated in every relevant docstring and config comment: padding shifts logits
slightly, so a greedy argmax **tie** can break the other way. On the pinned stack
(torch 2.2.2 / transformers 4.41) batch sizes 1/4/8 were verified to reproduce unbatched
generations byte-for-byte over ragged prompts in both formats, but this is not guaranteed for
an arbitrary model. Hence the standing rule:

$$\textbf{Keep } B \textbf{ and } \texttt{sort\_by\_length} \textbf{ FIXED across every model compared.}$$

Both are recorded in each `summary.json` so comparability is checkable after the fact rather
than reconstructed from a launch command.

### 1.5 Artifact discipline

Every benchmark writes (a) per-item records and (b) a `summary.json` carrying the headline
metric **plus every setting that can move it**. Two further conventions matter:

- **Streaming, indexed writes.** FaithEval streams predictions as they are produced, so a
  killed job still leaves usable output; the `index` field is what makes partial rows
  attributable. Since execution order is length-sorted, `_sort_predictions_file` restores
  dataset order before returning — the ordering is an execution detail, not a result, and the
  artifact must be diffable against a run made under a different `sort_by_length`.
- **Re-scorability.** HaluEval stores the judge's `raw_judgement` next to the parsed verdict.
  This is what makes §7's re-analysis possible with no GPU: generation is the expensive step
  and is paid once, while parser choice becomes a post-hoc, revisable decision.

---

## 2. What the three benchmarks measure, and why all three are needed

| | FaithEval | HaluEval | harness (lm_eval) |
| --- | --- | --- | --- |
| **Model's role** | answer generator | **Yes/No judge** | ranked continuation scorer |
| **Question asked** | is the answer faithful to *this context*? | can the model *recognize* a hallucination? | does the model prefer true continuations? |
| **Output consumed** | free-form generation | free-form generation | log-likelihoods |
| **Scoring** | substring / exact match | strict Yes/No parse | argmax over fixed options |
| **Format-sensitive?** | **yes**, both directions | **yes**, punishes verbosity | **no** |
| **Grounding** | supplied context | world knowledge + given text | fixed option set |

The axes are complementary in a way that matters for the thesis claim:

- **Generation vs. discrimination.** FaithEval scores what the model *says*; HaluEval scores
  what it *recognizes*. A preference-optimized model can improve on one and not the other,
  and the two are not redundant measurements of one latent "hallucination" quantity.
- **Grounded vs. parametric.** FaithEval's ground truth is the supplied context — including
  contexts that are counterfactual, i.e. that *contradict* world knowledge (§3.2). It
  therefore measures context-following, not knowledge. TruthfulQA (via the harness) measures
  the parametric axis instead.
- **Parser-dependent vs. parser-free.** This is the axis §7 turns on. The harness path is the
  only one in the suite where $\mathcal{P}$ is the identity, which makes it the **control**
  against which the other two are read.

---

## 3. FaithEval — contextual faithfulness of the model's own generations

### 3.1 The construct

FaithEval presents a context and a question and asks for an answer. Faithfulness is defined
*relative to the context*, and each task isolates one way a context can be adversarial:

| Task | Context property | Faithful behaviour | Scoring |
| --- | --- | --- | --- |
| `unanswerable` | does not contain the answer | refuse — say "unknown" | `phrase_match` |
| `inconsistent` | multiple documents support contradictory answers | flag the conflict | `phrase_match` |
| `counterfactual` | contradicts common sense / world knowledge | follow the context anyway | `answer_match` |

The three probe distinct failure modes: fabricating an absent answer, silently picking one
side of a contradiction, and overriding the context with parametric knowledge. Only the
third rewards *ignoring* what the model believes — which is why it uses a different scoring
rule and why its prompt deliberately carries **no** task-specific instruction
(`configs/counterfactual.yaml`: the instruction line is commented out, matching upstream).

### 3.2 Tasks are data, not code

A task is fully described by a YAML file read into a frozen `TaskConfig`
(`config.py:23-41`): `dataset_name`, `scoring`, `task_specific_prompt`, the valid-phrase
lists, and the `*_column` names. Adding a task or a prompt variant is a new YAML plus an
entry in `SUPPORTED_TASKS` — no change to generation or scoring code. This is the same
"data, not code" discipline HaluEval applies to its judge instructions (§4.2).

Validation is placed where the invariant actually lives. `TaskConfig.__post_init__` rejects
`phrase_match` with empty `valid_phrases`; but `strict_valid_phrases` is only reachable
through the run-time `--strict-match` flag, so `EvalConfig.__post_init__` checks *that*
combination (`config.py:72-91`). Without the second check, `--strict-match` on a config
lacking the strict list would score every example wrong and report **accuracy 0.0 with no
error** — a silent zero indistinguishable from total model failure.

The multiple-choice variant (§3.8) needed code once: a third scoring mode, `choice_loglik`, with
its own check (`TaskConfig` rejects it without `choices_column` and `answer_key_column`). The
task itself is data again: `configs/counterfactual_mc.yaml`.

### 3.3 The prompt

```
You are an expert in retrieval question answering. \n
Please respond with the exact answer only. Do not be verbose or provide extra information.
{task_specific_prompt}
Context: {context}
Question: {question}
Answer:
```

The trailing space before the newline in the base instruction is FaithEval's own prompt text
and is preserved byte-for-byte; `prompting.py` writes it as an explicit `\n` escape so the
source carries no trailing whitespace while the string stays identical to the published
prompt. This is not pedantry — it is part of the conditioning, and "tidying" it changes every
prompt and hence every score.

Note the instruction **"respond with the exact answer only. Do not be verbose"**. §7 shows
that this instruction is precisely what the arms under test stopped obeying, and that the
scoring rules convert that disobedience into apparent hallucination differences.

### 3.4 Normalization and the two scoring rules

All comparisons run over SQuAD-style normalization (`metrics.normalize_answer`): underscores
to spaces, lowercase, punctuation (plus `‘’´``) to spaces, articles `a|an|the` removed,
whitespace collapsed.

**`phrase_match`** ($\texttt{unanswerable}$, $\texttt{inconsistent}$) is true iff any valid
phrase occurs **as a substring** of the normalized prediction:

$$\mathcal{S}_{\text{phrase}}(g) = \mathbb 1\!\left[\ \exists\, p \in V:\ p \subseteq \text{norm}(g)\ \right]$$

This is a substring test, not a word match, and it is therefore **biased upward**. `"not"` —
a valid phrase for `unanswerable` — matches inside *notable*, *nothing*, *cannot*, *another*,
so answers that never refused still score correct. The bias is retained deliberately: it is
FaithEval's own rule, and switching to word-boundary matching would move this fork away from
both the published numbers and every run already recorded in the project. The methodological
consequence is stated in the docstring and inherited here:

> **`phrase_match` accuracies are an upper bound.** Prefer `--strict-match` — which narrows
> each task to a single unambiguous phrase (`unknown`, `conflict`) — whenever the absolute
> level matters rather than the relative ordering of arms.

**`answer_match`** (`counterfactual`) is exact equality after normalization against any
reference:

$$\mathcal{S}_{\text{exact}}(g) = \mathbb 1\!\left[\ \exists\, r \in R:\ \text{norm}(g) = \text{norm}(r)\ \right]$$

The two rules have **opposite length biases**, which is the single most consequential fact in
this document: a longer answer is more likely to contain a valid phrase by chance, and less
likely to equal a reference exactly. §7.

### 3.5 Data loading: local JSONL, deliberately

Splits are read from `data/faitheval/<slug>/<split>.jsonl` and built with
`Dataset.from_list`; nothing touches the Hub or the arrow cache at eval time.

The reason is a genuine version incompatibility, not caution. The cluster pins
`datasets<3`, but an arrow cache written by `datasets>=4` encodes list columns with the newer
`List`/`LargeList` feature types, which resolve to `typing.List` (not a dataclass) under the
pin and die with `TypeError: must be called with a dataclass type or instance`. Plain JSONL
carries no version-stamped metadata and loads on any version. `scripts/prepare_datasets.py`
materializes the files once on an internet-connected machine; `FAITHEVAL_DATA_DIR` relocates
them; the Hub path is preserved commented at the bottom of `data.py` for unpinned online use.

The harness cannot take this route — lm_eval loads its own datasets — so it solves the same
problem with a runtime shim instead (§5.5).

### 3.6 Metrics reported

Beyond `accuracy = num_correct / num_examples`, each summary carries a **degeneration probe**:

| Field | What it detects |
| --- | --- |
| `mean_prediction_tokens` | the model ignoring "exact answer only" |
| `truncation_rate` | share of predictions that hit the $L$-token budget |
| `mean_prediction_words` | continuity with earlier runs |

Tokens are primary because they are directly comparable to `max_new_tokens`, the budget the
probe exists to detect a model running into; `truncation_rate` names that failure outright.
The token count re-tokenizes the *decoded* string — the pipeline discards raw generated ids —
so it is within a token or two of the true length, which is all the probe needs.

These fields are not decoration. Under §7 they turn out to carry more of the signal than
`accuracy` does, because an arm whose mean answer length runs into the hundreds of tokens is
failing the instruction, and that failure shows up as near-zero exact-match accuracy *and* a
multi-hour runtime from the same cause.

### 3.7 Decoding

Greedy by default (`do_sample=False`), $L = 256$. `--temperature` / `--top-p` are rejected
without `--do-sample` rather than silently ignored. Greedy decoding removes sampling noise
from the estimate; the residual non-determinism is only the padding-tie effect of §1.4.

### 3.8 Multiple-choice scoring (opt-in, modified protocol)

`counterfactual_mc` scores the counterfactual split without generating anything. The prompt is
unchanged (§3.3). Each option $c_j$ in the item's `choices` is scored as the model's answer:

$$s_j = \sum_{t} \log p_\theta\!\left(c_{j,t} \mid \text{prompt},\, c_{j,<t}\right),\qquad
\hat\jmath = \arg\max_j s_j,\qquad \hat\jmath_{\text{norm}} = \arg\max_j \frac{s_j}{|c_j|_{\text{chars}}}$$

`accuracy` is the share of items where $\hat\jmath$ is the `answerKey` option. `accuracy_norm`
uses the per-character score instead. That is lm-eval's `acc_norm`, which removes the advantage
short options get from having fewer tokens to pay for.

- **Context tokens.** The context is tokenized exactly as `prompt_token_lengths` counts it.
- **Option tokens.** After a chat template's assistant header the option follows bare. In the
  concatenation format it follows one space, continuing the "Answer:" line.
- **Padding.** Sequences are right-padded; under causal attention, padding after a sequence cannot
  change its logits. Each forward pass holds `batch_size` sequences, the memory of a generation
  batch of the same size.
- **Raw logits.** The scores come from the model's raw logits; no generation config is applied.

**What it measures.** There is no answer parser, so the length bias of §7.1 does not apply: a
verbose model and a terse one are scored on the same option strings. It is not the published
protocol. It is reported as `faitheval.mc`, beside `counterfactual` and never instead of it (I8).

---

## 4. HaluEval — hallucination *recognition*

### 4.1 The construct, and how it differs

HaluEval inverts the model's role. For each item the model is shown a
`(question / dialogue history / document, candidate answer)` pair where the candidate is
**randomly** either the ground-truth output or a pre-generated hallucinated one, and must
answer `Yes` (hallucinated) or `No` (faithful).

$$y_i = \begin{cases}\texttt{Yes} & u_i > 0.5 \quad (\text{shown the hallucinated output})\\ \texttt{No} & \text{otherwise}\end{cases}, \qquad u_i \sim \mathcal U(0,1)$$

So the evaluated checkpoint is the **judge**, not the answer generator — unlike every other
benchmark in the suite. Three tasks (`qa`, `dialogue`, `summarization`) differ only in field
names and system message; each split is 10,000 rows.

### 4.2 Judge instructions as data

Each task's instruction is a plain `.txt` at
`evaluation/<task>/<task>_evaluation_instruction.txt`, read at run time and prepended to the
per-example fields. Same principle as FaithEval's `configs/`: the prompt changes without a
code change.

Each `build_*_request` returns **both** renderings side by side — `message` (chat turns, with
a system preamble) and `prompt` (the flat string the original sent to `openai.Completion`).
`generate` picks by what the model supports. The fallback is therefore HaluEval's *own*
completion format, not one this wrapper invented — though note it omits the system preamble
the chat rendering carries.

### 4.3 The label-sampling problem, and what this fork does

Upstream **never seeds** the coin flip. Consequences, spelled out in
`evaluation/REPRODUCIBILITY.md`:

1. Each run scores a different random 50/50 partition — the judge effectively sees a
   different dataset every time.
2. Reported accuracy is a noisy random variable, not a fixed number for a given model.
3. Comparing two checkpoints without pinning the seed is **unpaired**: part of any delta is
   *which half got sampled*.

Note what the problem is **not**: judge-side stochasticity is already eliminated (OpenAI at
`temperature=0.0`, local HF at `do_sample=False`). The randomness is purely in label
sampling.

The document enumerates three remedies — (1) seed it; (2) evaluate *both* classes per item,
which is deterministic and decomposes accuracy into TPR/TNR at $2n$ judge calls; (3) keep the
design and sweep $K$ seeds for mean $\pm$ std. **This fork implements option 1 plus the
diagnostics option 2 was wanted for:**

- `--seed` (launcher `halueval.seed`, default 42) pins the draw. The published sampling
  design is kept, because it is what makes these numbers comparable to the paper.
- **TPR and TNR are reported anyway**, at $n$ calls rather than $2n$. Option 2's headline
  argument was that a constant-"Yes" judge scores ~50 % and looks unremarkable; per-class
  recall on the sampled half exposes exactly that, on half the sample.
- Option 3 is a one-liner: `./run_all.sh --multirun run='[halueval]' halueval.seed=42,43,44,45,46`.

**Seed–order independence.** The draw is executed for the whole split in ascending index
order *before* any batching or reordering (`evaluate.py`, "Phase 1"). `random()` is therefore
called exactly once per index in a fixed order, so the same seed yields identical labels
regardless of `batch_size` or `sort_by_length`. The length-sort (Phase 1.5) moves only *when*
a row is judged, never *which* answer it is shown. This is pinned by
`tests/test_evaluate_ordering.py`.

### 4.4 The parser, and why an unparseable verdict is *wrong*

The published rule, reproduced character for character:

```python
ans = text.replace(".", "")
if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
    return None          # -> "failed!", counted INCORRECT, kept in the denominator
return "Yes" if "Yes" in ans else "No"
```

Two properties are decisive:

1. **No word boundaries.** Bare `"No" in ans` also fires on *Not*, *Note*, *None*,
   *November*. A `Yes` verdict that goes on to use any of them reads as containing both
   tokens and is discarded as ambiguous.
2. **Unparseable $\Rightarrow$ incorrect**, not excluded. Accuracy therefore silently mixes
   *judgement quality* with *output-format compliance*.

This is reproduced faithfully — changing it would put this fork's numbers on a different
scale from the paper's — but it is instrumented rather than left implicit.

### 4.5 The decode budget is part of the protocol

Because an unreadable verdict scores as wrong, $L$ is not cosmetic: a judge cut off
mid-sentence ("*it does not contain hallucin|*") is counted as a **wrong answer** rather than
an unfinished one. `DEFAULT_MAX_NEW_TOKENS` therefore matches the OpenAI endpoint each prompt
format stands in for:

| `prompt_format` | Stands in for | Default $L$ |
| --- | --- | --- |
| `concat` | `openai.Completion.create(...)`, whose own default `max_tokens` is 16 | **16** |
| `chat_template` | `openai.ChatCompletion.create(...)`, called with *no* `max_tokens` | **128** |

A 16-token cap on the chat format would penalize verbose-but-correct judges that the
published protocol scored fine; measured on this project's checkpoints, 16 $\to$ 64 **halved**
the unparseable rate on the same rows. 128 rather than literally uncapped because
10,000 rows $\times$ 3 splits makes an unbounded budget impractical, and a judge that has not
said Yes/No within 128 tokens is refusing, not truncated.

### 4.6 Diagnostics: the three ways a judge fails

`_Tally` produces `correct` / `incorrect` / `accuracy` exactly as upstream does, then adds
metrics that cost nothing and change no score. They exist because accuracy alone cannot
distinguish:

| Failure mode | `accuracy` | `format_compliance` | `tpr` / `tnr` |
| --- | --- | --- | --- |
| judges wrongly | drops | stays 1.0 | both moderate |
| never emits a verdict | drops | **drops** | both drop |
| answers one constant | $\approx$ label base rate | 1.0 | **one is 0.0** |

The third is the dangerous one: a judge answering "No" to everything scores ~0.50 and looks
merely mediocre. Formally, with $\pi = \Pr[y=\texttt{Yes}] \approx 0.5$, a constant-"No" judge
has $\hat\mu = 1-\pi$, $\mathrm{TPR}=0$, $\mathrm{TNR}=1$. Reported fields:

$$\texttt{format\_compliance} = \frac{n - \texttt{failed}}{n},\qquad
\texttt{judged\_yes\_rate} = \frac{\texttt{judged\_yes}}{n - \texttt{failed}}$$

with `judged_yes_rate` $\in \{0,1\}$ flagged outright as *degenerate judge*. **This mattered
immediately: the checkpoints evaluated in `outputs/eval/v3` answer `"No"` to every row on
several arms.**

### 4.7 Offline re-scoring

`score_results.py` recomputes every metric from a stored results file — no model, no GPU, no
re-run — and reports **two scorings side by side**:

- **`strict`** — the published rule, reproduced exactly. *The only one that belongs in a
  results table without a caveat.*
- **`lenient`** — case-insensitive, word-boundary, first-verdict-wins. **Not** comparable to
  published numbers; it exists to *size the gap*.

The interpretation rule is what makes the pair useful: when the two agree, the strict parser
is costing nothing and the failures are real refusals; when they diverge, **the difference is
the share of the score that is output formatting rather than hallucination detection**.

The tool also surfaces non-comparability rather than absorbing it: rows lacking
`ground_truth` are excluded from the denominator and reported as `num_skipped`, the signature
of an interrupted run whose accuracy is over fewer rows than the split has. A run predating
the `raw_judgement` field still scores strictly (the stored `judgement` is already the parsed
verdict) but has no lenient scoring available — precisely the gap that field closes.
`tests/test_scoring.py` pins `parse_strict` against the live copy in `evaluate.py` so the two
cannot drift.

### 4.8 Constrained scoring (opt-in, modified protocol)

`halueval.scoring=constrained` (`evaluate.py --scoring constrained`) replaces decoding and parsing
with one prefill per row. For prompt $x_i$, read the next-token distribution at the first answer
position, and pool each verdict over the vocabulary tokens that spell it, each token id once:
$V_{\text{Yes}} = \{v : \operatorname{decode}(v) \in \{\texttt{Yes}, \texttt{␣Yes}, \texttt{yes},
\texttt{␣yes}, \texttt{YES}, \texttt{␣YES}\}\}$, and likewise $V_{\text{No}}$. Matching the decoded
text of the vocabulary, rather than encoding each spelling, keeps a SentencePiece tokenizer's
`▁Yes` and `Yes` as two tokens instead of collapsing two spellings onto one:

$$\ell^{\text{Yes}}_i = \log \sum_{v \in V_{\text{Yes}}} p_\theta(v \mid x_i),\qquad
\ell^{\text{No}}_i = \log \sum_{v \in V_{\text{No}}} p_\theta(v \mid x_i),\qquad
s_i = \ell^{\text{Yes}}_i - \ell^{\text{No}}_i$$

The headline is $\text{AUROC}(s, y)$ with $y_i = [\text{ground truth} = \texttt{Yes}]$
(Mann–Whitney, ties at average rank), with its Hanley–McNeil SE. Beside it the summary reports:

- `accuracy_argmax`, `tpr`, `tnr` and `judged_yes_rate`, for the argmax verdict
  $\arg\max(\ell^{\text{Yes}}, \ell^{\text{No}})$;
- the verdict mass $m_i = e^{\ell^{\text{Yes}}_i} + e^{\ell^{\text{No}}_i}$, as its mean, its
  median and `frac_mass_below_half`.

Each design decision is pinned by a test:

- **Same labels.** Phase 1 (the label draw) and Phase 1.5 (the execution order) are shared with
  generate mode, so each row is shown the same answer in both, by construction.
- **Same prompt tokens.** The prefill renders exactly what `prompt_token_lengths` counts, which is
  what the pipeline generates from.
- **Positions under left padding.** A plain forward does not derive positions from the attention
  mask (`generate` does), so `position_ids = cumsum(mask) − 1`. Batch 3 reproduces batch 1.
- **Raw logits.** No logits processor is applied, including a checkpoint's own
  `repetition_penalty`. The vocabulary argmax equals greedy's first token with the penalty off.
- **Artifacts apart.** Rows go to `<task>_<label>_constrained_results.json`. Under
  `constrained`, the original results file is never opened; `both` writes both over one model
  load.

**Why AUROC, not accuracy.** A threshold-free score cannot be earned by answering one label: a
constant judge scores 0.5 however the labels fall. That is exactly the degenerate case §4.6
exposes.

**Why the verdict mass must be reported.** When $m_i \ll 1$, "Yes" and "No" are both unlikely
continuations. The argmax between them is then a forced choice, not the model's conclusion.

### 4.9 Eval-side decontamination (opt-in, modified protocol)

503 of the 10,000 `summarization` rows are documents the RAG-Truth study trained or selected on
(`SP-DPO-Base/KNOWN_ISSUES.md` §5, measured by `analysis/overlap.py`). There are three ways to
rescore the stored rows without them:

- `score_results.py --exclude LIST --emit-summary DIR`, for one results file;
- `analysis/run_decontam.sh`, for a whole eval root;
- `halueval.decontam.enabled=true`, at the end of a live run.

The `*_decontam_summary.json` scores five row sets side by side, each with its SE:

| Row set | Rows | Note |
| --- | ---: | --- |
| original | 10,000 | |
| decontaminated | 9,497 | the headline |
| seen | 503 | |
| clean | 9,367 | |
| unseen-exposed | 130 | documents only the test side or the release saw |

It also reports the seen-minus-clean contrast, whose SE is ≈ 0.023 at accuracy ≈ 0.5. That
contrast is the diagnostic. The headline itself can move by at most
$0.0503 \cdot |\text{acc}_{\text{seen}} - \text{acc}_{\text{clean}}|$, about one eval SE. A
constrained results file gets the same treatment, scored by AUROC.

---

## 5. harness-eval — the community-standard protocol

### 5.1 Why a second TruthfulQA number

`TruthfulQA-reproduce` already scores TruthfulQA using the **original authors' scripts**.
That is not how the field reports it: modern papers and leaderboards run TruthfulQA through
EleutherAI's lm-evaluation-harness. So `harness-eval` is deliberately **parallel**, not a
replacement:

| | protocol | comparable to |
| --- | --- | --- |
| `TruthfulQA-reproduce` | `prompt_style: chat` (default) | the suite's other benchmarks |
| `harness-eval` | lm_eval defaults, `apply_chat_template: false` | published / leaderboard scores |

Both are reported, and **their MC1/MC2 will differ — that difference is itself the
measurement** of how much the prompt rendering is worth. The harness path additionally
reaches multilingual TruthfulQA (Okapi's 31 languages, HiTZ's 5), which the original-script
path cannot.

> **Attribution.** The evaluation is EleutherAI's lm-evaluation-harness running TruthfulQA
> (Lin, Hilton & Evans, 2022) and its multilingual extensions (Lai et al., 2023; Calvo
> Figueras et al., 2025). Cite those, not this wrapper.

### 5.2 Scope of the adapter

lm_eval owns the tasks, prompts, and scoring. The module does exactly three translations —
model args (§1.2), task-name resolution (§5.3), result normalization (§5.4) — and nothing
else. `lm_eval` (and hence torch and datasets) is imported **only** inside `evaluator` and the
`--list-tasks` path, so `--help` and argument errors stay instant.

### 5.3 Task resolution: tags don't aggregate

`resolve_tasks` validates names against the *installed* lm_eval (spellings drift across
0.4.x), expands globs, collapses duplicates preserving order, and raises with a
`difflib` close-match suggestion on an unknown name. Names come from the union of whatever
`all_tasks` / `all_groups` / `all_subtasks` / `all_tags` attributes exist, because that
surface has shifted between versions.

The methodological point: all three configured benches are lm_eval **tags**, not **groups**.
A tag expands to subtasks and lm_eval returns per-subtask rows with **no `groups`
aggregate** — so **there is no single cross-language MC2**. `flatten_results` handles
`groups` if present but never depends on it, and keys rows on `task_name`, never on `alias`
(group-member aliases arrive indented, `" - truthfulqa_mc1"`).

```
truthfulqa              -> truthfulqa_mc1, truthfulqa_mc2, truthfulqa_gen        (English)
truthfulqa_multilingual -> truthfulqa_<lang>_{mc1,mc2}, 31 okapi languages       (62 tasks)
truthfulqa-multi        -> truthfulqa-multi_{mc1,mc2,gen}_{en,es,ca,eu,gl}       (HiTZ)
```

### 5.4 Result normalization

lm_eval returns a deeply nested dict keyed by `"metric,filter"` pairs. `flatten_results`
emits one row per `(task, metric, filter)` carrying `value`, `stderr`, `higher_is_better`,
`num_fewshot`, `version`, and `n_samples`. Two details worth naming:

- `*_stderr` keys are consumed as their base metric's `stderr`, not emitted as rows; lm_eval
  writes the literal string `"N/A"` for an absent stderr, which is normalized to `None`.
- `higher_is_better` travels **with the row**. This is what lets the downstream analysis layer
  rank a metric correctly without a lookup table that could disagree between two figures
  (`analysis_theory.md` §1).

Outputs: `summary.json` (caller provenance + lm_eval's own `lm_eval_version`,
`transformers_version`, `git_hash`, `date`, `chat_template_sha` + flat rows),
`lm_eval_results.json` (raw dump minus samples), and optionally `samples.jsonl`
(one record per document, `ensure_ascii=False` to keep 31 languages' text literal).

### 5.5 Version-pinning as a correctness concern

Three decisions are regression-tested because each is a silent-wrongness risk, verified
against lm_eval source at tags v0.4.5 and v0.4.12:

1. **`model_args` dict over string** — §1.2.
2. **PEFT adapters keep the checkpoint's own tokenizer** — §1.2. Scoring an adapter with the
   base tokenizer here but the checkpoint tokenizer everywhere else would break exactly the
   cross-benchmark comparison the thesis rests on.
3. **Every behaviour-bearing kwarg is passed explicitly.** `simple_evaluate`'s
   `fewshot_as_multiturn` default flipped `False → True` between 0.4.5 and 0.4.12, and the
   guard rejecting multiturn without a chat template was removed. So `evaluator` passes every
   such kwarg by keyword (`simple_evaluate` is `@positional_deprecated`), `config.py`
   **reinstates** the multiturn-requires-chat-template check, and `test_tasks.py` asserts by
   `inspect.signature` that the installed `simple_evaluate` accepts the whole kwarg set.

The `datasets` incompatibility of §3.5 recurs here, and cannot be solved by pre-materializing
JSONL because lm_eval loads its own data. `install_list_feature_compat` instead wraps the
feature deserializer to rewrite `List`/`LargeList` `_type`s to the legacy `Sequence`; it is a
true no-op on `datasets>=4` and idempotent.

### 5.6 Settings that move the number

| Setting | Effect | Rule |
| --- | --- | --- |
| `apply_chat_template` | **materially** moves scores — MC2 0.06 vs 0.26 on a smoke test | `false` for leaderboard comparability; fix across compared models |
| `num_fewshot` | `null` = task default (TruthfulQA is 0-shot; its QA primer is baked into the prompt upstream) | leave null unless ablating |
| `limit` (`num_samples`) | applies **per task** — 5 over 62 okapi tasks is 310 evaluations, not 5 | any non-null value makes the run non-comparable to published scores |
| `batch_size` | `"auto"` on GPU (lm_eval probes via CUDA OOM retries); fixed on CPU, where that probing is unreliable | — |

Every one of these is written into `summary.json` provenance.

### 5.7 Why this is the suite's control

The harness path scores TruthfulQA MC1/MC2 by **log-likelihood ranking over fixed options**.
There is no generation, no parser, and no length sensitivity: $\mathcal{P}$ is the identity
and $\mathcal{G}$ produces scores rather than text. It is the only *original* protocol in scope
where the confound of §7 **cannot operate by construction**, which is what makes it the reference
against which FaithEval and HaluEval results must be read. The opt-in modified protocols
constrained HaluEval (§4.8) and FaithEval `counterfactual_mc` (§3.8) share the property, but they
are not the published protocols and are reported beside the originals, never instead of them.

(`truthfulqa_gen` is the exception within the tag — it *is* generative and rouge/bleu-scored,
so it does not share the control property of the MC tasks.)

---

## 6. Cross-cutting invariants

These hold across all three benchmarks and are the conditions under which a comparison
between two checkpoints is valid at all.

**I1 — One model interface.** A checkpoint is resolved identically (weights, base merge,
tokenizer) in all three. Violating this makes cross-benchmark statements meaningless (§1.2).

**I2 — Fixed generation settings across compared arms.** $B$, `sort_by_length`, $L$,
`apply_chat_template`, and HaluEval's `seed` must be identical across every arm in a
comparison. All are recorded in `summary.json` precisely so this is auditable post hoc.

**I3 — Greedy decoding.** Every generative path decodes greedily, so no sampling variance
enters. Residual non-determinism is padding-tie only (§1.4).

One caveat is unverified. Qwen2.5-0.5B-Instruct's `generation_config.json` is believed to set
`repetition_penalty: 1.1`, which the pipeline would then apply to these "greedy" decodes. The
modified-protocol pilot prints the config (`SP-DPO-Base/FUTURE_EXTENSIONS.md` §7). The constrained
and multiple-choice scorers read raw logits and are unaffected either way.

**I4 — Order-independence of results.** Execution order is a throughput decision; artifacts
are restored to dataset order (FaithEval) or carry an `index` (both). HaluEval's labels are
drawn before reordering (§4.3).

**I5 — Provenance travels with the number.** Nothing that can move a score is left to a
launch command: `prompt_format`, `max_new_tokens`, `batch_size`, `sort_by_length`, `seed`,
`limit`, `apply_chat_template`, and lm_eval's own version stamps are all in the summary.

**I6 — Reproduce upstream, instrument alongside.** Where a published rule is
methodologically poor (HaluEval's parser, FaithEval's substring match), it is **kept** and a
diagnostic is added next to it. The published number stays comparable; the diagnostic makes
it interpretable. No metric in this suite silently deviates from its source protocol.

**I7 — What varies across seeds is *between-run* variance only.** Every CI and $p$-value
produced downstream counts seeds, not items; item-level uncertainty is a separate quantity
that each benchmark's own `stderr` estimates where it reports one (`analysis_theory.md` §1.1).

**I8 — Modified protocols never replace originals.** Some scorers depart from the published
protocol: constrained HaluEval (§4.8), decontamination (§4.9) and multiple choice (§3.8). Each
runs only when switched on, and writes its own artifacts, in the study to its own eval root. The
analysis reports each under its own benchmark name (`halueval.constrained`, `halueval.decontam`,
`halueval.constrained_decontam`, `faitheval.mc`) in a separate `modified/` tree. There it never
shares an aggregate, a comparison family or a figure with an original number. With the switches
off, every original artifact is byte-identical to what it was before the switches existed.

I6 keeps the published rule; I8 keeps an alternative from quietly taking its place.

---

## 7. The format confound — the framework's principal validity threat

This section states, as methodology, what `analysis/format_confound.md` established
empirically over the `outputs/eval/v3` ensemble (132 HaluEval result files, 1,311,552 scored
rows). It is the most important caveat in this document.

### 7.1 The mechanism

Write the reported score as $\hat\mu = \frac1n\sum_i \mathcal{S}(\mathcal{P}(g_i), y_i)$ and
let $\ell_i = |g_i|$ be the output length. The three scorers in scope have **different, and
in two cases opposite, dependencies on $\ell$**:

| Scorer | Rule | $\partial\,\mathbb E[\mathcal{S}] / \partial \ell$ |
| --- | --- | --- |
| HaluEval `parse_strict` | one capitalized `Yes`/`No`, else **wrong** | **negative** — longer text more likely to contain both/neither |
| FaithEval `answer_match` | exact equality with gold | **negative** — only a short span can equal the reference |
| FaithEval `phrase_match` | substring containment | **positive** — more tokens, more chances to hit a phrase |
| harness MC1/MC2 | log-likelihood ranking | **zero** — no generation |

If an intervention changes output length without changing faithfulness, $\hat\mu$ moves on
the first three, in opposite directions, and not at all on the fourth.

### 7.2 The observed signature

That is exactly what was observed. Instruction tuning made the arms verbose:

| Scorer | Length bias | Observed effect of training |
| --- | --- | --- |
| HaluEval `parse_strict` | punishes | large apparent **drop** |
| FaithEval `answer_match` (counterfactual) | punishes | large apparent **drop** |
| FaithEval `phrase_match` (unanswerable) | rewards | large apparent **gain** |
| TruthfulQA log-likelihood | none | small, uniform gain |

Effects appear exactly where the parser is length-sensitive and vanish where it is not —
a property of $\mathcal{P}$, not of $\mathcal{M}$.

The decisive quantities:

- $\mathrm{corr}(\texttt{format\_compliance}, \texttt{accuracy}) = \mathbf{0.9974}$ over all
  132 HaluEval files. Reported accuracy is, to three significant figures,
  $\texttt{compliance} \times 0.498$.
- Accuracy **among rows that parsed**: $\mathbf{0.4981 \pm 0.0165}$ — chance — on every arm,
  every seed, every task, against a reported range of 0.020–0.527.
- 17.9 % of all rows were discarded as unparseable and scored wrong.
- On FaithEval `counterfactual`, exact match reports 0.005 overall while **27.1 % of
  predictions *contain* the gold answer**.

> **A metric that is a deterministic multiple of "did the output parse" is not measuring
> hallucination.** The 25-point spread in the arm table is the compliance column and nothing
> else.

### 7.3 What this does and does not invalidate

**Does not survive:** any ranking of arms by HaluEval accuracy, or by FaithEval
`counterfactual` / `unanswerable`, or any base-vs-trained comparison on those axes. In
particular the curriculum-ordering contrasts (`asc` / `desc` / `shuffled`) on HaluEval are
**compliance contrasts**.

**Survives:** TruthfulQA MC1/MC2 (§5.7), and the aggregation machinery itself, which
correctly aggregated the numbers it was given.

**The honest negative result:** after 2 epochs no arm exceeds chance on HaluEval binary
judgement *even on the rows it answers in the required format*. That is a real, reportable
finding — and a stronger one than the artefactual ranking it replaces, because it is not a
measurement of formatting.

Note also that the base model's ~0.50 is **degenerate, not good**: ~100 % compliance times
chance accuracy. HaluEval as scored cannot distinguish a perfect detector from a model that
always says `No`.

### 7.4 Remediation, in order of preference

1. **Never report raw HaluEval accuracy alone.** Report `format_compliance` and conditional
   accuracy (`correct / parsed`) as two columns; they decompose the metric into the
   formatting part and the judgement part. Only the latter is the claim.
2. **Constrain the decode rather than loosening the parser.** *Implemented, opt-in (§4.8).*
   Score HaluEval by comparing the log-probabilities of the `Yes` / `No` continuations, or
   restrict sampling to those tokens. Compliance becomes 100 % by construction and the metric
   measures discrimination. This is strictly preferable to `parse_lenient`, which merely
   re-labels the artefact: it recovers only part of the gap and is non-comparable to published
   numbers either way.
3. **Report `counterfactual` under a parser-free rule beside exact match**, since exact match
   against a chat model measures verbosity; exact match stays for comparability with the paper.
   *Implemented, opt-in, as multiple choice (§3.8).* Containment as a clearly-caveated secondary
   is still proposed (`SP-DPO-Base/FUTURE_EXTENSIONS.md` §6).
4. **Check $L$.** Many generations terminate mid-sentence, so a verdict that would have
   arrived after the cutoff is lost outright; some share of the 17.9 % is truncation, not
   refusal. FaithEval's `truncation_rate` (§3.6) is the corresponding probe.
5. **Treat `phrase_match` as an upper bound**, per its own docstring, and prefer
   `--strict-match` where the absolute level matters.

### 7.5 The general lesson for this framework

The suite's design already contained the instruments needed to detect this — `format_compliance`,
`tpr`/`tnr`, `judged_yes_rate`, `mean_prediction_tokens`, `truncation_rate`, `raw_judgement`,
and the strict/lenient pair — and they were added *before* the anomaly appeared, on the
argument of §4.6 that accuracy alone cannot separate three distinct failure modes. That is
the generalizable methodological claim:

> **A benchmark that reports only its headline metric cannot be debugged.** Every scorer in
> this framework therefore reports, alongside the score, the quantities that would explain a
> surprising value — and preserves enough raw output that the scoring decision can be revisited
> without re-running generation.

---

## 8. Reproducing and extending

### 8.1 Running

```bash
cd eval/Eval_master

# All three, one checkpoint
./run_all.sh model.id=/path/to/final_checkpoint run='[faitheval,halueval,harness]'

# Smoke test, no GPU, print commands only
./run_all.sh model.dtype=float32 model.device_map=cpu num_samples=5 dry_run=true

# Seed sweep for HaluEval error bars (paired across arms)
./run_all.sh --multirun run='[halueval]' halueval.seed=42,43,44,45,46
```

Artifacts land under `${hydra:runtime.output_dir}` in per-benchmark subdirectories, so
timestamped and `--multirun` runs isolate automatically. `continue_on_error: true` keeps a run
going past a failure, but the **process still exits non-zero** — a run whose benchmarks all
died must not read as a complete evaluation to `run_all.sh`, SLURM, or CI
(`run_benchmarks.py:359-363`).

### 8.2 Re-scoring without a GPU

```bash
python HaluEval-reproduce/evaluation/score_results.py \
    outputs/eval/v3/*/*/halueval/*_results.json --json /tmp/halueval_scores.json
```

The decontaminated HaluEval summarization variant (§4.9) is re-scored the same way, into a separate
root, so the original one is only read:

```bash
./analysis/run_decontam.sh --root outputs/eval/v3 --out-root outputs/eval/v3_modified
```

The FaithEval counterfactual exact-vs-contains join reads
`*/faitheval/counterfactual_predictions.jsonl` and maps each row by `index` into
`FaithEval-reproduce/data/faitheval/FaithEval-counterfactual-v1.0/test.jsonl`, applying
`normalize_answer` to both sides. Note the predictions file stores only
`index` / `question` / `prediction` / `correct` — **the gold answer must be joined back in
from the dataset.**

### 8.3 Extension points

| To add | Change | Code change needed |
| --- | --- | --- |
| a FaithEval task or prompt variant | new `configs/<task>.yaml` + `SUPPORTED_TASKS` entry | none to generation/scoring |
| a HaluEval judge instruction | edit `evaluation/<task>/<task>_evaluation_instruction.txt` | none |
| an lm_eval benchmark | add its name to `harness.tasks` (globs allowed) | none — validated against the installed lm_eval |
| a new alternative HaluEval scoring | new parser in `score_results.py` | offline, no re-generation |

`src/run_eval.py --list-tasks truthfulqa` prints what the installed lm_eval knows.

### 8.4 The RAG-Truth benchmark: split and batching

RAG-Truth is outside this document's three benchmarks, but it runs through the same launcher,
and four of its settings matter for any comparison:

- **Split.** `conf/ragtruth/default.yaml` defaults to `split: test`, the 450 release-test
  sources. `split=all` evaluates all 2,965, including every source a RAG-Truth-trained model saw
  (`SP-DPO-Base/KNOWN_ISSUES.md` §5), and the CLI warns when it is used.
- **Batching.** Stage 1 (generation, greedy) and Stage 2 (the 13B detector) both batch. Each
  left-pads, sorts longest first and writes in dataset order. Batched greedy generation reproduces
  batch 1 exactly on the test model; I2 applies to the batch sizes.
- **Seeding.** The detector samples, with the authors' decoding (T = 0.05, top_p 0.95, top_k 40),
  so it is seeded per run directory. Its outputs depend on the batch size and the seed. Both are
  recorded in `summary.json`, with the split and `elapsed_s`.
- **Shared detection.** For many checkpoints, `ragtruth.stage=generate` runs per checkpoint. Then
  `analysis/run_ragtruth_detect.sh` loads the detector once for all of them, skipping every run
  directory that already has a `summary.json`.

---

## References

[1] Ming et al. (2024). *FaithEval: Can Your Language Model Stay Faithful to Context, Even If
"The Moon Is Made of Marshmallows".* Salesforce AI Research.
[2] Li, Cheng, Zhao, Nie & Wen (2023). *HaluEval: A Large-Scale Hallucination Evaluation
Benchmark for Large Language Models.* EMNLP.
[3] Lin, Hilton & Evans (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL.
[4] Gao et al. *A framework for few-shot language model evaluation* (lm-evaluation-harness).
EleutherAI.
[5] Lai et al. (2023). *Okapi: Instruction-tuned Large Language Models in Multiple Languages
with Reinforcement Learning from Human Feedback.* (multilingual TruthfulQA)
[6] Calvo Figueras et al. (2025). *TruthfulQA-Multi.* HiTZ.
[7] Rajpurkar et al. (2016). *SQuAD.* EMNLP. (the answer-normalization convention of §3.4)

**Companion documents.** `analysis/analysis_theory.md` (estimands, regimes and inference over
seeds); `analysis/format_confound.md` (the empirical investigation §7 formalizes);
`FaithEval-reproduce/ARCHITECTURE.md`, `HaluEval-reproduce/ARCHITECTURE.md` and
`harness-eval/ARCHITECTURE.md` (module maps); `HaluEval-reproduce/evaluation/REPRODUCIBILITY.md`
(the label-sampling analysis of §4.3); `README-runner.md` (the launcher's config surface).
