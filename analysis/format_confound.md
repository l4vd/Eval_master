# The Format Confound: Why the v3 Arm Table Measures Output Shape, Not Faithfulness

This document records the investigation into an apparent paradox in the `outputs/eval/v3`
ensemble results: every trained arm appears to *collapse* on HaluEval (0.21–0.50 vs. a base
of ~0.50) and on FaithEval `counterfactual` (0.001–0.013 vs. a base of 0.052), while
simultaneously *improving* on FaithEval `unanswerable` (0.26 → 0.52) and TruthfulQA
(mc2 0.418 → ~0.46).

**Finding: this is one artefact, not four results.** Three of those four metrics are scored
by parsers sensitive to answer *length and shape*. Instruction tuning made the models
verbose. The benchmarks that punish verbosity went down, the benchmark that rewards it went
up, and the one benchmark immune to it (TruthfulQA, log-likelihood ranking) moved barely at
all. No arm demonstrates a change in hallucination-detection ability.

Companion documents: `analysis_theory.md` (what the aggregation layer estimates) and
`../HaluEval-reproduce/evaluation/score_results.py` (the strict/lenient parsers used here).

---

## 1. The claim, stated precisely

For HaluEval, over all 132 result files in `outputs/eval/v3` (10 arms × seeds × 3 tasks,
1 311 552 scored rows):

| Quantity | Value |
|---|---|
| `corr(format_compliance, accuracy)` | **0.9974** |
| Accuracy **among rows that parsed** | **0.4981 ± 0.0165** (n = 132) |
| Range of that conditional accuracy | 0.389 – 0.577 |
| Reported accuracy range | 0.020 – 0.527 |
| Compliance range | 0.035 – 1.000 |
| Rows discarded as unparseable | 235 272 / 1 311 552 (**17.9 %**) |

Read the second row carefully. **On every arm, every seed and every task, the models are at
chance on the questions the scorer could actually read.** The 25-point spread in the
headline table is the compliance column and nothing else; reported accuracy is, to three
significant figures, `compliance × 0.498`.

A metric that is a deterministic multiple of "did the output parse" is not measuring
hallucination.

## 2. Mechanism (HaluEval)

`parse_strict` in `HaluEval-reproduce/evaluation/score_results.py` reproduces the published
rule: strip ".", then require exactly one of the capitalised substrings `Yes` / `No`.
Answers containing **both or neither** are unparseable, and — this is the load-bearing part
— an unparseable row is scored **wrong**, not excluded.

The base model emits a bare verdict; the trained models write essays:

| Run | median generation | compliance | reported acc |
|---|---|---|---|
| `off-sp-orpo` / seed_1337 / dialogue | **2 chars** (`"No"`) | 1.000 | 0.499 |
| `dpo` / seed_42 / qa | **2 chars** | 1.000 | 0.504 |
| `orpo_asc` / seed_7 / dialogue | 243 chars | 0.329 | 0.163 |
| `off-sp-dpo_asc` / seed_42 / summarization | 397 chars | 0.035 | 0.020 |

In the worst run, **99 % of failures are prose longer than 50 characters** and 94 % contain
*neither* bare token. A representative discarded answer:

> "The summary does **not** contain any factual contradictions based on the given documents.
> […] Therefore, there is **no** need to determine if the sum"

That answer is *semantically correct* (`ground_truth = No`) and is scored wrong twice over:
the strict rule has no word boundaries, so `No` matches inside "not" / "no need", while the
model never emits a standalone verdict. It is also truncated mid-sentence — see §5.

### 2.1 The base model's 0.50 is degenerate, not good

The natural misreading is "base 0.507 beats dpo 0.460, so training hurt." Both numbers are
chance. Base sits at ~100 % compliance × ~50 % accuracy: it emits a parseable verdict every
time and is right at the label base rate. `score_results.py` already carries a
`!! degenerate judge` warning for exactly this shape, and several runs trip the
`judged_yes_rate ∈ {0, 1}` condition — e.g. `off-sp-orpo`/seed_1337 answers `No` to all
10 000 dialogue items for an "accuracy" of 0.499.

**HaluEval as currently scored cannot distinguish a perfect hallucination detector from a
model that always says `No`.** Reporting base and trained arms on this axis compares two
constants.

## 3. The same confound, opposite sign (FaithEval)

FaithEval routes tasks through two scorers in
`FaithEval-reproduce/src/faitheval/metrics.py`, and they have *opposite* length biases.

### 3.1 `counterfactual` → `answer_match` (exact equality) — punishes verbosity

`answer_match` requires normalised prediction == normalised reference. The prompt
(`prompting.py:BASE_INSTRUCTION`) says *"respond with the exact answer only. Do not be
verbose."* The trained models emit 476–920 character essays. Joining predictions against
`FaithEval-counterfactual-v1.0/test.jsonl` gold answers:

| Arm | seeds | exact (reported) | gold string **contained** | median len |
|---|---|---|---|---|
| `dpo` | 4 | 0.004 | **0.279** | 572 |
| `sft` | 4 | 0.002 | **0.279** | 560 |
| `orpo` | 5 | 0.001 | **0.289** | 594 |
| `orpo-desc` | 5 | 0.001 | **0.271** | 551 |
| `orpo-asc` | 3 | 0.001 | **0.280** | 612 |
| `off-sp-dpo-asc` | 5 | 0.001 | **0.246** | 584 |
| `off-sp-dpo-desc` | 5 | 0.013 | **0.284** | 920 |
| `off-sp-dpo-shuf` / `orpo-shuf` | 5 | 0.010 | **0.254** | 476 |
| **Overall** | 45 files | **0.005** | **0.271** | 596 |

The exact-match column reproduces the published arm table. **~27 % of predictions contain
the correct gold answer and score zero.** The base model's 0.052 is not superior knowledge:
an untuned LM continues `Answer:` with a short span, which is the only thing exact match can
credit.

### 3.2 `unanswerable` / `inconsistent` → `phrase_match` (substring) — rewards verbosity

`phrase_match` is a substring test whose own docstring flags it as *"biased upward"*:
`"not"` matches inside *notable*, *nothing*, *cannot*, *another*. Longer answers hit a valid
phrase by chance more often. This is the one FaithEval task where the trained arms "improve"
(0.258 → 0.520) — and it is scored by the one rule that pays for extra tokens.

### 3.3 The control: TruthfulQA

`truthfulqa_mc1/mc2` is log-likelihood ranking over fixed options. **No generation, no
parser, no length sensitivity.** It is the only benchmark in the suite where the format
confound cannot operate, and it is the only one that stays nearly flat (mc1 0.269 → ~0.28,
mc2 0.418 → ~0.46 — a small genuine gain, uniform across arms and orderings).

The signature is decisive:

| Scorer | Length bias | Observed effect of training |
|---|---|---|
| HaluEval `parse_strict` | punishes | large apparent **drop** |
| FaithEval `answer_match` | punishes | large apparent **drop** |
| FaithEval `phrase_match` | rewards | large apparent **gain** |
| TruthfulQA loglik | none | small, uniform gain |

Effects appear exactly where the parser is length-sensitive and vanish where it is not.
That is a property of the scorers, not of the arms.

## 4. What this does and does not invalidate

**Does not survive:** any claim ranking arms by HaluEval accuracy, or by FaithEval
`counterfactual` / `unanswerable`, or any base-vs-trained comparison on those axes. In
particular the curriculum-ordering contrasts (`asc` / `desc` / `shuf`) on HaluEval are
compliance contrasts — the `asc` arms are simply the most verbose (mean compliance 0.55–0.57
vs. 0.95–0.98 for `dpo` / `sft` / `off-sp-orpo`).

**Survives:** TruthfulQA mc1/mc2, and the aggregation/CI machinery itself
(`analysis_theory.md` is unaffected — it correctly aggregated the numbers it was given).

**The honest negative result:** after 2 epochs, no arm exceeds chance on HaluEval binary
judgement *even on the rows it answers in the required format* (0.498 ± 0.017). That is a
real, reportable finding about the task, and it is stronger than the artefactual ranking it
replaces, because it is not a formatting measurement.

## 5. Recommended remediation

1. **Do not report raw HaluEval accuracy.** Report `format_compliance` and conditional
   accuracy (correct / parsed) as two separate columns. They decompose the metric into the
   part that is formatting and the part that is judgement; only the latter is the claim.
2. **Constrain the decode** rather than loosening the parser. Score HaluEval by comparing
   the logprobs of the `Yes` / `No` continuations, or restrict sampling to those tokens.
   Compliance then becomes 100 % by construction and the metric measures discrimination.
   This is preferable to `parse_lenient`, which merely re-labels the artefact: it recovers
   only part of the gap and remains non-comparable to published numbers.
3. **Report `counterfactual` under both rules** — exact match for comparability with the
   paper, containment as a clearly-caveated secondary — since exact match against a chat
   model measures verbosity.
4. **Check `max_new_tokens`.** Many generations terminate mid-sentence (see §2), so a
   verdict that would have arrived after the cutoff is lost outright. Some share of the
   17.9 % unparseable rows is truncation, not refusal.
5. Treat `phrase_match` results as an upper bound, per its own docstring, and prefer
   `--strict-match` where the absolute level matters.

**Status.** Items 2 and 3 are implemented as opt-in modified protocols. They are reported beside
the original numbers, never instead of them (`EVALUATION_THEORY.md` §4.8, §3.8).

- **Item 2:** `halueval.scoring=constrained` compares logP("Yes") with logP("No") in one prefill.
  It reports AUROC with its SE, the argmax accuracy and the verdict mass. There is no parser, so
  compliance is 100 % by construction.
- **Item 3:** the parser-free counterfactual score is multiple choice
  (`faitheval.tasks=[counterfactual_mc]`). It scores the log-likelihood of each answer option and
  reports `accuracy` and `accuracy_norm`. Containment as a caveated secondary is still proposed
  (`SP-DPO-Base/FUTURE_EXTENSIONS.md` §6).
- **Item 1** was already covered by `format_compliance` and the `acc_on_parsed` column of
  `EXPERIMENT_PROCEDURE.md` §5.6.
- **The training-data overlap** of `halueval/summarization` is a separate problem, handled by
  `halueval.decontam` (`SP-DPO-Base/KNOWN_ISSUES.md` §5).

The constrained and multiple-choice scores need a GPU and have not been run on `v3`, so §1–§4
stand as measured.

## 6. Reproducing this analysis

All numbers above come from stored generations; no model or GPU is required.

```bash
cd eval/Eval_master

# Per-file compliance, TPR/TNR, yes-rate, strict vs lenient (§1, §2)
python HaluEval-reproduce/evaluation/score_results.py \
    outputs/eval/v3/*/*/halueval/*_results.json --json /tmp/halueval_scores.json
```

The two headline checks are one line each, over the files that command scores:

* `corr(format_compliance, accuracy) = 0.9974`
* `num_correct / (num_examples - num_failed) = 0.4981 ± 0.0165`, over all 132 files.

The counterfactual exact-vs-contains join (§3.1) reads
`outputs/eval/v3/*/*/faitheval/counterfactual_predictions.jsonl`, maps each row by its
`index` field into
`FaithEval-reproduce/data/faitheval/FaithEval-counterfactual-v1.0/test.jsonl`, and applies
`faitheval.metrics.normalize_answer` to both sides before comparing with `==` (exact) and
`in` (contains). Note that the predictions file stores only `index` / `question` /
`prediction` / `correct` — the gold answer must be joined back in from the dataset.

---

*Scope note: arm labels above are the on-disk run-dir names in `outputs/eval/v3`
(`dpo_ensemble-2epochs` etc.), mapped to the published table's short names (`dpo`, `sft`,
`off-sp-dpo-asc`, …). The per-arm exact-match column in §3.1 reproduces the published
`counterfactual` figures, which confirms the mapping. The `base` arm is not present in the
local `v3` tree: its compliance is inferred from its reported accuracy via the ratio in §1
(implying ~1.00 for all three HaluEval tasks) rather than measured directly.*
