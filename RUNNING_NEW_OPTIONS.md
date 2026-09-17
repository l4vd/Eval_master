# Running the new evaluation options for a model

This page shows how to run every option added by the train↔eval overlap and modified-protocol
work, for one model or for a trained checkpoint ensemble. It is a runbook: what each option means
and why it exists is in the linked references.

Run everything from `Eval_master/` (on Windows, in Git Bash). Selecting benchmarks works exactly
as before (`run=[...]`, `--benchmarks a,b`); the options below are switches inside a benchmark.

## Contents

- [The options at a glance](#the-options-at-a-glance)
- [Three rules](#three-rules)
- [0. Set up](#0-set-up)
- [1. One model](#1-one-model)
- [2. A trained checkpoint ensemble](#2-a-trained-checkpoint-ensemble)
- [3. Analyse](#3-analyse)
- [4. Overlap checker](#4-overlap-checker)
- [5. Files each option writes](#5-files-each-option-writes)
- [6. Checks after a run](#6-checks-after-a-run)
- [7. Optional: all variants in one run (overview)](#7-optional-all-variants-in-one-run-overview)
- [Option reference](#option-reference)

---

## The options at a glance

| Option | Switch | Protocol | Cost | Reference |
|---|---|---|---|---|
| HaluEval constrained Yes/No | `halueval.scoring=constrained` | modified | one prefill per row, cheaper than the original run | [conf/halueval](conf/halueval/README.md#modified-protocols-opt-in), [theory §4.8](EVALUATION_THEORY.md) |
| HaluEval decontamination, live | `halueval.decontam.enabled=true` | modified | none (reuses the run's rows) | [conf/halueval](conf/halueval/README.md#modified-protocols-opt-in), theory §4.9 |
| HaluEval decontamination, finished runs | `analysis/run_decontam.sh` or `score_results.py --exclude` | modified | CPU, no model | [analysis README](analysis/README.md#traineval-overlap-and-decontamination) |
| FaithEval counterfactual as multiple choice | `faitheval.tasks=[counterfactual_mc]` | modified | minutes per model | [conf/faitheval](conf/faitheval/README.md#multiple-choice-counterfactual_mc-opt-in), theory §3.8 |
| RAG-Truth on the held-out split | `ragtruth.split=test` (now the default) | original | — | [conf/ragtruth](conf/ragtruth/README.md#why-split-test) |
| RAG-Truth batching and seeding | `ragtruth.batch_size`, `ragtruth.detector.batch_size`, `ragtruth.detector.seed` | original | faster | [conf/ragtruth](conf/ragtruth/README.md) |
| RAG-Truth generate now, detect later | `ragtruth.stage=generate` + `analysis/run_ragtruth_detect.sh` | original | one detector load for all models | [conf/ragtruth](conf/ragtruth/README.md#many-checkpoints-generate-then-detect-once) |
| Analysis split by protocol | `analysis.cli --protocol`, repeated `--arm` | — | — | [analysis README](analysis/README.md#original-and-modified-protocols) |
| Train↔eval overlap checker | `analysis/run_overlap.sh` | — | CPU | [analysis README](analysis/README.md#traineval-overlap-and-decontamination) |

With no switch set, every run, command line and output file is exactly what it was before.

## Three rules

1. **Modified protocols go to their own root, `$EVAL_ROOT_MOD`, never `$EVAL_ROOT`.**
   `--resume`, the §5.5 census and the format check treat every `*summary*.json` under
   `$EVAL_ROOT` as a finished original run.
2. **The modified root holds only modified outputs.** The analysis merges an arm's two roots, and
   it stops with `two of its run dirs report <benchmark>/<task>/<metric> for seed <N>` if both
   roots contain the same original metric. So in `$EVAL_ROOT_MOD`:
   - use `halueval.scoring=constrained`, not `both` (`both` also writes the original summary);
   - list only `counterfactual_mc` in `faitheval.tasks`, not the original tasks;
   - run only `halueval,faitheval` there.

   `both` is for a standalone scratch root that is never merged with an original one.
3. **Keep every setting identical across the models you compare.** That covers `model.dtype`, the
   batch sizes, `halueval.scoring`, the decontamination toggle and its list, and the detector seed.
   Test runs with `num_samples` go to a scratch directory: `--resume` does not check how many
   samples a finished summary used.

---

## 0. Set up

Paths follow `SP-DPO-Base/EXPERIMENT_PROCEDURE.md` §5.2. Adapt the first two lines.

```bash
cd Eval_master

export MODEL=/path/to/model_or_checkpoint          # Hub id, local checkpoint or LoRA adapter
export NAME=my_model                               # a label: the directory name under each root
export BASE_MODEL=/gpfs/project/$USER/models/huggingface/hub/Qwen/Qwen2.5-0.5B-Instruct
export EVAL_ROOT=outputs/eval/ragtruth_study
export EVAL_ROOT_MOD=outputs/eval/ragtruth_study_modified
export EVAL_FIXED="model.dtype=bfloat16 harness.batch_size=8"
export EVAL_MODIFIED="halueval.scoring=constrained halueval.decontam.enabled=true faitheval.tasks=[counterfactual_mc]"
mkdir -p "$EVAL_ROOT" "$EVAL_ROOT_MOD"
```

**Look before you run.** `dry_run=true` prints every benchmark command without loading a model:

```bash
./run_all.sh model.id="$MODEL" output_dir=/tmp/dry run='[halueval,faitheval]' \
    $EVAL_FIXED $EVAL_MODIFIED dry_run=true
```

In the printed HaluEval commands, look for `--scoring constrained` and, on `summarization` only,
`--exclude-list …/halueval__summarization.json`.

**Smoke test** (scratch root, a few rows, CPU is fine for a tiny model):

```bash
./run_all.sh model.id="$MODEL" output_dir=/tmp/smoke hydra.run.dir=/tmp/smoke \
    run='[halueval,faitheval]' num_samples=20 $EVAL_MODIFIED
```

---

## 1. One model

The layout is `<root>/<NAME>/run_0/`, the same as `base/run_0`. The finished-run tools below rely
on it.

### 1.1 Original protocols (unchanged)

```bash
./run_all.sh model.id="$MODEL" \
    output_dir="$EVAL_ROOT/$NAME/run_0" hydra.run.dir="$EVAL_ROOT/$NAME/run_0" \
    run='[faitheval,halueval,harness]' $EVAL_FIXED
```

### 1.2 Constrained HaluEval and FaithEval multiple choice

Both modified scorers in one launch, into the modified root. `$EVAL_MODIFIED` carries
`halueval.decontam.enabled=true`, so the run also writes the decontaminated constrained summary for
`summarization`, at no extra cost.

```bash
./run_all.sh model.id="$MODEL" \
    output_dir="$EVAL_ROOT_MOD/$NAME/run_0" hydra.run.dir="$EVAL_ROOT_MOD/$NAME/run_0" \
    run='[halueval,faitheval]' $EVAL_FIXED $EVAL_MODIFIED
```

One of them only:

```bash
# constrained HaluEval only
./run_all.sh model.id="$MODEL" output_dir="$EVAL_ROOT_MOD/$NAME/run_0" hydra.run.dir="$EVAL_ROOT_MOD/$NAME/run_0" \
    run='[halueval]' $EVAL_FIXED halueval.scoring=constrained

# FaithEval multiple choice only
./run_all.sh model.id="$MODEL" output_dir="$EVAL_ROOT_MOD/$NAME/run_0" hydra.run.dir="$EVAL_ROOT_MOD/$NAME/run_0" \
    run='[faitheval]' $EVAL_FIXED 'faitheval.tasks=[counterfactual_mc]'
```

A subset of HaluEval tasks: `'halueval.tasks=[summarization]'`. Only `summarization` has an
exclusion list; the other tasks run and print
`== halueval.decontam: no exclusion list for '<task>'; original numbers only.`

### 1.3 Decontaminated HaluEval from the finished original run

No model and no GPU. It reads the original `summarization` results and writes the
`halueval.decontam` summary into the modified root.

```bash
mkdir -p "$EVAL_ROOT_MOD/$NAME/run_0/halueval"
.venv/bin/python HaluEval-reproduce/evaluation/score_results.py \
    "$EVAL_ROOT/$NAME/run_0/halueval/"summarization_*_results.json \
    --exclude decontamination/ragtruth/halueval__summarization.json \
    --emit-summary "$EVAL_ROOT_MOD/$NAME/run_0/halueval"
[[ -f "$EVAL_ROOT/$NAME/run_0/run_metadata.json" ]] && \
    cp -n "$EVAL_ROOT/$NAME/run_0/run_metadata.json" "$EVAL_ROOT_MOD/$NAME/run_0/"
```

- `score_results.py` is stdlib-only, so any Python 3 works (on Windows: `.venv/Scripts/python.exe`).
- `--emit-summary` refuses the results' own directory.
- Without `--emit-summary` it only prints the tables, which is useful for a quick look.
- For every model under a root at once, use [`run_decontam.sh`](#23-decontaminated-halueval-for-the-whole-root).

### 1.4 RAG-Truth

For a single model, run both stages in one go. The default `split` is now `test` and the batch
sizes and seed come from the conf. It needs the 13B detector `llama2_13b_lora_0217` in the HF cache.

```bash
./run_all.sh model.id="$MODEL" \
    output_dir="$EVAL_ROOT/$NAME/run_0" hydra.run.dir="$EVAL_ROOT/$NAME/run_0" \
    run='[ragtruth]' ragtruth.stage=all $EVAL_FIXED
```

RAG-Truth is an original protocol, so it goes to `$EVAL_ROOT`. For many models, generate first and
detect once ([§2.4](#24-rag-truth-generate-per-checkpoint-then-detect-once)).

---

## 2. A trained checkpoint ensemble

`run_eval_checkpoints.sh` evaluates every `seed_*` checkpoint of a training group into
`<out>/seed_<N>/`. Hydra overrides go after `--launcher-extra`, which must come last.

```bash
export ARM=off_sp_dpo_asc
export GROUP=/path/to/outputs/<date>/<time>_<METHOD>_ensemble
```

Add `--dry-run` to any command below to print the plan without loading a model.

### 2.1 Base model, once per root

```bash
# original
./run_all.sh model.id="$BASE_MODEL" output_dir="$EVAL_ROOT/base/run_0" hydra.run.dir="$EVAL_ROOT/base/run_0" \
    run='[faitheval,halueval,harness]' $EVAL_FIXED
# modified
./run_all.sh model.id="$BASE_MODEL" output_dir="$EVAL_ROOT_MOD/base/run_0" hydra.run.dir="$EVAL_ROOT_MOD/base/run_0" \
    run='[halueval,faitheval]' $EVAL_FIXED $EVAL_MODIFIED
```

### 2.2 Modified protocols for every seed

```bash
./analysis/run_eval_checkpoints.sh --checkpoints "$GROUP/seed_*" --out "$EVAL_ROOT_MOD/$ARM" \
    --benchmarks halueval,faitheval --resume --launcher-extra $EVAL_FIXED $EVAL_MODIFIED
```

On the cluster, submit one job per seed (`--checkpoints "$GROUP/seed_$S"`), as in
EXPERIMENT_PROCEDURE §5.4 step 3.

`--resume` knows the variants. It reads the job's own overrides to decide which summary counts as
finished:

| Override | Finished when this exists |
|---|---|
| `halueval.scoring=constrained` | `*_constrained_summary.json` |
| `halueval.scoring=both` | the original and the constrained summary |
| `faitheval.tasks=[…counterfactual_mc…]` | `counterfactual_mc_summary.json` |
| `ragtruth.stage=generate` | `generation_summary.json` |

### 2.3 Decontaminated HaluEval for the whole root

Once the original HaluEval runs in `$EVAL_ROOT` have finished:

```bash
./analysis/run_decontam.sh --root "$EVAL_ROOT" --out-root "$EVAL_ROOT_MOD"
```

- It scores every `$EVAL_ROOT/<arm>/<run>/halueval/summarization_*_results.json`.
- It writes into the same `<arm>/<run>/halueval/` path under `$EVAL_ROOT_MOD` and copies
  `run_metadata.json`.
- It refuses `--out-root` equal to `--root`.
- `--list FILE.json` (repeatable) picks lists; the default is every
  `decontamination/ragtruth/halueval__*.json`.

The decontaminated **constrained** variant cannot come from here: its results live in
`$EVAL_ROOT_MOD`, which this script may not write into. The live runs of §2.1 and §2.2 write it,
because `$EVAL_MODIFIED` carries `halueval.decontam.enabled=true`. `--resume` does not check for
it: a seed run without that override keeps no constrained decontaminated summary until its
`halueval/` directory is deleted and the seed is re-run.

### 2.4 RAG-Truth: generate per checkpoint, then detect once

```bash
# Stage 1, base and every seed (small model, batched)
./run_all.sh model.id="$BASE_MODEL" output_dir="$EVAL_ROOT/base/run_0" hydra.run.dir="$EVAL_ROOT/base/run_0" \
    run='[ragtruth]' ragtruth.split=test ragtruth.stage=generate $EVAL_FIXED
./analysis/run_eval_checkpoints.sh --checkpoints "$GROUP/seed_*" --out "$EVAL_ROOT/$ARM" \
    --benchmarks ragtruth --resume --launcher-extra $EVAL_FIXED ragtruth.split=test ragtruth.stage=generate

# Stage 2, ONE job over the whole root (13B detector loaded once)
./analysis/run_ragtruth_detect.sh --root "$EVAL_ROOT" --dtype bfloat16 --dry-run   # list pending dirs
./analysis/run_ragtruth_detect.sh --root "$EVAL_ROOT" --dtype bfloat16
```

- **What the detection job picks up:** `<root>/*/seed_*/ragtruth/` and `<root>/base/run_*/ragtruth/`
  that have `generations.jsonl` but no `summary.json`.
- **Resuming:** re-submit the same command after a walltime kill.
- **Layouts it cannot see:** a single-model `<NAME>/run_0` dir. Use `stage=all` (§1.4) for those.
- **`--dtype`:** must equal `model.dtype`.
- **Other flags:** `--detector-model-id`, `--detector-batch-size` and `--detector-seed` override
  the conf; `--extra …` (last) forwards raw flags.
- **Pilot first:** do base plus one seed, read `elapsed_s` in `generation_summary.json` and
  `summary.json`, then project the cost of the full sweep.

---

## 3. Analyse

### 3.1 Merge each arm's two roots

Repeat `--arm` with the same name, once per root. The original tree keeps today's layout; the
modified protocols go to `<out>/modified/`.

```bash
./analysis/run_analysis.sh \
    --arm "$ARM=$EVAL_ROOT/$ARM"             --arm "$ARM=$EVAL_ROOT_MOD/$ARM" \
    --arm "base=$EVAL_ROOT/base/run_0"       --arm "base=$EVAL_ROOT_MOD/base/run_0" \
    --reference base --benchmarks faitheval,halueval \
    --out outputs/analysis/$ARM
```

For one model, point `--arm` at `$EVAL_ROOT/$NAME/run_0` and `$EVAL_ROOT_MOD/$NAME/run_0`. With a
single run on each side the comparison is a difference, with no significance test.

### 3.2 Choose what is written

| Flag | Effect |
|---|---|
| `--protocol both` (default) | original tree in `--out/`, modified tree in `--out/modified/` (only if modified records exist) |
| `--protocol original` | original tree only, as before the change |
| `--protocol modified` | `--out/modified/` only |
| `--benchmarks halueval` | HaluEval **and** all its variants |
| `--benchmarks halueval.constrained` | that variant only |
| `--exclude faitheval.mc` | drops one variant (same matching) |

RAG-Truth gets its own call when it is not in the benchmark list you use elsewhere:
`--benchmarks ragtruth --out outputs/analysis/$ARM/ragtruth` with the `$EVAL_ROOT` arms only.

### 3.3 How modified results are named

| Benchmark name | Comes from | Primary metric |
|---|---|---|
| `halueval.constrained` | `*_constrained_summary.json` | `auroc` (SE as `stderr`) |
| `halueval.decontam` | `*_decontam_summary.json` | `accuracy` on the kept rows |
| `halueval.constrained_decontam` | `*_constrained_decontam_summary.json` | `auroc` |
| `faitheval.mc` | `counterfactual_mc_summary.json` | `accuracy` |

Figures for variants are named with `__` (`halueval__constrained.png`) and titled
`[modified protocol]`. Modified comparisons form their own multiplicity family and are context
only.

---

## 4. Overlap checker

Only needed when the training data changes or a corpus is added. The committed lists in
[`decontamination/ragtruth/`](decontamination/ragtruth/) already cover the current study.

```bash
# report only: outputs/overlap/ragtruth/overlap_report.{json,md}
./analysis/run_overlap.sh --train ragtruth="$TRAIN_ROOT/data/ragtruth" \
    --harness-samples "$EVAL_ROOT/base/run_0/harness/samples.jsonl"

# a new corpus, before training on it
./analysis/run_overlap.sh --train mycorpus='jsonl:data/mycorpus/train.jsonl:context,chosen,rejected:source_id'
```

- **Default training data:** `--train` defaults to `SP-DPO-Base/data/ragtruth` two levels above
  `Eval_master`. That matches the local checkout; on the cluster, where the two repos are siblings,
  pass `--train` explicitly as above.
- **Which data to point at:** the same `data/ragtruth` the study trained on. Compare its
  `pairing_report.json` sha256 with the HPC copy.
- **`--harness-samples`:** checks all 817 harness TruthfulQA items. Without it the CSV (790 items)
  is used.
- **`--write-exclusions`:** overwrites the committed lists and changes their sha256, which
  `EVAL_BLOCK.txt` records. Only use it deliberately, before any evaluation that reads the lists.
- **Leave the matching defaults alone** (`--ngram 13 --near-dup 0.5 --df-max 10
  --exposure-topk 3`): `KNOWN_ISSUES.md` §5 was measured with them.

---

## 5. Files each option writes

`<label>` is the model path with `/` and `\` replaced by `_`.

| Option | Directory | Files |
|---|---|---|
| `halueval.scoring=constrained` | `<run>/halueval/` | `<task>_<label>_constrained_results.json`, `<task>_<label>_constrained_summary.json` |
| `halueval.scoring=both` | `<run>/halueval/` | the above plus the original `<task>_<label>_{results,summary}.json` |
| `halueval.decontam.enabled=true` | `<run>/halueval/` | `summarization_<label>[_constrained]_decontam_summary.json` |
| `score_results.py --emit-summary` / `run_decontam.sh` | the `--emit-summary` / `--out-root` dir | `summarization_<label>_decontam_summary.json` (+ `run_metadata.json`) |
| `faitheval.tasks=[counterfactual_mc]` | `<run>/faitheval/` | `counterfactual_mc_predictions.jsonl`, `counterfactual_mc_summary.json` |
| `ragtruth.stage=generate` | `<run>/ragtruth/` | `generations.jsonl`, `generation_summary.json` |
| detection (`stage=all` or the shared job) | `<run>/ragtruth/` | `detections.jsonl`, `summary.json` |
| `run_overlap.sh` | `outputs/overlap/<train>/` | `overlap_report.{json,md}` |
| `run_overlap.sh --write-exclusions` | `decontamination/<train>/` | `<benchmark>__<task>.json` + a copy of the report |

---

## 6. Checks after a run

| Check | Where | What you want |
|---|---|---|
| Constrained verdicts are real, not forced | `*_constrained_summary.json` | `mean_verdict_mass` well above 0.5, `frac_mass_below_half` small. Low mass means Yes and No are both unlikely, and the AUROC reads tail noise |
| MC not driven by option length | `counterfactual_mc_summary.json` | `accuracy` and `accuracy_norm` rank the models the same way |
| Decontamination saw the full dataset | `*_decontam_summary.json` | row counts 9,497 kept / 503 seen / 9,367 clean / 130 unseen-exposed. A `num_samples` run prints `INCOMPLETE run` |
| RAG-Truth ran on the held-out split | `generation_summary.json`, `summary.json` | `split: test`; batch sizes and seed as configured; read `elapsed_s` |
| No accidental `split=all` | run log | no warning containing `generates for all 2,965 release sources` |
| One value per setting across models | EXPERIMENT_PROCEDURE §5.5 census | run it once per root; modified runs are keyed by `scoring` / `variant` |
| Inherited decoding config | `python -c "from transformers import GenerationConfig as G; print(G.from_pretrained('$BASE_MODEL'))"` | if `repetition_penalty` ≠ 1.0, `FUTURE_EXTENSIONS.md` §7 applies to the generation-parsed benchmarks. Constrained and MC scoring read raw logits and are unaffected |

**CUDA out of memory?** Lower `halueval.batch_size`, `faitheval.batch_size`, `ragtruth.batch_size`
or `ragtruth.detector.batch_size`, and use the new value for **every** model you compare. All are
recorded in the summaries.

---

## 7. Optional: all variants in one run (overview)

**Descriptive, not pre-registered.** This is for looking at the results. It is not part of the
study procedure. The three rules above stay the official ones, EXPERIMENT_PROCEDURE §5.4–§5.8 is
unchanged, and the deciding metric stays harness MC2. Every output of this mode is labelled
"overview — descriptive, not pre-registered".

**What it does.**

1. One launch computes every variant of a checkpoint.
2. Each variant is written to its own **sibling dir**, `<run>/<bench>.<variant>/`, named like its
   analysis benchmark. The original dirs keep exactly their content. The procedure's census,
   format check, decontam check and `run_decontam.sh` glob `*/*/<bench>/…`, so they never see
   the sibling dirs. That makes this the one allowed exception to rule 1.
3. Nothing already computed is computed again. Resume works per **unit** (one variant × one
   task), not per seed.
4. `run_analysis.sh --variants-overview` plots how the arms compare under every variant.

If you want `$EVAL_ROOT` untouched, point the overview at a copy. `analysis.adopt` and
`analysis.derive` work on any root.

| Variant dir | Kind | Produced by |
|---|---|---|
| `halueval`, `faitheval`, `harness` | GPU | the original runs (unchanged) |
| `halueval.constrained` | GPU (prefill) | `evaluate.py --scoring constrained`, or `both` with `--constrained-output-dir` |
| `faitheval.mc` | GPU (prefill) | `run_eval.py --task counterfactual_mc` |
| `halueval.parsed` | CPU | `score_results.py --emit-variant parsed`: strict parser, accuracy over parsed rows only |
| `halueval.lenient` | CPU | `score_results.py --emit-variant lenient`: word-boundary parser; `unavailable` without `raw_judgement` |
| `halueval.decontam`, `halueval.constrained_decontam` | CPU | `score_results.py --exclude` (summarization) |
| `faitheval.strict` | CPU | `rescore.py --rule strict` (`unknown` / `conflict` only) |
| `faitheval.wordmatch` | CPU | `rescore.py --rule wordmatch` (valid phrases at word boundaries) |
| `faitheval.contains` | CPU | `rescore.py --rule contains`: gold answer at word boundaries, per answer-length stratum, exact match alongside |

RAG-Truth and TruthfulQA have no variants; the overview only resumes them as a whole. The
`repetition_penalty` variant is not part of this.

`contains` uses word boundaries. Its counterfactual level on local v3 is 0.248, against 0.271
for the plain substring join in `analysis/format_confound.md` §3.1.

### 7.1 Existing roots: CPU variants only (no GPU)

```bash
./analysis/run_derive.sh --root "$EVAL_ROOT" --dry-run   # what would be derived
./analysis/run_derive.sh --root "$EVAL_ROOT"             # derive into sibling dirs
.venv/bin/python -m analysis.variants --root "$EVAL_ROOT"   # status table + recorded knobs
```

`derive` reads the originals and writes only sibling dirs. Re-running it derives nothing new,
unless a source file changed: every CPU summary records `source_sha256`. The FaithEval
re-scorer needs only PyYAML, and `--faitheval-python` picks another interpreter.

Optional: to reuse GPU results that already exist in a legacy modified root, copy them in with
`python -m analysis.adopt --from "$EVAL_ROOT_MOD" --into "$EVAL_ROOT" [--dry-run]`. This copies
only constrained HaluEval and `counterfactual_mc` files, and only into runs with the same
`model_id` and label. It never overwrites and never touches the source root.

### 7.2 One checkpoint, every variant

```bash
./run_all.sh model.id=/path/to/ckpt run='[faitheval,halueval,harness]' \
    variants='[all]' resume=true output_dir=outputs/overview/ckpt dry_run=true   # plan first
```

The status matrix (variant × task) prints first. The GPU work then depends on what is missing:

- **New checkpoint:** the original runs; HaluEval with `--scoring both` (one model load per
  task); one `counterfactual_mc` process; then all CPU derivations.
- **Checkpoint whose originals are done:** 3 constrained-only HaluEval processes and 1 MC
  process. A finished original is never re-decoded.

`variants` also takes a list, for example `variants='[halueval.constrained,faitheval.strict]'`.
A CPU variant pulls in only the source tasks it needs.

It refuses (exit 2):

- the legacy variant switches (`halueval.scoring`, `halueval.decontam.enabled`,
  `faitheval.strict_match`, `counterfactual_mc` in `faitheval.tasks`);
- GPU units whose recorded settings differ from this launch (`num_samples`, `seed`, batch
  size, …). Pass `recompute_stale=true` to re-run those units.

An old FaithEval summary does not record `num_samples`. Its row count is checked against the
dataset instead, and it is `unverified` where the dataset is absent (`strict_provenance=true`
makes that stale). Before a GPU unit runs, its old summary is deleted, so a killed run reads as
missing. Each launch appends to `<run>/launches.jsonl`. That entry records the unit statuses,
what ran, the overrides and the code version (`EVAL_MASTER_REV`, else `git rev-parse HEAD`).

### 7.3 A training day on the cluster: `qsub_overview_day.sh`

The overview counterpart of `qsub_eval_day.sh`. For one training day it writes:

- one GPU job per ensemble (`<DATE>/*/seed_*`). This is `qsub_eval_day.sh` with `VARIANTS` set;
- with `BASE_MODEL` set, one GPU job for the base model into `$OUT_ROOT/base/run_0`;
- one CPU job that derives the CPU variants over the **whole** `OUT_ROOT` and writes a status
  table next to the job scripts.

The derive job can run alongside the GPU jobs: a unit being recomputed has no summary, so its
derivations wait for it. Modes are `print` (write jobs, show `qsub` lines), `check` (run every
job here with `--dry-run`) and `submit`.

```bash
DAY=/gpfs/project/$USER/git-source/SP_DPO/SP-DPO-Base/outputs/2026-09-17
EXTRA="model.dtype=bfloat16 harness.batch_size=8"          # the same as the originals ran with
BASE_MODEL=/gpfs/project/$USER/models/huggingface/hub/Qwen/Qwen2.5-0.5B-Instruct \
EXTRA="$EXTRA" ./analysis/qsub_overview_day.sh "$DAY" check

QSUB_RES="-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=24:00:00 -A <project>" \
QSUB_RES_derive="-l select=1:ncpus=2:mem=16gb -l walltime=04:00:00 -A <project>" \
BASE_MODEL=... EXTRA="$EXTRA" ./analysis/qsub_overview_day.sh "$DAY" submit

# a root whose originals are done: CPU variants only, no GPU
DERIVE_ONLY=1 OUT_ROOT=outputs/eval/v4-2 QSUB_RES_derive="..." ./analysis/qsub_overview_day.sh - submit
```

| Env | Default | Meaning |
|---|---|---|
| `VARIANTS` | `all` | `all` or a comma list (`halueval.constrained,faitheval.mc`) |
| `OUT_ROOT`, `SUFFIX`, `BENCHES` | as `qsub_eval_day.sh` | where the ensembles land: `$OUT_ROOT/<METHOD>_ensemble$SUFFIX/` |
| `EXTRA` | — | Hydra overrides. Use the **same** ones the originals ran with, or every GPU unit reads as stale (exit 2). Legacy variant keys are refused. |
| `RECOMPUTE_STALE` | unset | `1` re-runs stale GPU units (`recompute_stale=true`) |
| `BASE_MODEL`, `BASE_DIR` | unset, `base/run_0` | adds the base-model job |
| `DERIVE`, `DERIVE_ONLY` | `1`, unset | `DERIVE=0` drops the CPU job; `DERIVE_ONLY=1` writes only it (the day argument is then `-`) |
| `QSUB_RES`, `QSUB_RES_base`, `QSUB_RES_derive` | —, `QSUB_RES`, `QSUB_RES` | resources per job type |
| `JOB_NAME`, `JOB_DIR` | `Overview`, `logs/qsub_jobs/overview/<DATE>` | |
| `EVAL_MASTER_DIR` | this checkout | where the jobs `cd` to |

Every job exports `EVAL_MASTER_REV`. It is taken from `$OUT_ROOT/EVAL_BLOCK.txt` when that file
exists, else from `git`, and it ends up in each run's `launches.jsonl`. With `variants=`
present, `run_eval_checkpoints.sh --resume` skips a seed only when no unit is missing, and it
forwards `resume=true`.

`qsub_eval_day.sh` accepts the same `VARIANTS` directly (`all` or a list) if you want only the
ensemble jobs.

### 7.4 Analyse

```bash
./analysis/run_analysis.sh --arm base=... --arm dpo='.../seed_*' ... --reference base \
    --variants-overview --out outputs/analysis/overview
python -m analysis.overview --from outputs/analysis/overview   # re-plot without re-parsing
```

With the flag, the sibling dirs are parsed and their records join the `modified/` tree. The
original tree stays byte-identical. `--out/overview/` then holds:

- one figure per benchmark task: every arm under every variant, mean ± seed-bootstrap CI, base
  as a dashed line, one arm order and colour everywhere (sorted by MC2);
- `rank_grid.png`;
- `delta_vs_base.png`: (arm − base) / pooled seed SD;
- `faitheval_length.png`: accuracy against answer length;
- `overview.tsv` / `.md`.

The overview has no p-values and never writes a `comparisons.json`. Without the flag, the
analysis ignores sibling dirs entirely.

### 7.5 The comparison sets on the cluster: `qsub_overview_compare.sh`

The overview counterpart of `qsub_comparisons.sh`, and one CPU job. It first brings the CPU
variants of `OUT_ROOT` up to date and writes `status.tsv`. It then runs
`run_analysis.sh --variants-overview --no-compare` once per arm set, into its own
`ANALYSIS_ROOT` (default `outputs/analysis/<root>_overview`). The §5.7 comparisons and the §5.8
table stay with `qsub_comparisons.sh`.

Arms are found as in `qsub_comparisons.sh`: `$OUT_ROOT/<METHOD>_ensemble$SUFFIX/`, with base at
`$OUT_ROOT/base/run_0`. Arm sets whose arms are missing are left out and listed.

| `SETS` entry | Overviews written | Dashed line (Δ reference) |
|---|---|---|
| `claims` (default) | `claimA_dpo`, `claimA_orpo`, `sign_dpo`, `sign_orpo`, `claimB_dpo`, `claimB_orpo`: the §5.7 pairs | the pair's reference arm |
| `families` (default) | `family_dpo`, `family_orpo`: base plus every evaluated arm of that objective | `base` |
| `all` | `all_arms`: base plus every evaluated arm (past 8 arms, the colours fold to grey) | `base` |

```bash
OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh          # print
OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh check    # dry-run the job
QSUB_RES="-A <project> -l select=1:ncpus=2:mem=16gb -l walltime=04:00:00" \
    OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh submit
```

Other env: `BENCHES`, `BASE_DIR`, `VARIANTS` (derive step, default `all`), `DERIVE=0` (skip
the derive step), `PLOT=0` (tables only), `ANALYSIS_ROOT`, `JOB_NAME` (`Overview`), `JOB_DIR`
(`logs/qsub_jobs/overview`), `EVAL_MASTER_DIR`.

### 7.6 End to end: train, evaluate, overview

The overview needs nothing special from training. A day trained with
`SP-DPO-Base/scripts/qsub_train.sh` feeds straight into it:

```bash
# 1) train (SP-DPO-Base): a materialized profile variant and some arms
cd /gpfs/project/$USER/git-source/SP_DPO/SP-DPO-Base
PROFILE=materialize VARIANT=ups0_k8 MATERIALIZE_ARGS="--upsilon 0.0 --k0 8 --eta 0.55" \
ARMS="dpo curriculum" ARGS_dpo="training.learning_rate=5e-5" SEEDS_off_sp_dpo_legacy="42 1337 2024" \
QSUB_RES="-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=24:00:00 -A <project>" \
  bash scripts/qsub_train.sh submit
#    -> outputs/<DATE>/<HH-MM-SS>_<arm>_ensemble/seed_*   (ARMS="dpo curriculum" = dpo plus
#       all seven off_sp_* arms, the ORPO ones included)

# 2) evaluate every variant (Eval_master); keep the profile variant apart by SUFFIX
cd /gpfs/project/$USER/git-source/SP_DPO/Eval_master
SUFFIX=-ups0_k8 BASE_MODEL=... EXTRA="model.dtype=bfloat16 harness.batch_size=8" QSUB_RES="..." \
  ./analysis/qsub_overview_day.sh ../SP-DPO-Base/outputs/<DATE> submit

# 3) after the GPU jobs: the descriptive overviews
SUFFIX=-ups0_k8 QSUB_RES="..." ./analysis/qsub_overview_compare.sh submit
```

Keep two names apart:

- **`VARIANT` in `qsub_train.sh`** is the *difficulty profile* the curriculum trained on
  (`ups0_k8`).
- **`VARIANTS` in the overview scripts** are *scoring variants* of the evaluation.

A second profile variant changes the checkpoints. Give it its own `SUFFIX` or `OUT_ROOT`, and
never mix it into the arms of another profile. `base` is shared, and it is evaluated only once
per root.

---

## Option reference

| Key / flag | Default | Values |
|---|---|---|
| `halueval.scoring` | `generate` | `generate`, `constrained`, `both` |
| `halueval.decontam.enabled` | `false` | `true`, `false` |
| `halueval.decontam.exclusion_dir` | `decontamination/ragtruth` | dir relative to `Eval_master` |
| `faitheval.tasks` | `[unanswerable, inconsistent, counterfactual]` | add or use `counterfactual_mc` |
| `ragtruth.split` | `test` (was `null`) | `test`, `all`, `train`, `dev` |
| `ragtruth.stage` | `all` | `all`, `generate`, `detect` |
| `ragtruth.batch_size` | `8` | int ≥ 1 |
| `ragtruth.detector.batch_size` | `4` | int ≥ 1 |
| `ragtruth.detector.seed` | `42` | int |
| `run_eval_checkpoints.sh --resume` | off | variant-aware (see §2.2) |
| `run_analysis.sh --protocol` | `both` | `original`, `modified`, `both` |
| `run_analysis.sh --arm NAME=SPEC` (repeated name) | — | merges the specs into one arm |
| `score_results.py --exclude LIST.json` | — | score only the kept rows |
| `score_results.py --emit-summary DIR` | — | write the decontaminated summary (needs `--exclude`) |
| `run_decontam.sh --root / --out-root / --list` | — / — / all lists | see §2.3 |
| `run_ragtruth_detect.sh --root / --dtype / --dry-run` | — / `bfloat16` / off | see §2.4 |
| `run_overlap.sh --train / --harness-samples / --write-exclusions` | ragtruth / — / off | see §4 |
| `variants` / `resume` / `recompute_stale` / `strict_provenance` | `null` / `false` / `false` / `false` | overview only, see §7 |
| `evaluate.py --constrained-output-dir DIR` | `--output-dir` | where constrained artifacts go |
| `score_results.py --emit-variant lenient\|parsed` | — | with `--emit-summary`; exit 3 = unavailable |
| FaithEval `src/rescore.py --rule strict\|wordmatch\|contains\|lenient` | — | offline re-scoring, see §7 |
| `run_derive.sh` / `analysis.variants` / `analysis.adopt` | — | see §7.1 |
| `run_analysis.sh --variants-overview` | off | see §7.4 |
| `qsub_eval_day.sh` `VARIANTS=all\|<list>` | unset | see §7.3 |
| `qsub_overview_day.sh` / `qsub_overview_compare.sh` | — | cluster jobs for the overview, see §7.3 and §7.5 |

The underlying CLIs accept the same options directly: `evaluate.py --scoring`,
`evaluate.py --exclude-list`, FaithEval `run_eval.py --task counterfactual_mc`, and RAG-Truth
`run_eval.py --batch-size`, `--detector-batch-size`, `--detector-seed` and
`--stage detect --output-dirs DIR…`. Called directly, RAG-Truth's own defaults are still batch 1 and
unseeded; the launcher passes the conf values.
