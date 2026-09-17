# Analysis module (`analysis/`)

Reads the benchmark run dirs produced by [`run_benchmarks.py`](../run_benchmarks.py),
**aggregates** across seeds, **compares** arms with the statistically correct test, and
**plots** the result. It never re-reads a model — it only consumes on-disk summaries —
so it is fast, offline, and re-runnable.

The unit of comparison is an **arm** (e.g. `base`, `sft`, `dpo`): a set of run dirs, one
per seed, treated as an ensemble. A one-run arm with no recoverable seed is a **fixed
point** (e.g. an untrained base evaluated once) and gets the one-sample regime instead of
a paired test.

```
run dirs ──discover──▶ (run_dir, seed) pairs ──parse──▶ MetricRecord rows
                                                              │
                                                    aggregate (across seeds)
                                                              │
                                          ┌───────────────────┼───────────────────┐
                                     ArmAggregate         compare (arm vs ref)   plot
                                     mean/std/CI          paired | one-sample    figures
```

## Contents

- [Install](#install)
- [First: do the eval dirs exist yet?](#first-do-the-eval-dirs-exist-yet)
- [Three scripts, one job each](#three-scripts-one-job-each)
- [Cheat sheet — every flag at a glance](#cheat-sheet--every-flag-at-a-glance)
- [Recipes](#recipes)
- [Declaring arms](#declaring-arms----arm-namespec)
- [Reference arm & comparison regime](#reference-arm--comparison-regime)
- [Benchmark selection](#benchmark-selection)
- [Original and modified protocols](#original-and-modified-protocols)
- [Optional: the all-variants overview](#optional-the-all-variants-overview)
- [Overriding the primary metric](#overriding-the-primary-metric----primary-map)
- [Generating runs first (`--run-evals`)](#generating-runs-first----run-evals)
- [Train↔eval overlap and decontamination](#traineval-overlap-and-decontamination)
- [RAG-Truth: shared detection job](#rag-truth-shared-detection-job)
- [Outputs](#outputs)
- [Key concepts](#key-concepts)
- [Testing](#testing)
- [Source map](#source-map)

---

## Install

The analysis package ships with the launcher (`eval-master`). It needs `numpy` (core);
`scipy` and `matplotlib` are lazy extras pulled in only when you compare / plot:

```bash
# from Eval_master/, in its own .venv
uv sync --extra stats --extra plot     # or: pip install -e '.[stats,plot]'

uv sync --extra stats --extra plot --extra dev #for dev dependencies
```

Without `stats`, the Wilcoxon test degrades gracefully (returns NaN stat/p, keeps the
bootstrap CI). Without `plot`, use `--no-plot`.

---

## First: do the eval dirs exist yet?

**The analysis scripts below never run a model — they only read eval run dirs that
already contain the per-benchmark subfolders (`faitheval/`, `harness/`, ...).** Pointing
`--arm dpo=.../seed_*` at a _training_ ensemble (a `..._ensemble/seed_<SEED>/` group dir
of `final_checkpoint`s) therefore finds **zero** records — the ensemble looks "unrecognised".
That directory holds checkpoints, not benchmark results; something has to evaluate them first.

[`run_eval_checkpoints.sh`](run_eval_checkpoints.sh) is that step. It walks a training
ensemble, runs [`run_benchmarks.py`](../run_benchmarks.py) on each seed's checkpoint, and
writes the results to a **seed-preserving** `<out>/seed_<SEED>/` layout — exactly what the
analysis scripts expect, paired by seed value:

```bash
# evaluate every seed checkpoint, then compare the ensemble against base in one go
./run_eval_checkpoints.sh \
    --checkpoints '/gpfs/.../08-48-52_sft_ensemble/seed_*' \
    --out outputs/eval/sft_ensemble \
    --analyze --name sft --arm base=/gpfs/.../20-38-30 --reference base

# or just produce the eval dirs (no --analyze) and hand them to run_analysis.sh yourself
./run_eval_checkpoints.sh --checkpoints '.../seed_*' --out outputs/eval/sft_ensemble
./run_analysis.sh --arm sft='outputs/eval/sft_ensemble/seed_*' \
    --arm base=/gpfs/.../20-38-30 --reference base --out outputs/analysis
```

| Flag                       | Meaning                                                                                                                     |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--checkpoints SPEC`       | Training ensemble: a `..._ensemble/` group dir, a `.../seed_*` glob, or one seed dir.                                       |
| `--out DIR`                | Root for the produced eval dirs (`<DIR>/seed_<SEED>/` each).                                                                |
| `--checkpoint-subdir NAME` | Model subdir inside each seed (default `final_checkpoint`; falls back to the seed dir itself).                              |
| `--benchmarks a,b`         | Subset of the five to run (default: all).                                                                                   |
| `--num-samples N`          | Per-benchmark sample cap forwarded to the launcher.                                                                         |
| `--dry-run`                | Print the launcher commands without loading any model.                                                                      |
| `--stop-on-error`          | Abort on the first failed seed (default: keep going; a partial ensemble still aggregates).                                  |
| `--analyze`                | Chain into `analysis.cli` on the produced ensemble (`--name`, `--arm`, `--reference`, `--analysis-out`, `--no-plot` apply). |
| `--launcher-extra ...`     | Extra Hydra overrides forwarded verbatim to the launcher (**must come last**).                                              |

> Seeds are preserved by **value** (`seed_42`, not `seed_0`), matching the training-side
> naming, so the downstream paired comparison lines up by seed identity. A
> `run_metadata.json` carrying the seed + source checkpoint is stamped into each output dir.

Already have eval dirs (or used the training-side pipeline that writes them)? Skip straight
to the three analysis scripts.

## Three scripts, one job each

All three are thin shell wrappers that resolve `Eval_master/.venv` (honouring
`$VENV_ROOT`, or `$PYTHON` directly) and forward every flag to a Python module — so you
can always call the module directly instead. Pick the one that matches how much of the
pipeline you want to run:

> **If a wrapper cannot find its interpreter it now fails, naming the path it looked
> under and the current `$VENV_ROOT`.** It used to fall through to a bare `python`, so on
> the cluster the failure surfaced as an unattributable `: No such file or directory` out
> of `exec`. The usual cause is a `$VENV_ROOT` exported in the shell (or SLURM script)
> that `setup_envs_HPC.sh` was *not* run with: the wrappers then look under
> `$VENV_ROOT/Eval_master` while the envs sit in-repo. Keep the two in sync (both accept
> `--venv-root`), or `unset VENV_ROOT`.

| Script                                             | Runs                                                                                    | Skip this if...                                                               |
| -------------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **[`run_analysis.sh`](run_analysis.sh)**           | aggregate → compare → plot (everything)                                                 | you just want numbers, not figures — use `run_eval_ensemble.sh`               |
| **[`run_eval_ensemble.sh`](run_eval_ensemble.sh)** | aggregate (+ compare); **no figures**                                                   | you already have `records.jsonl` and only want to replot — use `run_plots.sh` |
| **[`run_plots.sh`](run_plots.sh)**                 | replot from persisted `records.jsonl` / `comparisons.json`; **no eval, no aggregation** | you don't have persisted artifacts yet                                        |

```
run_eval_checkpoints.sh = python -m analysis.eval_checkpoints   # checkpoints -> eval dirs
run_analysis.sh         = python -m analysis.cli
run_eval_ensemble.sh    = python -m analysis.cli --no-plot
run_plots.sh            = python -m analysis.plot

# around the evals rather than part of the analysis pipeline
run_overlap.sh          = python -m analysis.overlap              # train <-> eval overlap, exclusion lists
run_decontam.sh         = score_results.py --exclude, per run     # finished HaluEval runs -> modified root
run_ragtruth_detect.sh  = python -m analysis.ragtruth_detect      # one detector load for many run dirs
```

---

## Cheat sheet — every flag at a glance

### `run_eval_checkpoints.sh` (→ `analysis.eval_checkpoints`)

| Flag                       | Default            | Meaning                                                                                    |
| -------------------------- | ------------------ | ------------------------------------------------------------------------------------------ |
| `--checkpoints SPEC`       | — (required)       | Training ensemble: a `..._ensemble/` group dir, a `.../seed_*` glob, or one seed dir.      |
| `--out DIR`                | — (required)       | Root for the produced eval dirs (`<DIR>/seed_<SEED>/` each).                               |
| `--checkpoint-subdir NAME` | `final_checkpoint` | Model subdir inside each seed dir; falls back to the seed dir itself if absent.            |
| `--benchmarks a,b`         | all five           | Subset to run per checkpoint; also forwarded to `analysis.cli` when `--analyze` is set, so extra `--arm`s scoped to more benchmarks don't leak in. |
| `--num-samples N`          | full dataset       | Per-benchmark sample cap forwarded to the launcher.                                        |
| `--python PATH`            | this interpreter   | Interpreter to run the launcher with.                                                      |
| `--dry-run`                | off                | Print the planned launcher commands without loading any model.                             |
| `--stop-on-error`          | off                | Abort on the first failed seed (default: keep going; a partial ensemble still aggregates). |
| `--resume`                 | off                | Skip seeds whose requested benchmarks all wrote their summary. The `--launcher-extra` overrides decide which summary counts: `halueval.scoring=constrained` needs `*_constrained_summary.json`, `faitheval.tasks=[counterfactual_mc]` needs `counterfactual_mc_summary.json`, `ragtruth.stage=generate` needs `generation_summary.json`. |
| `--launcher-extra ...`     | none               | Extra Hydra overrides forwarded to the launcher. **Must be last** (`nargs=REMAINDER`).     |
| `--analyze`                | off                | After evaluating, chain into `analysis.cli` on the produced ensemble.                      |
| `--name NAME`              | `dpo`              | Arm name for the produced ensemble when `--analyze`.                                       |
| `--arm NAME=SPEC`          | none               | Extra arm(s) for `--analyze` (e.g. `base=<eval_dir>`). Repeatable.                         |
| `--reference NAME`         | auto               | Reference arm for `--analyze`.                                                             |
| `--analysis-out DIR`       | `<out>/analysis`   | Output dir for `--analyze`.                                                                |
| `--no-plot`                | off                | Skip figures in `--analyze`.                                                               |
| `--exclude a,b`            | none                | Benchmark exclude-list, forwarded to `analysis.cli` for `--analyze`.                       |
| `--primary-map FILE`       | built-in            | Primary-metric override, forwarded to `analysis.cli` for `--analyze`.                      |
| `--rng-seed N`              | `0`                 | Bootstrap RNG seed, forwarded to `analysis.cli` for `--analyze`.                           |
| `--allow-seed-mismatch`    | off                 | Forwarded to `analysis.cli` for `--analyze` (see paired-mode note below).                  |

### `run_analysis.sh` / `run_eval_ensemble.sh` (→ `analysis.cli`)

| Flag                    | Default                  | Meaning                                                                                          |
| ----------------------- | ------------------------ | ------------------------------------------------------------------------------------------------ |
| `--arm NAME=SPEC`       | — (required, repeatable) | Declare one arm; repeating a NAME merges its specs (an arm's original and modified roots). See [Declaring arms](#declaring-arms----arm-namespec). |
| `--out DIR`             | `outputs/analysis`       | Where results are written.                                                                       |
| `--reference NAME`      | auto                     | Arm every other arm is compared against. See [Reference arm](#reference-arm--comparison-regime). |
| `--benchmarks a,b,c`    | all present              | Include-list. A base name (`halueval`) also selects its modified variants; a dotted name (`halueval.constrained`) selects that one only. |
| `--exclude a,b`         | none                     | Exclude-list, applied after include (same matching).                                             |
| `--protocol P`          | `both`                   | Which trees to write: `original` → `--out` (the usual layout), `modified` → `--out/modified/`, or `both`. See [protocols](#original-and-modified-protocols). |
| `--no-compare`          | off                      | Aggregate only; skip cross-arm comparison.                                                       |
| `--no-plot`             | off                      | Skip figures (no matplotlib needed). _(`run_eval_ensemble.sh` always sets this.)_                |
| `--allow-seed-mismatch` | off                      | Paired mode intersects shared seeds instead of failing loudly on a mismatch.                     |
| `--primary-map FILE`    | built-in                 | YAML/JSON overriding the primary-metric-per-benchmark map.                                       |
| `--rng-seed N`          | `0`                      | Bootstrap RNG seed (reproducible CIs).                                                           |
| `--mc-method M`         | `holm`                   | Multiplicity correction across the paired family: `holm`, `bh` or `none`.                         |
| `--run-evals`           | off                      | Drive the Hydra launcher first to produce run dirs. Requires `--models`.                         |
| `--models id1,id2`      | —                        | Comma-separated model ids for `--run-evals`.                                                     |
| `--eval-sweep-dir DIR`  | `outputs/analysis_sweep` | Hydra `sweep.dir` for `--run-evals`.                                                             |
| `--eval-extra ...`      | none                     | Extra Hydra overrides, forwarded verbatim. **Must be last** (`nargs=REMAINDER`).                 |
| `--dry-run`             | off                      | With `--run-evals`, print the launcher command instead of running it.                            |

### `run_plots.sh` (→ `analysis.plot`)

| Flag               | Default          | Meaning                                                                  |
| ------------------ | ---------------- | ------------------------------------------------------------------------ |
| `--from DIR`       | — (required)     | Analysis dir containing `records.jsonl` (+ optional `comparisons.json`). |
| `--out DIR`        | `<from>/figures` | Figure output dir.                                                       |
| `--reference NAME` | none             | Reference label for delta plots.                                         |
| `--benchmarks a,b` | all              | Include-list.                                                            |
| `--exclude a,b`    | none             | Exclude-list.                                                            |
| `--protocol P`     | `both`           | Plot only `original` or only `modified` benchmarks.                      |

### `run_overlap.sh` (→ `analysis.overlap`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--train NAME=SPEC` | `ragtruth=<SP-DPO-Base>/data/ragtruth` | Training corpus, repeatable: a `data/ragtruth` dir, or `jsonl:PATH:field,field[:group_field]` (PATH may be a glob). |
| `--train-run-config PATH` | none | A training run's `.hydra/config.yaml`; flags `halueval*` / `truthfulqa*` data as identity overlap. Repeatable. |
| `--harness-samples PATH` | none | A harness `samples.jsonl`, to check all 817 TruthfulQA items (the local CSV has 790). |
| `--benchmarks a,b` | all five | Subset of `halueval,faitheval,truthfulqa,harness,ragtruth_benchmark`. |
| `--ragtruth-split S` | `test` | RAG-Truth benchmark split to tier; the `all` identity counts are reported too. |
| `--ngram` `--near-dup` `--df-max` `--exposure-topk` | `13` `0.5` `10` `3` | Matching parameters. Keep the defaults: `KNOWN_ISSUES.md` §5 was measured with them. |
| `--out DIR` | `outputs/overlap` | Report: `<DIR>/<train>/overlap_report.{json,md}`. |
| `--write-exclusions` | off | Also write `<exclusions-dir>/<train>/<benchmark>__<task>.json` for every non-empty list, beside a copy of the report. |
| `--exclusions-dir DIR` | `decontamination` | Where those go (committed). |

### `run_decontam.sh`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--root DIR` | — (required) | Eval root with finished `<arm>/<run>/halueval/*_results.json`. Only read. |
| `--out-root DIR` | — (required) | The modified-protocol root; must differ from `--root`. Gets `<arm>/<run>/halueval/*_decontam_summary.json` and `run_metadata.json`. |
| `--list LIST.json` | every `decontamination/ragtruth/halueval__*.json` | Exclusion list(s), repeatable. |

### `run_ragtruth_detect.sh` (→ `analysis.ragtruth_detect`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--root DIR` | — (required) | Eval root. Every `*/seed_*/ragtruth/` and `base/run_*/ragtruth/` with `generations.jsonl` and no `summary.json` is detected. |
| `--dtype` | `bfloat16` | Detector dtype — match the eval block's `model.dtype`. |
| `--detector-model-id` `--detector-base-model-id` `--detector-tokenizer-id` | `conf/ragtruth/default.yaml` | The detector. |
| `--detector-batch-size N` `--detector-seed N` | conf (`4`, `42`) | Batching and the per-directory seed; both recorded in `summary.json`. |
| `--dry-run` | off | List the pending run dirs and print the command. |
| `--extra ...` | none | Raw flags for `run_eval.py` (**must be last**). |

---

## Recipes

```bash
./run_eval_checkpoints.sh \
    --checkpoints '/gpfs/.../08-48-52_sft_ensemble/seed_*' \
    --out outputs/eval/sft_ensemble \
    --analyze --name sft \
    --arm base='/gpfs/.../20-38-30' \
    --reference base \
    --dry-run

# 1) One ensemble, aggregate + plot, no comparison:
./run_analysis.sh --arm dpo='outputs/.../*_dpo_ensemble/seed_*' --no-compare

# 2) base (fixed point) vs SFT vs DPO ensembles, full compare + figures:
./run_analysis.sh \
    --arm base=outputs/.../base_run \
    --arm sft='outputs/.../*_sft_ensemble/seed_*' \
    --arm dpo='outputs/.../*_dpo_ensemble/seed_*' \
    --reference base --out outputs/analysis

# 3) Aggregate only (no figures), restrict to two benchmarks:
./run_eval_ensemble.sh --arm dpo='outputs/.../seed_*' --no-compare \
    --benchmarks faitheval,harness

# 4) Re-plot persisted artifacts later, excluding one benchmark:
./run_plots.sh --from outputs/analysis --exclude ragtruth

# 5) Produce evals for a model list first (Hydra --multirun), then analyse + plot:
./run_analysis.sh --run-evals --models ckptA,ckptB --reference model0 \
    --eval-extra run='[faitheval,harness]' num_samples=50

# 6) Override which metric counts as "primary" for a benchmark:
./run_analysis.sh --arm base=... --arm dpo=... --primary-map my_primary.yaml
```

---

## Declaring arms — `--arm NAME=SPEC`

`NAME` is the arm label (used in tables, plots, and as a `--reference` target). `SPEC`
resolves to run dirs via [`discover.py`](discover.py) and accepts:

| Spec form                                                               | Resolves to                                                                                                           |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| a **leaf run dir** (has `run_metadata.json` or a benchmark subfolder)   | itself                                                                                                                |
| a **container dir** (a `--multirun` root, or a `*_ensemble/` group dir) | its run-dir children (`seed_*`, numeric `0/1/2`, or benchmark-bearing); non-run children like `ensemble/` are ignored |
| a **glob**                                                              | every matching run dir                                                                                                |

**Seed inference** (per run dir): `run_metadata.json`'s `"seed"` field → else a
`seed_<N>` directory name → else `None`. There is deliberately **no hash fallback**: an
unrecoverable seed stays `None`, and paired mode refuses to fabricate a seed identity.

**Fixed-point markers** — append to the spec to force the regime:

- `NAME=SPEC!fixed` → treat the arm as a fixed point (one-sample regime, no seed variance).
- `NAME=SPEC!ensemble` → force ensemble treatment even for a single run dir.
- Default (no marker): auto — a single run dir with no recoverable seed is a fixed point.

```bash
--arm base=outputs/2026-07-20/01-19-15                      # single run, auto fixed point
--arm dpo='outputs/.../*_dpo_ensemble/seed_*'               # glob → ensemble of seeds
--arm base=outputs/.../base_run!fixed                       # force fixed point
```

**Repeating a name merges.** `--arm dpo=$EVAL_ROOT/dpo --arm dpo=$EVAL_ROOT_MOD/dpo` reads both
roots as one arm; this is how an arm's original-protocol runs and its modified-protocol runs enter
one call. The merged arm lists each seed once, and it is a fixed point only if every spec is one.
It fails if two of its run dirs report the same metric for the same seed, which means the roots
overlap.

## Reference arm & comparison regime

| Flag               | Default | Meaning                                                                                                                                                         |
| ------------------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--reference NAME` | auto    | The arm every other arm is compared **against**. Auto-pick order: an arm literally named `base` → else the first fixed-point arm → else the first arm declared. |

The comparison **regime is chosen automatically** per arm pair (see
[`compare.py`](compare.py)):

- **paired** — both arms are multi-seed ensembles with real seeds. Values are paired by
  seed **value** (never by list position — this fixes a training-side bug), then a
  Wilcoxon signed-rank test + paired bootstrap CI of the differences.
- **one-sample** — either side is a fixed point (zero seed variance, so a paired test is
  degenerate). The ensemble is characterised (mean ± std + bootstrap CI) and its distance
  from the fixed point is reported. No Wilcoxon.

**Multiplicity.** The grid is `(arms - 1) x primary keys` simultaneous tests against one
reference, so p-values are family-adjusted across the paired comparisons: `--mc-method holm`
(family-wise error rate, the default), `bh` (false discovery rate), or `none`. The raw
`p_value` is kept beside `p_value_adjusted` in `comparisons.json` so the correction stays
auditable; the significance stars carry the **adjusted** value. One-sample cells have no
p-value and are excluded from the family, as are tests that did not run (`mc_family_size`
records what was actually corrected over).

> **Read this before interpreting any p-value here.** The exact two-sided Wilcoxon p-value
> is bounded below by `2 / 2**n`, so at **n = 5 seeds no paired result can reach p < 0.05**
> (minimum 0.0625) — for any effect size, before any correction. An absence of stars at
> n ≤ 5 is a property of the sample size, never evidence of equivalence. Six seeds is the
> minimum at which the test can reject at all; ten is where it has usable power. See
> [`analysis_theory.md`](analysis_theory.md) §5.2.

> **Read this before interpreting HaluEval or FaithEval `counterfactual` numbers.** Those
> metrics are scored by length-sensitive parsers, and in the `v3` runs they track output
> *format* almost perfectly: `corr(format_compliance, accuracy) = 0.9974`, with accuracy
> among rows that parsed pinned at chance (0.4981 ± 0.0165) across all 132 files. Arm
> rankings on those axes are compliance rankings. See
> [`format_confound.md`](format_confound.md).

`--allow-seed-mismatch` controls the paired-mode safety check:

| Flag                    | Default behavior                                                                                                                 | With the flag                                                               |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `--allow-seed-mismatch` | Paired mode **fails loudly** (`SeedMismatchError`) if the two arms don't share an identical seed set — re-run the missing seeds. | Silently intersect and test over the shared seeds (still aligned by value). |

## Benchmark selection

| Flag                 | Default     | Meaning                                                                            |
| -------------------- | ----------- | ---------------------------------------------------------------------------------- |
| `--benchmarks a,b,c` | all present | Comma-separated **include** list; only these benchmark folders are parsed/plotted. |
| `--exclude a,b`      | none        | Comma-separated **exclude** list, applied after include.                           |

Benchmark names: `faitheval`, `truthfulqa`, `halueval`, `ragtruth`, `harness`, plus the modified
variants below. A base name selects the benchmark **and** its variants (`--benchmarks halueval`
keeps `halueval.constrained`); a dotted name selects exactly one variant.

## Original and modified protocols

A modified protocol is a scorer that departs from the published one. It runs only when switched
on, and it is reported under its **own benchmark name**, so it can never be averaged into, compared
with or plotted as an original number:

| Benchmark name | Written by | Primary metric |
| --- | --- | --- |
| `halueval.constrained` | `evaluate.py --scoring constrained` (`*_constrained_summary.json`) | `auroc` (its SE in `stderr`) |
| `halueval.decontam` | `score_results.py --exclude … --emit-summary`, or `run_decontam.sh` (`*_decontam_summary.json`) | `accuracy` on the kept rows; `seen_minus_clean` and per-row-set accuracies alongside |
| `halueval.constrained_decontam` | the same, over constrained results | `auroc` |
| `faitheval.mc` | `faitheval.tasks=[counterfactual_mc]` (`counterfactual_mc_summary.json`, `scoring: choice_loglik`) | `accuracy` (+ `accuracy_norm`) |

**How a summary is routed.** The parser reads each summary's own `scoring` / `variant` field; a
summary without either parses exactly as before. A dot in the benchmark name is the whole
definition of "modified", so an old `records.jsonl` reloads as original.

`analysis.cli` writes the two protocols to separate trees:

- **original** → `--out/`, in the layout every existing call reads. Adding modified roots to a
  call leaves these files byte-identical.
- **modified** → `--out/modified/`: records, aggregate, comparisons, `.tex` tables carrying a
  protocol row, and `figures/`. It is written only when there are modified records.
  - An arm without modified runs has no row in this tree.
  - If the reference arm has none, the tree skips its comparisons.

`--protocol original|modified|both` chooses which trees to write. Modified comparisons form their
own multiplicity family and are context only, whatever their p-value.

## Optional: the all-variants overview

**Descriptive, not pre-registered**, and off unless asked for. The runbook is
[RUNNING_NEW_OPTIONS.md §7](../RUNNING_NEW_OPTIONS.md#7-optional-all-variants-in-one-run-overview).

**Layout.** Every variant sits in a sibling dir of the original, named like its benchmark:
`<run>/halueval.constrained/`, `<run>/faitheval.strict/`, … The registry in
[`variants.py`](variants.py) is the single source of truth for launcher, resume, derivation
and parsing. Its unit statuses are:

- `done`;
- `missing` (plus `waiting on <source>` for a CPU unit);
- `stale(reason)`: recorded settings or source sha differ;
- `unavailable`: lenient without `raw_judgement`;
- `unverified(field)`: counted as done.

| Tool | Does |
| --- | --- |
| `python -m analysis.variants --root R` | Status of every unit under a root, totals still to compute, and the recorded knobs per variant. |
| `./analysis/run_derive.sh --root R` (→ `analysis.derive`) | Derives every CPU variant (`halueval.parsed`, `.lenient`, `.decontam`, `.constrained_decontam`, `faitheval.strict`, `.wordmatch`, `.contains`). No GPU; originals are only read. |
| `python -m analysis.adopt --from R_mod --into R` | Optional. Copies legacy constrained / `counterfactual_mc` GPU artifacts into sibling dirs. Never overwrites; the source root is unchanged. |
| `run_analysis.sh --variants-overview` | Also parses the sibling dirs (their records join `modified/`) and writes `--out/overview/`. |
| `python -m analysis.overview --from OUT` | Re-plots the overview from `OUT/records.jsonl` + `OUT/modified/records.jsonl`. |
| `./analysis/qsub_overview_day.sh <DATE dir> [print\|check\|submit]` | PBS jobs for one training day: a GPU job per ensemble (`qsub_eval_day.sh` with `VARIANTS`), an optional base-model job (`BASE_MODEL`), and a CPU derive job over the whole root (`DERIVE_ONLY=1` for that alone). |
| `./analysis/qsub_overview_compare.sh [print\|check\|submit]` | One CPU PBS job: derive, a status table, then one overview per arm set (`SETS`: `claims` = the §5.7 pairs, `families`, `all`) into `outputs/analysis/<root>_overview/`. |

The cluster workflow, with every env variable and an end-to-end example starting from
`SP-DPO-Base/scripts/qsub_train.sh`, is in
[RUNNING_NEW_OPTIONS.md §7.3–§7.6](../RUNNING_NEW_OPTIONS.md#73-a-training-day-on-the-cluster-qsub_overview_daysh).
`qsub_eval_day.sh` itself accepts `VARIANTS=all` or a comma list of variants.

New benchmark names, each with `accuracy` as its primary metric:

- `halueval.parsed`, `halueval.lenient`;
- `faitheval.strict`, `faitheval.wordmatch`, `faitheval.contains`.

The FaithEval variants also carry `mean_prediction_words`. `contains` adds `exact_match` and
`accuracy_len_<1_5|6_20|21_60|61_plus>`. None of these are primary.

A new-style FaithEval summary with `strict_match: true` routes to `faitheval.strict`. A summary
in the wrong sibling dir is skipped with a warning. A variant found both flat and in its sibling
dir is an error.

**Without `--variants-overview`, dotted dirs are never read,** so the §5.8 analysis of a root
that also holds overview dirs is byte-identical. With the flag, the original tree is still
byte-identical.

`overview/` holds only descriptive outputs; it never writes a `comparisons.json`:

| File | Shows |
| --- | --- |
| `halueval_<task>.png`, `faitheval_<task>.png`, `harness.png` | One panel per variant, each with its own y-axis: arms as mean ± seed-bootstrap CI with faint seed dots, the reference arm as a dashed line, chance at 0.5 where meaningful. Arm order (by harness MC2) and colours are the same everywhere. |
| `rank_grid.png` | Arms × variants, each cell the rank (1 = best, direction-aware). MC2 is the first column. |
| `delta_vs_base.png` | (arm − base) / pooled seed SD, positive = better, diverging scale. |
| `faitheval_length.png` | Arm accuracy against arm answer length per variant, and `contains` per length stratum. |
| `overview.tsv`, `overview.md` | Arm × variant × task: mean, CI, seeds, rank, Δ/SD. |

## Overriding the primary metric — `--primary-map`

Each benchmark has a **primary** ("headline") metric that drives the default comparisons
and the main plots. The built-in defaults (from [`parse.py`](parse.py)):

| Benchmark  | Primary metric                                                |
| ---------- | ------------------------------------------------------------- |
| faitheval  | per-task `accuracy` (the synthesized `mean` task is excluded) |
| halueval   | per-task `accuracy` (ditto)                                   |
| ragtruth   | `overall` / `hallucination_rate`                              |
| truthfulqa | `MC1`, `MC2`                                                  |
| harness    | `truthfulqa_mc1` / `truthfulqa_mc2`, metric `acc`             |
| halueval.constrained, halueval.constrained_decontam | per-task `auroc` |
| halueval.decontam, faitheval.mc | per-task `accuracy` |

Override with a YAML or JSON file: `--primary-map my_primary.yaml`. Shape:

```yaml
# A benchmark listed here uses ONLY its listed entries; unlisted benchmarks keep the
# built-in default. "metric" matches that metric on any task; "task:metric" matches exactly.
faitheval: ["accuracy"] # accuracy on every faitheval task
harness: ["truthfulqa_mc2:acc"] # only mc2 acc
truthfulqa: ["MC2"]
```

## Generating runs first — `--run-evals`

Drive the Hydra launcher's `--multirun` to _generate_ run dirs before analysing them —
no separate launch step.

| Flag                   | Meaning                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--run-evals`          | Enable this mode. Requires `--models`.                                                                       |
| `--models id1,id2`     | Comma-separated model ids swept as `model.id=id1,id2` (one numbered run dir each).                           |
| `--eval-sweep-dir DIR` | Hydra `sweep.dir` (default `outputs/analysis_sweep`); becomes `<DIR>/0`, `<DIR>/1`, …                        |
| `--eval-extra ...`     | Extra Hydra overrides forwarded to the launcher. `nargs=REMAINDER` — **must come last** on the command line. |
| `--dry-run`            | Print the launcher command without running it.                                                               |

If you don't also pass `--arm`, each numbered sweep subdir becomes an arm named
`model0`, `model1`, …. This path needs the launcher + per-benchmark venvs (it is not
exercised by the offline test suite).

```bash
python -m analysis.cli --run-evals --models ckptA,ckptB --reference model0 \
    --eval-extra run='[faitheval,harness]' num_samples=50
```

## Train↔eval overlap and decontamination

[`overlap.py`](overlap.py) indexes a training corpus (default: SP-DPO-Base's `data/ragtruth`) and
tiers every row of every benchmark against it: `exact`, `near_duplicate` (≥ 50 % of the row's
13-grams), `partial` or `none`. Each matched row also gets the strongest *exposure* of its matched
sources: `train` > `val` > `test` > `release_only`. A missing benchmark file is listed as
`not_checked`, never counted as zero overlap. `SP-DPO-Base/KNOWN_ISSUES.md` §5 is its report in
prose, and `tests/test_overlap.py` pins the two together on the real data.

```bash
./analysis/run_overlap.sh --harness-samples outputs/eval/<arm>/seed_42/harness/samples.jsonl --write-exclusions
```

`--write-exclusions` writes the committed lists, each beside a copy of the report whose sha256 it
records. Today there is one: `decontamination/ragtruth/halueval__summarization.json`, with 503
excluded rows and 130 unseen-exposed controls. [`run_decontam.sh`](run_decontam.sh) applies the
lists to finished runs without a GPU, writing `halueval.decontam` summaries into a **separate**
root:

```bash
./analysis/run_decontam.sh --root "$EVAL_ROOT" --out-root "$EVAL_ROOT_MOD"
./analysis/run_analysis.sh --arm dpo="$EVAL_ROOT/dpo" --arm dpo="$EVAL_ROOT_MOD/dpo" ...   # -> <out>/modified/
```

## RAG-Truth: shared detection job

RAG-Truth's Stage 2 loads a 13B detector. For many checkpoints, run `ragtruth.stage=generate` per
checkpoint, then detect once. Use `run_eval_checkpoints.sh --resume`, which for `stage=generate`
waits for `generation_summary.json`:

```bash
./analysis/run_ragtruth_detect.sh --root "$EVAL_ROOT" --dtype bfloat16 --dry-run   # list pending dirs
./analysis/run_ragtruth_detect.sh --root "$EVAL_ROOT" --dtype bfloat16
```

The job loads the detector once and skips every run dir that already has a `summary.json`, so
re-submitting it resumes. Each dir gets `detections.jsonl` and `summary.json` exactly as
`stage=all` would write them.

---

## Outputs

Written under `--out` (default `outputs/analysis/`):

| File                      | Written by   | Contents                                                                                                     |
| ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------ |
| `records.jsonl`           | always       | The long-form `MetricRecord` dump — one line per scalar. The reload point for standalone plotting.           |
| `aggregate.json`          | always       | Per-arm, per-key mean / std / 95% CI / per-seed values.                                                      |
| `aggregate.tex`           | always       | Booktabs LaTeX table of the primary metrics (thesis appendix).                                               |
| `comparisons.json`        | if comparing | One row per arm-vs-reference comparison: regime, raw + signed delta, and the paired-test / one-sample block. |
| `comparisons.tex`         | if comparing | LaTeX table: signed Δ + Wilcoxon p and family-adjusted p (paired) or CI (one-sample). Stars follow the adjusted p. |
| `figures/*.png` + `*.pdf` | if plotting  | See below. Every figure is written as both a 150-dpi PNG and a vector PDF.                                   |
| `modified/…`              | if there are modified records | The same files for the modified protocols, with a protocol row in each `.tex`. See [protocols](#original-and-modified-protocols). |
| `overview/…`              | `--variants-overview` only | Descriptive arm-comparison figures and table across every variant. See [the overview](#optional-the-all-variants-overview). |

### Figures (`figures/`)

| Figure                       | Function                      | Shows                                                                                                                                                            |
| ---------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<bench>.png`                | `plot_benchmark`              | One benchmark, arms side by side on each primary (task, metric), CI error bars + individual seed points.                                                         |
| `<bench>_tasks.png`          | `plot_tasks`                  | Per-task grouped bars within a benchmark for one metric.                                                                                                         |
| `cross_benchmark_panels.png` | `plot_cross_benchmark_panels` | Small multiples — one panel per benchmark on its **native** scale (no cross-scale mixing), primary-metric mean per arm.                                          |
| `ranked_deltas.png`          | `plot_ranked_deltas`          | Direction-normalized signed delta of each arm vs. reference, sorted. Positive = improvement even for `hallucination_rate` (lower-is-better metrics are flipped). |
| `paired_<...>.png`           | `plot_paired_deltas`          | Per-seed deltas + mean + CI + Wilcoxon stars. **Paired regime only** — no-ops with a warning for one-sample comparisons.                                         |

Arm colours come from the fixed **Okabe–Ito** colourblind-safe palette, assigned in a
stable order so colour follows arm identity (never rank, never cycled). More than 8 arms
fold to grey with a warning.

A modified variant's figure names replace the dot with `__` (`halueval__constrained.png`), and its
titles carry `[modified protocol]`.

---

## Key concepts

- **Signed value / delta.** Metrics carry a `higher_is_better` flag; `signed_value` flips
  lower-is-better metrics so a positive delta always means "the arm improved on the
  reference," letting deltas be ranked across metrics of opposite direction. The one
  inverted metric is RAGTruth's `hallucination_rate` (see the direction table in
  [`parse.py`](parse.py)).
- **Bootstrap CIs** are a pure-numpy percentile bootstrap (10 000 resamples, α = 0.05),
  vendored verbatim from the training repo so figures reproduce the training-side ensemble
  numbers exactly. Seed with `--rng-seed`.
- **Single instance is not special-cased.** One model / one seed flows through as an
  ensemble of size one: `n_seeds == 1`, `std == 0`, degenerate CI.
- **Partial runs aggregate.** A missing or unreadable benchmark folder warns and is
  skipped; the rest still aggregates. Nothing hardcodes "five benchmarks" — adding one is
  a one-line edit to the `PARSERS` registry in [`parse.py`](parse.py).

## Testing

Offline, no model downloads — [`fixtures.py`](fixtures.py) materialises realistic run
dirs (each benchmark's real on-disk shape) that the suite parses:

```bash
uv run --extra dev --extra stats pytest      # from Eval_master/
```

One test is marked `slow`: `tests/test_overlap.py::test_real_data_reproduces_known_issues_section_5`.
It reads the real SP-DPO-Base and benchmark data (~20 s) and is skipped when either is absent;
`-m 'not slow'` leaves it out.

`tests/test_qsub_scripts.py` needs bash ≥ 4 (Git Bash on Windows; WSL's bash is skipped). It
runs the `qsub_*.sh` writers in `print` mode, then executes the written jobs against a stub
Eval_master whose `module`, `python` and wrapper scripts only log their arguments, so no HPC is
needed. `tests/test_variants.py` needs Hydra, which the Eval_master `.venv` has. It composes the
real config, stubs the subprocess runner, and compares every builder's commands with the
launcher as of commit `5c8cf56`, the last one before the overview.

## Source map

| File                                         | Role                                                                                                                                                                  |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`model.py`](model.py)                       | `MetricRecord` (one scalar for arm/seed/benchmark/task/metric) + `RecordSet` wrapper. The canonical intermediate; everything downstream consumes it.                  |
| [`discover.py`](discover.py)                 | Resolve an arm spec (dir / group dir / glob) to `(run_dir, seed)` pairs; infer seeds.                                                                                 |
| [`eval_checkpoints.py`](eval_checkpoints.py) | Upstream producer: evaluate a training `..._ensemble/seed_*` of checkpoints through the launcher into a seed-preserving eval layout the rest of the package consumes. |
| [`parse.py`](parse.py)                       | Per-benchmark summary readers → `MetricRecord` rows. Holds the metric-direction and primary-metric tables.                                                            |
| [`spec.py`](spec.py)                         | Arm declaration, `AnalysisConfig`, `NAME=SPEC` parsing, primary-metric override.                                                                                      |
| [`aggregate.py`](aggregate.py)               | Across-seed mean / std / bootstrap-CI per (benchmark, task, metric).                                                                                                  |
| [`compare.py`](compare.py)                   | Arm-vs-reference comparison; auto-selects paired vs one-sample.                                                                                                       |
| [`stats.py`](stats.py)                       | Vendored bootstrap CI, one-sample summary, seed-aligned Wilcoxon, Holm/BH multiplicity adjustment.                                                                    |
| [`plot.py`](plot.py)                         | Five matplotlib figure types; also a standalone re-plot entry point.                                                                                                  |
| [`report.py`](report.py)                     | JSON / JSONL / LaTeX writers.                                                                                                                                         |
| [`format_confound.md`](format_confound.md) | Why the `v3` HaluEval / FaithEval-counterfactual arm tables measure output shape rather than faithfulness, and what to report instead. |
| [`fixtures.py`](fixtures.py)                 | Synthetic run-dir generators (used by the offline tests).                                                                                                             |
| [`overlap.py`](overlap.py)                   | Train↔eval text-overlap checker: tiers, exposure, report, exclusion lists (stdlib only).                                                                             |
| [`ragtruth_detect.py`](ragtruth_detect.py)   | The shared RAG-Truth detection job over an eval root (one detector load).                                                                                             |
| [`cli.py`](cli.py)                           | `python -m analysis.cli` — ties it all together.                                                                                                                      |
| [`variants.py`](variants.py)                 | Optional overview: variant registry, per-unit status, CPU derivation commands, root census (stdlib only).                                                             |
| [`derive.py`](derive.py)                     | Optional overview: derive every CPU variant over a root.                                                                                                              |
| [`adopt.py`](adopt.py)                       | Optional overview: copy legacy modified-protocol GPU artifacts into sibling dirs.                                                                                     |
| [`overview.py`](overview.py)                 | Optional overview: descriptive arm-comparison figures, rank grid, Δ-vs-base grid, table.                                                                              |
| [`qsub_overview_day.sh`](qsub_overview_day.sh), [`qsub_overview_compare.sh`](qsub_overview_compare.sh) | Optional overview: PBS job writers, counterparts of `qsub_eval_day.sh` / `qsub_comparisons.sh`.                                                      |
| [`qsub_lib.sh`](qsub_lib.sh)                 | Shared helpers of the `qsub_*.sh` writers: bracket-aware override splitting, job header, code version.                                                                |
