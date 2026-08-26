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
- [Overriding the primary metric](#overriding-the-primary-metric----primary-map)
- [Generating runs first (`--run-evals`)](#generating-runs-first----run-evals)
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
| `--arm NAME=SPEC`       | — (required, repeatable) | Declare one arm. See [Declaring arms](#declaring-arms----arm-namespec).                          |
| `--out DIR`             | `outputs/analysis`       | Where results are written.                                                                       |
| `--reference NAME`      | auto                     | Arm every other arm is compared against. See [Reference arm](#reference-arm--comparison-regime). |
| `--benchmarks a,b,c`    | all present              | Include-list of benchmark folders.                                                               |
| `--exclude a,b`         | none                     | Exclude-list, applied after include.                                                             |
| `--no-compare`          | off                      | Aggregate only; skip cross-arm comparison.                                                       |
| `--no-plot`             | off                      | Skip figures (no matplotlib needed). _(`run_eval_ensemble.sh` always sets this.)_                |
| `--allow-seed-mismatch` | off                      | Paired mode intersects shared seeds instead of failing loudly on a mismatch.                     |
| `--primary-map FILE`    | built-in                 | YAML/JSON overriding the primary-metric-per-benchmark map.                                       |
| `--rng-seed N`          | `0`                      | Bootstrap RNG seed (reproducible CIs).                                                           |
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

`--allow-seed-mismatch` controls the paired-mode safety check:

| Flag                    | Default behavior                                                                                                                 | With the flag                                                               |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `--allow-seed-mismatch` | Paired mode **fails loudly** (`SeedMismatchError`) if the two arms don't share an identical seed set — re-run the missing seeds. | Silently intersect and test over the shared seeds (still aligned by value). |

## Benchmark selection

| Flag                 | Default     | Meaning                                                                            |
| -------------------- | ----------- | ---------------------------------------------------------------------------------- |
| `--benchmarks a,b,c` | all present | Comma-separated **include** list; only these benchmark folders are parsed/plotted. |
| `--exclude a,b`      | none        | Comma-separated **exclude** list, applied after include.                           |

Benchmark names: `faitheval`, `truthfulqa`, `halueval`, `ragtruth`, `harness`.

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
| [`fixtures.py`](fixtures.py)                 | Synthetic run-dir generators (used by the offline tests).                                                                                                             |
| [`cli.py`](cli.py)                           | `python -m analysis.cli` — ties it all together.                                                                                                                      |
