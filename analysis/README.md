# Analysis module (`analysis/`)

Reads the benchmark run dirs produced by [`run_benchmarks.py`](../run_benchmarks.py),
**aggregates** across seeds, **compares** arms with the statistically correct test, and
**plots** the result. It never re-reads a model — it only consumes on-disk summaries —
so it is fast, offline, and re-runnable.

The unit of comparison is an **arm** (e.g. `base`, `sft`, `dpo`): a set of run dirs, one
per seed, treated as an ensemble. A one-run arm with no recoverable seed is a **fixed
point** (e.g. an untrained base evaluated once) and gets the one-sample regime instead of
a paired test.

## Pipeline at a glance

```
run dirs ──discover──▶ (run_dir, seed) pairs ──parse──▶ MetricRecord rows
                                                              │
                                                    aggregate (across seeds)
                                                              │
                                          ┌───────────────────┼───────────────────┐
                                     ArmAggregate         compare (arm vs ref)   plot
                                     mean/std/CI          paired | one-sample    figures
```

| File | Role |
| --- | --- |
| [`model.py`](model.py) | `MetricRecord` (one scalar for arm/seed/benchmark/task/metric) + `RecordSet` wrapper. The canonical intermediate; everything downstream consumes it. |
| [`discover.py`](discover.py) | Resolve an arm spec (dir / group dir / glob) to `(run_dir, seed)` pairs; infer seeds. |
| [`parse.py`](parse.py) | Per-benchmark summary readers → `MetricRecord` rows. Holds the metric-direction and primary-metric tables. |
| [`spec.py`](spec.py) | Arm declaration, `AnalysisConfig`, `NAME=SPEC` parsing, primary-metric override. |
| [`aggregate.py`](aggregate.py) | Across-seed mean / std / bootstrap-CI per (benchmark, task, metric). |
| [`compare.py`](compare.py) | Arm-vs-reference comparison; auto-selects paired vs one-sample. |
| [`stats.py`](stats.py) | Vendored bootstrap CI, one-sample summary, seed-aligned Wilcoxon. |
| [`plot.py`](plot.py) | Five matplotlib figure types; also a standalone re-plot entry point. |
| [`report.py`](report.py) | JSON / JSONL / LaTeX writers. |
| [`fixtures.py`](fixtures.py) | Synthetic run-dir generators (used by the offline tests). |
| [`cli.py`](cli.py) | `python -m analysis.cli` — ties it all together. |

## Install

The analysis package ships with the launcher (`eval-master`). It needs `numpy` (core);
`scipy` and `matplotlib` are lazy extras pulled in only when you compare / plot:

```bash
# from Eval_master/, in its own .venv
uv sync --extra stats --extra plot     # or: pip install -e '.[stats,plot]'
```

Without `stats`, the Wilcoxon test degrades gracefully (returns NaN stat/p, keeps the
bootstrap CI). Without `plot`, use `--no-plot`.

## Entry points

Three thin shell wrappers resolve `Eval_master/.venv` (honouring `$VENV_ROOT`) and call
the module. All flags are forwarded, so you can call the module directly instead.

| Wrapper | Equivalent | Does |
| --- | --- | --- |
| [`run_analysis.sh`](run_analysis.sh) | `python -m analysis.cli` | Full pipeline: aggregate → compare → plot. |
| [`run_eval_ensemble.sh`](run_eval_ensemble.sh) | `python -m analysis.cli --no-plot` | Aggregate (+ compare) only; no figures. |
| [`run_plots.sh`](run_plots.sh) | `python -m analysis.plot` | Re-plot from persisted `records.jsonl` / `comparisons.json`. |

---

## `analysis.cli` parameters

Run `python -m analysis.cli --arm NAME=SPEC ... [options]`.

### Declaring arms — `--arm NAME=SPEC` (repeatable, required)

`NAME` is the arm label (used in tables, plots, and as a `--reference` target). `SPEC`
resolves to run dirs via [`discover.py`](discover.py) and accepts:

| Spec form | Resolves to |
| --- | --- |
| a **leaf run dir** (has `run_metadata.json` or a benchmark subfolder) | itself |
| a **container dir** (a `--multirun` root, or a `*_ensemble/` group dir) | its run-dir children (`seed_*`, numeric `0/1/2`, or benchmark-bearing); non-run children like `ensemble/` are ignored |
| a **glob** | every matching run dir |

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

### Reference & regime — `--reference`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--reference NAME` | auto | The arm every other arm is compared **against**. Auto-pick order: an arm literally named `base` → else the first fixed-point arm → else the first arm declared. |

The comparison **regime is chosen automatically** per arm pair (see
[`compare.py`](compare.py)):

- **paired** — both arms are multi-seed ensembles with real seeds. Values are paired by
  seed **value** (never by list position — this fixes a training-side bug), then a
  Wilcoxon signed-rank test + paired bootstrap CI of the differences.
- **one-sample** — either side is a fixed point (zero seed variance, so a paired test is
  degenerate). The ensemble is characterised (mean ± std + bootstrap CI) and its distance
  from the fixed point is reported. No Wilcoxon.

`--allow-seed-mismatch` controls the paired-mode safety check:

| Flag | Default behavior | With the flag |
| --- | --- | --- |
| `--allow-seed-mismatch` | Paired mode **fails loudly** (`SeedMismatchError`) if the two arms don't share an identical seed set — re-run the missing seeds. | Silently intersect and test over the shared seeds (still aligned by value). |

### Benchmark selection

| Flag | Default | Meaning |
| --- | --- | --- |
| `--benchmarks a,b,c` | all present | Comma-separated **include** list; only these benchmark folders are parsed/plotted. |
| `--exclude a,b` | none | Comma-separated **exclude** list, applied after include. |

Benchmark names: `faitheval`, `truthfulqa`, `halueval`, `ragtruth`, `harness`.

### Toggles

| Flag | Effect |
| --- | --- |
| `--no-compare` | Aggregate each arm only; skip cross-arm comparison. The single-ensemble path (also implied when fewer than 2 arms are given). |
| `--no-plot` | Skip figures (no matplotlib needed). |
| `--out DIR` | Analysis output dir (default `outputs/analysis`). |
| `--rng-seed N` | Bootstrap RNG seed (default `0`) — makes CIs reproducible. |

### Overriding the primary metric — `--primary-map`

Each benchmark has a **primary** ("headline") metric that drives the default comparisons
and the main plots. The built-in defaults (from [`parse.py`](parse.py)):

| Benchmark | Primary metric |
| --- | --- |
| faitheval | per-task `accuracy` (the synthesized `mean` task is excluded) |
| halueval | per-task `accuracy` (ditto) |
| ragtruth | `overall` / `hallucination_rate` |
| truthfulqa | `MC1`, `MC2` |
| harness | `truthfulqa_mc1` / `truthfulqa_mc2`, metric `acc` |

Override with a YAML or JSON file: `--primary-map my_primary.yaml`. Shape:

```yaml
# A benchmark listed here uses ONLY its listed entries; unlisted benchmarks keep the
# built-in default. "metric" matches that metric on any task; "task:metric" matches exactly.
faitheval: ["accuracy"]                       # accuracy on every faitheval task
harness:   ["truthfulqa_mc2:acc"]             # only mc2 acc
truthfulqa: ["MC2"]
```

### Producing runs first — `--run-evals`

Drive the Hydra launcher's `--multirun` to *generate* run dirs before analysing them —
no separate launch step.

| Flag | Meaning |
| --- | --- |
| `--run-evals` | Enable this mode. Requires `--models`. |
| `--models id1,id2` | Comma-separated model ids swept as `model.id=id1,id2` (one numbered run dir each). |
| `--eval-sweep-dir DIR` | Hydra `sweep.dir` (default `outputs/analysis_sweep`); becomes `<DIR>/0`, `<DIR>/1`, … |
| `--eval-extra ...` | Extra Hydra overrides forwarded to the launcher. `nargs=REMAINDER` — **must come last** on the command line. |
| `--dry-run` | Print the launcher command without running it. |

If you don't also pass `--arm`, each numbered sweep subdir becomes an arm named
`model0`, `model1`, …. This path needs the launcher + per-benchmark venvs (it is not
exercised by the offline test suite).

```bash
python -m analysis.cli --run-evals --models ckptA,ckptB --reference model0 \
    --eval-extra run='[faitheval,harness]' num_samples=50
```

---

## `analysis.plot` parameters (standalone re-plot)

`python -m analysis.plot` (or [`run_plots.sh`](run_plots.sh)) reloads persisted
artifacts and re-renders figures with **no eval attached** — restyle long after the runs
finished.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--from DIR` (required) | — | Analysis dir containing `records.jsonl` (+ optional `comparisons.json`). |
| `--out DIR` | `<from>/figures` | Figure output dir. |
| `--reference NAME` | none | Reference label for delta plots. |
| `--benchmarks a,b` | all | Include list (drops others from every figure). |
| `--exclude a,b` | none | Exclude list. |

---

## Outputs (under `--out`, default `outputs/analysis/`)

| File | Written by | Contents |
| --- | --- | --- |
| `records.jsonl` | always | The long-form `MetricRecord` dump — one line per scalar. The reload point for standalone plotting. |
| `aggregate.json` | always | Per-arm, per-key mean / std / 95% CI / per-seed values. |
| `aggregate.tex` | always | Booktabs LaTeX table of the primary metrics (thesis appendix). |
| `comparisons.json` | if comparing | One row per arm-vs-reference comparison: regime, raw + signed delta, and the paired-test / one-sample block. |
| `comparisons.tex` | if comparing | LaTeX table: signed Δ + Wilcoxon p (paired) or CI (one-sample). |
| `figures/*.png` + `*.pdf` | if plotting | See below. Every figure is written as both a 150-dpi PNG and a vector PDF. |

### Figures (`figures/`)

| Figure | Function | Shows |
| --- | --- | --- |
| `<bench>.png` | `plot_benchmark` | One benchmark, arms side by side on each primary (task, metric), CI error bars + individual seed points. |
| `<bench>_tasks.png` | `plot_tasks` | Per-task grouped bars within a benchmark for one metric. |
| `cross_benchmark_panels.png` | `plot_cross_benchmark_panels` | Small multiples — one panel per benchmark on its **native** scale (no cross-scale mixing), primary-metric mean per arm. |
| `ranked_deltas.png` | `plot_ranked_deltas` | Direction-normalized signed delta of each arm vs. reference, sorted. Positive = improvement even for `hallucination_rate` (lower-is-better metrics are flipped). |
| `paired_<...>.png` | `plot_paired_deltas` | Per-seed deltas + mean + CI + Wilcoxon stars. **Paired regime only** — no-ops with a warning for one-sample comparisons. |

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

## Worked examples

```bash
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
```
