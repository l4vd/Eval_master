# PBS job scripts (`qsub_*.sh`)

A reference for every script in the project that writes PBS jobs: what it does, how to call
it, and its settings. The procedure itself (which runs, in which order, and what they decide)
is in `SP-DPO-Base/EXPERIMENT_PROCEDURE.md` §1 and §5. The optional overview is described in
[RUNNING_NEW_OPTIONS.md §7](RUNNING_NEW_OPTIONS.md).

## 1. Overview

| Script | Repo | Job type | What it does |
|---|---|---|---|
| [`scripts/qsub_train.sh`](../../SP-DPO-Base/scripts/qsub_train.sh) | SP-DPO-Base | GPU, one per arm (+ optional profile job) | Trains the §1 arms. One job per arm runs all of that arm's seeds. |
| [`analysis/qsub_eval_day.sh`](analysis/qsub_eval_day.sh) | Eval_master | GPU, one per ensemble | Evaluates every training ensemble of one day (`outputs/<DATE>/*/seed_*`). |
| [`analysis/qsub_comparisons.sh`](analysis/qsub_comparisons.sh) | Eval_master | CPU, one job | Runs the §5.7 comparisons (Claims A/B/C, sign, LR sensitivity) and the family tables. |
| [`analysis/qsub_overview_day.sh`](analysis/qsub_overview_day.sh) | Eval_master | GPU per ensemble + optional base job + CPU derive job | *Optional overview.* Evaluates one day under all scoring variants. |
| [`analysis/qsub_overview_compare.sh`](analysis/qsub_overview_compare.sh) | Eval_master | CPU, one job | *Optional overview.* Derives the CPU variants and writes descriptive plots and tables per arm set. |
| [`analysis/qsub_lib.sh`](analysis/qsub_lib.sh) | Eval_master | — | Shared helpers only; it is sourced, never run. |

The official path is `qsub_train.sh` → `qsub_eval_day.sh` → `qsub_comparisons.sh`.
The optional overview replaces the last two steps with `qsub_overview_day.sh` →
`qsub_overview_compare.sh`. It is descriptive and not pre-registered, and it never changes
§5.4–§5.8.

## 2. Things all scripts share

**Modes.** The mode is the first positional argument. The `*_day.sh` scripts take it second,
after the day directory.

| Mode | Effect |
|---|---|
| `print` (default) | Writes the `.pbs` job scripts and prints the `qsub` lines. Nothing is submitted. |
| `check` | Like `print`, then tests the jobs on the login node without loading a model. What exactly it runs is listed per script below. |
| `submit` | Writes the jobs and runs `qsub`. It needs `QSUB_RES`. |

**Settings are environment variables**, set in front of the command (`VAR=value ./script ...`).
Apart from the mode and the day directory, the scripts take no `--flags`.

**Common variables**

| Variable | Meaning |
|---|---|
| `QSUB_RES` | The resource flags passed to `qsub`, as one string, e.g. `"-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=24:00:00 -A <project>"`. The script splits it into separate flags. It is required for `submit`. |
| `JOB_NAME` | The `qsub -N` name. |
| `JOB_DIR` | Where the `.pbs` files are written. It is relative to the repo root. |

**Paths.** Every script `cd`s to its own repo root first, so it can be called from anywhere.
Relative paths such as `OUT_ROOT` are relative to that repo root, not to your shell.

**Job environment.** Each job loads the HPC modules, sources `.env`, and activates the repo's
`.venv`. Two scripts differ in where the job runs:

- `qsub_eval_day.sh` and `qsub_comparisons.sh` always `cd` to the fixed path
  `/gpfs/project/ladan102/git-source/SP_DPO/Eval_master/`.
- `qsub_overview_*.sh` `cd` to `EVAL_MASTER_DIR`, which defaults to the checkout you ran them
  from.
- `qsub_train.sh` `cd`s to the SP-DPO-Base checkout you ran it from.

**Arm layout.** Training writes `SP-DPO-Base/outputs/<DATE>/<HH-MM-SS>_<arm>_ensemble/seed_*`.
Evaluation drops the time prefix and writes `<OUT_ROOT>/<arm>_ensemble<SUFFIX>/seed_*`.
The comparison scripts look for arms at exactly that path, and for base at
`<OUT_ROOT>/<BASE_DIR>`.

**Two similar names.**

- `VARIANT` in `qsub_train.sh` is the *difficulty profile* the curriculum trained on
  (e.g. `ups0_k16_eta055`).
- `VARIANTS` in the eval and overview scripts are *scoring variants* of the evaluation
  (see `analysis/variants.py`).

If you train with a second profile, keep its evaluation apart with its own `SUFFIX` or
`OUT_ROOT`.

---

## 3. `SP-DPO-Base/scripts/qsub_train.sh`

### What it does

The script writes one PBS job per training arm of EXPERIMENT_PROCEDURE §1. Each job runs all
of that arm's seeds one after another:

- the i.i.d. arms run through `run_pipeline_multiseed.sh`;
- the curriculum arms (`off_sp_*`) run through `run_off_sp_dpo_multiseed.sh`.

With `PROFILE=materialize` or `PROFILE=run`, the script first writes a profiling job. The
curriculum jobs then wait for it (`-W depend=afterok:<id>`) and never profile themselves.
`ALPHA_REPLAY`, `REPLAY_ACCUMULATE` and `CE_LAMBDA` are unset in every job on purpose (§1 Step 4).

`check` runs each job with `--dry-run`. The arm jobs then compose the Hydra config with
`scripts/show_config.py` and print `name_or_path`, `data_dir`, `lr_schedule` and
`num_train_epochs`. The profile job only checks that the anchor model or the raw store exists.

### Example

```bash
cd /gpfs/project/$USER/git-source/SP_DPO/SP-DPO-Base

# print the jobs for all arms
./scripts/qsub_train.sh

# check two arms and the curriculum group
ARMS="dpo orpo curriculum" ./scripts/qsub_train.sh check

# submit: re-materialize a profile, then train dpo (other LR) and every curriculum arm
PROFILE=materialize VARIANT=ups0_k8 MATERIALIZE_ARGS="--upsilon 0.0 --k0 8 --eta 0.55" \
ARMS="dpo curriculum" ARGS_dpo="training.learning_rate=5e-5" \
SEEDS_off_sp_dpo_legacy="42 1337 2024" \
QSUB_RES="-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=24:00:00 -A <project>" \
QSUB_RES_profile="-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=04:00:00 -A <project>" \
  ./scripts/qsub_train.sh submit
```

### Arguments

| Argument | Meaning |
|---|---|
| `$1` | Mode: `print` (default), `check` or `submit`. |

### Settings

**Arm selection**

| Variable | Default | Meaning |
|---|---|---|
| `ARMS` | `all` | A space-separated list of arms and/or groups. |

The groups are:

- `iid`: `sft dpo orpo dpo_cosine orpo_cosine`
- `curriculum`: `off_sp_dpo_asc off_sp_dpo_shuf off_sp_dpo_desc off_sp_dpo_legacy off_sp_orpo_asc off_sp_orpo_shuf off_sp_orpo_desc`
- `all`: both groups.

An unknown arm name is an error.

| Arm | Experiment preset |
|---|---|
| `sft` | `sft_ragtruth_all` |
| `dpo` / `orpo` | `dpo_ragtruth_all` / `orpo_ragtruth_all` |
| `dpo_cosine` / `orpo_cosine` | `dpo_ragtruth_all_cosine` / `orpo_ragtruth_all_cosine` |
| `off_sp_dpo_asc` / `off_sp_orpo_asc` | `off_sp_{dpo,orpo}_ragtruth_all` |
| `off_sp_*_shuf` | `..._shuffled` |
| `off_sp_*_desc` | `..._descending` |
| `off_sp_dpo_legacy` | `off_sp_dpo_ragtruth_all_legacy` |

The arm name also becomes `METHOD`, i.e. the name of the ensemble directory.

**Per-arm overrides.** Replace `<arm>` with the arm name.

| Variable | Default | Meaning |
|---|---|---|
| `ARGS_<arm>` | empty; `ARGS_sft="training.num_train_epochs=2"` | Hydra overrides for `train.py`, separated by spaces. A single override must not contain a space. Setting `ARGS_sft=` (empty) removes the SFT default. |
| `SEEDS_<arm>` | `SEEDS` | The seed list for this arm, e.g. a smaller prefix for an optional arm. |
| `QSUB_RES_<arm>` | `QSUB_RES` | The resources for this arm's job. |

**Profiling.** These settings only matter when a curriculum arm is selected.

| Variable | Default | Meaning |
|---|---|---|
| `PROFILE` | `reuse` | `reuse` trains on `PROFILED_DIR` as it is, and `PROFILED_DIR/train.jsonl` must exist. `materialize` re-bins `PROFILE_STORE/raw` into `variants/VARIANT`; this is cheap. `run` scores the data with `ANCHOR_CKPT` into `PROFILE_STORE`, then materializes `VARIANT`. |
| `PROFILE_STORE` | `outputs/profiled/ragtruth_mixture_study` | The profile store: `raw/` and `variants/`. |
| `VARIANT` | `ups0_k16_eta055` | The profile variant under `variants/`. An empty value means the store's primary profile. `materialize` requires a value. |
| `MATERIALIZE_ARGS` | `--upsilon 0.0 --k0 16 --eta 0.55` | Arguments for `materialize_profile.py`. |
| `PROFILED_DIR` | `PROFILE_STORE/variants/VARIANT` | The directory the curriculum arms train on. You may only change it with `PROFILE=reuse`. |
| `QSUB_RES_profile` | `QSUB_RES` | The resources for the profiling job. |
| `ANCHOR_CKPT` | Qwen2.5-0.5B-Instruct | The anchor model used for scoring. |
| `PROFILE_SEED` | `42` | The seed used for profiling. |
| `PROFILE_MAX_LENGTH` | `4096` | `profiling.max_length`. |
| `N_BINS` | `10` | The number of difficulty bins. It is used for profiling and for `training.curriculum.n_bins`. |
| `PACING` | `staged` | `training.curriculum.pacing`. |
| `DATA` | `ragtruth_mixture` | The Hydra data config. |

**Shared settings**

| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | `42 1337 2024 7 99 123` | The six study seeds. |
| `BASE_MODEL` | `/gpfs/project/$USER_NAME/models/huggingface/hub/Qwen/Qwen2.5-0.5B-Instruct` | The model to train from. |
| `USER_NAME` | `ladan102` | Used in the model paths and exported to the job. |
| `QSUB_RES` | — | The default resources for every job. It is required for `submit`. |
| `JOB_NAME` | `Train` | The `qsub -N` name. |
| `JOB_DIR` | `logs/qsub_train/<YYYY-mm-dd_HH-MM-SS>` | Where the `profile.pbs` and `<arm>.pbs` files go. |

---

## 4. `Eval_master/analysis/qsub_eval_day.sh`

### What it does

For one training day, the script writes one GPU job per ensemble directory, i.e. per directory
that contains `seed_*`. Each job runs `run_eval_checkpoints.sh --resume` over all seeds of that
ensemble and writes to `<OUT_ROOT>/<arm>_ensemble<SUFFIX>/seed_*`.

If the same arm was trained twice that day, the second one keeps its time prefix in the
directory name.

Because of `--resume`, you can re-submit after a walltime kill. Seeds whose benchmarks already
wrote a summary are skipped. One job runs all six seeds one after another, so the walltime has
to cover the slowest arm × 6.

`check` runs each job here with `--dry-run`. This tests the modules, `.env`, `.venv` and the
checkpoint glob, and loads no model.

**Guards**

- `EXTRA` with modified-protocol keys (`scoring=constrained|both`, `decontam.enabled=true`,
  `counterfactual_mc`) is refused unless `OUT_ROOT` contains `modified`. Modified runs must go
  to their own root.
- With `VARIANTS` set, those keys and `strict_match=true` are refused in `EXTRA`, and so is a
  `variants=` override written by hand.

### Example

```bash
cd /gpfs/project/$USER/git-source/SP_DPO/Eval_master
DAY=/gpfs/project/$USER/git-source/SP_DPO/SP-DPO-Base/outputs/2026-09-10

# print, then check
./analysis/qsub_eval_day.sh "$DAY"
./analysis/qsub_eval_day.sh "$DAY" check

# submit the original protocol
QSUB_RES="-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=24:00:00 -A <project>" \
OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs \
  ./analysis/qsub_eval_day.sh "$DAY" submit

# modified protocols (§5.4 step 5): a separate root, only halueval + faitheval
OUT_ROOT=outputs/eval/v4-2_modified BENCHES=halueval,faitheval EXTRA="$EVAL_MODIFIED" \
QSUB_RES="..." ./analysis/qsub_eval_day.sh "$DAY" submit
# EVAL_MODIFIED="halueval.scoring=constrained halueval.decontam.enabled=true faitheval.tasks=[counterfactual_mc]"
```

### Arguments

| Argument | Meaning |
|---|---|
| `$1` (required) | The training day directory, `SP-DPO-Base/outputs/<DATE>`. |
| `$2` | Mode: `print` (default), `check` or `submit`. |

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `OUT_ROOT` | `outputs/eval/v4-2` | The eval root. For modified protocols it must contain `modified`. |
| `SUFFIX` | empty | Appended to each ensemble directory name, e.g. `-2epochs` or `-ups0_k8`. |
| `BENCHES` | `faitheval,halueval,harness` | The benchmarks to run. |
| `EXTRA` | empty | Hydra overrides passed on through `--launcher-extra`, separated by spaces. Brackets and quotes are respected, so `faitheval.tasks=[a, b]` stays one override. |
| `VARIANTS` | unset | Optional overview: `all`, or a comma list of names from `analysis/variants.py` such as `halueval.constrained,faitheval.mc`. It adds `variants=[...]` to the overrides. Each variant is written to its own `<bench>.<variant>/` directory next to the original one. |
| `EVAL_MASTER_REV` | from `$OUT_ROOT/EVAL_BLOCK.txt`, else git `HEAD` | The code version written to `launches.jsonl`. |
| `QSUB_RES` | — | Required for `submit`. |
| `JOB_NAME` | `DevSession` | The `qsub -N` name. |
| `JOB_DIR` | `logs/qsub_jobs/<DATE>` | One `<ensemble dir>.pbs` file per ensemble. |

---

## 5. `Eval_master/analysis/qsub_comparisons.sh`

### What it does

The script writes one CPU job that runs the §5.7 analyses with `run_analysis.sh` over the eval
directories that `qsub_eval_day.sh` wrote. If an arm is missing, every comparison that needs it
is left out, and the report lists both the missing arms and the comparisons it skipped.

If `ANALYSIS_ROOT/primary_confirmatory.yaml` does not exist, the job creates it with
`harness: ["truthfulqa_mc2:acc"]`.

| Output directory | Arms | Content |
|---|---|---|
| `claimA_dpo`, `claimA_orpo` | `off_sp_*_asc` vs `off_sp_*_shuf` | `mc2/` is the deciding test (with `--primary-map`); `all/` is context over `BENCHES`. |
| `sign_dpo`, `sign_orpo` | `off_sp_*_asc` vs `off_sp_*_desc` | `mc2/` and `all/`. |
| `claimB_dpo`, `claimB_orpo` | `off_sp_*_asc` vs `dpo` / `orpo` | `mc2/` and `all/`. |
| `claimC_dpo` | `off_sp_dpo_asc` vs `off_sp_dpo_legacy` | `all/` only; descriptive. |
| `lrsens_dpo`, `lrsens_orpo` | `*_cosine` vs `dpo` / `orpo` | `all/` only; descriptive. |
| `claimB_cos_dpo`, `claimB_cos_orpo` | `off_sp_*_asc` vs `*_cosine` | `all/` only; descriptive. |
| `table_dpo_family`, `table_orpo_family` | base + every evaluated arm of the family | Reference tables relative to base. |

For the descriptive rows, the arm with fewer seeds (legacy or cosine) sets the seed set. The
other arm is cut down to those seeds through symlinks in `ANALYSIS_ROOT/_subsets/`.

`check` only syntax-checks the job (`bash -n`) and lists every `run`/`mkdir`/`ln` line in it.
Nothing is executed.

### Example

```bash
cd /gpfs/project/$USER/git-source/SP_DPO/Eval_master

OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_comparisons.sh          # print
OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_comparisons.sh check    # list commands
QSUB_RES="-A DialSys -l select=1:ncpus=2:mem=16gb -l walltime=02:00:00" \
  OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_comparisons.sh submit
```

### Arguments

| Argument | Meaning |
|---|---|
| `$1` | Mode: `print` (default), `check` or `submit`. |

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `OUT_ROOT` | `outputs/eval/v4-2` | The eval root to read. It must exist. |
| `SUFFIX` | empty | Must match the `SUFFIX` used by `qsub_eval_day.sh`. |
| `BENCHES` | `faitheval,halueval,harness` | The benchmarks for the `all/` and table analyses. |
| `BASE_DIR` | `base/run_0` | Where base is, relative to `OUT_ROOT`. |
| `ANALYSIS_ROOT` | `outputs/analysis/<basename OUT_ROOT>` | Where the results go. |
| `QSUB_RES` | — | Required for `submit`. |
| `JOB_NAME` | `DevSession` | The `qsub -N` name. |
| `JOB_DIR` | `logs/qsub_jobs/compare` | The job file is `<basename OUT_ROOT><SUFFIX>.pbs`. |

---

## 6. `Eval_master/analysis/qsub_overview_day.sh` (optional overview)

### What it does

This is the overview counterpart of `qsub_eval_day.sh`. It is descriptive and not part of
§5.4–§5.8. It writes up to three kinds of jobs:

1. **One GPU job per ensemble.** This is `qsub_eval_day.sh` with `VARIANTS` set. Every variant
   is written to `<OUT_ROOT>/<arm>_ensemble<SUFFIX>/seed_*/<bench>.<variant>/`, and finished
   units are skipped.
2. **A base job** (`base.pbs`), only if `BASE_MODEL` is set. It runs `run_all.sh` for the base
   model into `<OUT_ROOT>/<BASE_DIR>` with the same variants.
3. **A CPU derive job** (`derive.pbs`). It runs `run_derive.sh` over the *whole* `OUT_ROOT`,
   which also covers seeds evaluated on other days. It then writes a status table to
   `JOB_DIR/overview_status.tsv`. You can run it while GPU jobs are still running: a unit that
   has no summary yet is skipped.

`check` runs every job here in dry-run mode. It prints the plans and status matrices and loads
no model.

### Example

```bash
cd /gpfs/project/$USER/git-source/SP_DPO/Eval_master
DAY=../SP-DPO-Base/outputs/2026-09-10

# check, with a base job
BASE_MODEL=/gpfs/project/$USER/models/huggingface/hub/Qwen/Qwen2.5-0.5B-Instruct \
  ./analysis/qsub_overview_day.sh "$DAY" check

# submit GPU jobs, the base job and the derive job
QSUB_RES="-l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=24:00:00 -A <project>" \
QSUB_RES_derive="-l select=1:ncpus=2:mem=16gb -l walltime=04:00:00 -A <project>" \
BASE_MODEL=... EXTRA="model.dtype=bfloat16 harness.batch_size=8" SUFFIX=-ups0_k8 \
  ./analysis/qsub_overview_day.sh "$DAY" submit

# CPU variants only, over an existing root: no day directory needed ("-")
DERIVE_ONLY=1 OUT_ROOT=outputs/eval/v4-2 QSUB_RES_derive="..." \
  ./analysis/qsub_overview_day.sh - submit
```

### Arguments

| Argument | Meaning |
|---|---|
| `$1` (required) | The training day directory, or `-` together with `DERIVE_ONLY=1`. |
| `$2` | Mode: `print` (default), `check` or `submit`. |

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `VARIANTS` | `all` | The scoring variants: `all`, or a comma list from `analysis/variants.py`. |
| `OUT_ROOT`, `SUFFIX`, `BENCHES` | as in `qsub_eval_day.sh` | Where the ensembles are written. |
| `EXTRA` | empty | Hydra overrides. Use the **same** ones the original runs used; otherwise every GPU unit counts as stale. |
| `RECOMPUTE_STALE` | unset | `1` re-runs stale GPU units (it adds `recompute_stale=true`). |
| `BASE_MODEL` | unset | The path of the base model. If it is unset, no base job is written. |
| `BASE_DIR` | `base/run_0` | Where base is written, relative to `OUT_ROOT`. |
| `DERIVE` | `1` | `0` writes no derive job. |
| `DERIVE_ONLY` | unset | `1` writes only the derive job. The day argument is then `-`. |
| `QSUB_RES` | — | The GPU resources. Required for `submit` unless `DERIVE_ONLY` is set. |
| `QSUB_RES_base` | `QSUB_RES` | The resources for the base job. |
| `QSUB_RES_derive` | `QSUB_RES` | The resources for the derive job (CPU). It is required for `submit` when `DERIVE` is not `0`. |
| `JOB_NAME` | `Overview` | The `qsub -N` name. |
| `JOB_DIR` | `logs/qsub_jobs/overview/<DATE>` (`.../derive` with `DERIVE_ONLY`) | Where the job files and the status table go. |
| `EVAL_MASTER_DIR` | this checkout | Where the base and derive jobs `cd` to. |
| `EVAL_MASTER_REV` | from `EVAL_BLOCK.txt`, else git | The code version written to `launches.jsonl`. |

---

## 7. `Eval_master/analysis/qsub_overview_compare.sh` (optional overview)

### What it does

This is the overview counterpart of `qsub_comparisons.sh`, and it writes one CPU job. The job
does three things in order:

1. It brings the CPU variants of `OUT_ROOT` up to date with `run_derive.sh`.
2. It writes `ANALYSIS_ROOT/status.tsv`.
3. For each arm set, it runs
   `run_analysis.sh --variants-overview --no-compare`. This shows every arm under every
   variant, as figures and a table, with no statistical tests.

It never writes a `comparisons.json`. The §5.7 tests and the §5.8 table stay with
`qsub_comparisons.sh`. Arms are looked up the same way as there.

| Arm set (`SETS`) | Output directories | Content |
|---|---|---|
| `claims` | `claimA_*`, `sign_*`, `claimB_*` | One overview per §5.7 pair. REF is the dashed line, and Δ/SD is ARM relative to REF. |
| `families` | `family_dpo`, `family_orpo` | Base plus every evaluated arm of one objective. |
| `all` | `all_arms` | Base plus every evaluated arm. With more than 8 arms, the extra ones are drawn in grey. |

`check` syntax-checks the job and runs it with `--dry-run`: it shows the derive plan, writes
the status table, and prints each overview command.

### Example

```bash
cd /gpfs/project/$USER/git-source/SP_DPO/Eval_master

OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh          # print
OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh check    # dry run
QSUB_RES="-A <project> -l select=1:ncpus=2:mem=16gb -l walltime=04:00:00" \
SETS="claims families all" \
  OUT_ROOT=outputs/eval/v4-2 SUFFIX=-2epochs ./analysis/qsub_overview_compare.sh submit

# tables only, derive already done
DERIVE=0 PLOT=0 QSUB_RES="..." ./analysis/qsub_overview_compare.sh submit
```

### Arguments

| Argument | Meaning |
|---|---|
| `$1` | Mode: `print` (default), `check` or `submit`. |

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `OUT_ROOT` | `outputs/eval/v4-2` | The eval root to read. It must exist. |
| `SUFFIX` | empty | Must match the eval run. |
| `BENCHES` | `faitheval,halueval,harness` | The benchmarks shown. |
| `BASE_DIR` | `base/run_0` | Where base is, relative to `OUT_ROOT`. |
| `SETS` | `claims families` | Any of `claims`, `families`, `all`, separated by spaces. |
| `VARIANTS` | `all` | The variants for the derive step and the status table. |
| `DERIVE` | `1` | `0` skips the derive step. |
| `PLOT` | `1` | `0` writes tables only (`--no-plot`). |
| `ANALYSIS_ROOT` | `outputs/analysis/<basename OUT_ROOT>_overview` | Where the results go. |
| `QSUB_RES` | — | Required for `submit`. |
| `JOB_NAME` | `Overview` | The `qsub -N` name. |
| `JOB_DIR` | `logs/qsub_jobs/overview` | The job file is `<basename OUT_ROOT><SUFFIX>.pbs`. |
| `EVAL_MASTER_DIR` | this checkout | Where the job `cd`s to. |

---

## 8. `Eval_master/analysis/qsub_lib.sh`

This file holds shared helpers for the Eval_master writers. It is sourced and does nothing on
its own.

- `split_overrides STRING` fills `extra_words` with the Hydra overrides in STRING. It splits
  only on whitespace outside brackets and quotes, and an unbalanced bracket or quote is an
  error.
- `hpc_header [MODULES]` prints the common job preamble: modules, `cd $EVAL_MASTER_DIR`,
  `.env`, `.venv`, and `PYTHONUNBUFFERED`.
- `eval_master_rev OUT_ROOT` returns the code version. It uses `$EVAL_MASTER_REV` if set, then
  the `eval_master` line of `OUT_ROOT/EVAL_BLOCK.txt`, then git `HEAD`.

The writers are tested by `analysis/tests/test_qsub_scripts.py`, which needs bash ≥ 4 (see
[analysis/README.md](analysis/README.md)).

---

## 9. End to end

```bash
# 1) train (SP-DPO-Base)
QSUB_RES="..." ./scripts/qsub_train.sh submit

# 2) evaluate the day (Eval_master), after the training jobs
QSUB_RES="..." OUT_ROOT=outputs/eval/v4-2 ./analysis/qsub_eval_day.sh ../SP-DPO-Base/outputs/<DATE> submit
#    optional: the modified protocols go to a separate root (see §4)

# 3) compare, after all eval jobs of all days
QSUB_RES="..." OUT_ROOT=outputs/eval/v4-2 ./analysis/qsub_comparisons.sh submit

# optional overview instead of 2) and 3)
QSUB_RES="..." QSUB_RES_derive="..." BASE_MODEL=... ./analysis/qsub_overview_day.sh ../SP-DPO-Base/outputs/<DATE> submit
QSUB_RES="..." ./analysis/qsub_overview_compare.sh submit
```

Submission does not chain across scripts. Only `qsub_train.sh` links its own jobs, the
profiling job and the curriculum jobs. Submit each later step once the jobs before it have
finished.
