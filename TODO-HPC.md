# Getting the full eval framework running on the HPC, offline

Scope: the launcher (`Eval_master`) plus all five benchmarks
(`FaithEval-reproduce`, `TruthfulQA-reproduce`, `HaluEval-reproduce`,
`RAGTruth-reproduce`, `harness-eval`). Compute nodes have **no internet**, so
everything (packages, models, datasets) must be staged from a login node first.

## 0. What already exists (don't redo this)

- Every project already has a `pyproject-HPC.toml` pinned to the cluster's
  known-good stack (Python 3.12, torch 2.2.2, transformers 4.41 — the same
  pins `topollm` / `SP-DPO-Base` use).
- `setup_envs_HPC.sh` already automates swapping in each `pyproject-HPC.toml`
  and `uv sync`-ing the launcher + all five benchmarks.
- Every benchmark README has an "Install (HPC)" section and a short offline
  note. `SP-DPO-Base/README-HPC.md` is the canonical recipe these all point
  back to — reuse its module names / mirror pattern rather than reinventing
  one.
- The launcher (`run_all.sh` / `run_benchmarks.py`) is already
  stack-agnostic; it just resolves each benchmark's own venv and shells out.

What's genuinely missing is: (1) the PyPI mirror URL plugged into 6 files,
(2) every model/dataset pre-downloaded into an HF cache the compute nodes can
reach, (3) a SLURM job script (none exists yet), and (4) a verification pass,
since the HPC pins are documented as "unverified at runtime."

## 1. Get the repo onto the cluster

- [ ] Confirm cluster account + SSH access, and that your SSH key is on
      GitHub (remote is `git@github.com:l4vd/Eval_master.git`).
- [ ] `git clone` from a **login node** (has internet) into project/scratch
      storage, not `$HOME` — checkpoints + HF cache are multi-GB and home
      quotas are typically small. Match wherever `SP-DPO-Base` already lives
      on the cluster (e.g. `/gpfs/project/$USER/...`) so relative paths and
      conventions line up.
- [ ] `module load Python/3.12.3 uv/0.10.2 CUDA/12.6.1` (adjust to whatever
      module names actually exist on this cluster — check with `module spider`).

## 2. Wire up the offline PyPI mirror

Every `pyproject-HPC.toml` (launcher + 5 benchmarks = 6 files) ships a
**commented-out** mirror block:

```toml
# [[tool.uv.index]]
# name = "cluster-mirror"
# url = "<CLUSTER_PYPI_MIRROR_URL>"
# default = true
```

- [ ] Get the cluster's internal PyPI mirror URL (ask HPC admins, or read it
      out of `topollm`'s working config if you have access to that repo —
      every HPC file here references it as the known-good baseline).
- [ ] Uncomment + fill in that block in all 6 `pyproject-HPC.toml` files:
      `Eval_master/pyproject-HPC.toml`, `FaithEval-reproduce/`,
      `TruthfulQA-reproduce/`, `HaluEval-reproduce/`, `RAGTruth-reproduce/`,
      `harness-eval/`.
- [ ] Without this, `uv sync` on a compute node (or a login node with a
      locked-down proxy) will simply fail to resolve packages.

## 3. Install everything

- [ ] From the `Eval_master` root: `./setup_envs_HPC.sh` — this backs up each
      `pyproject.toml`, swaps in `pyproject-HPC.toml`, drops the stale
      `uv.lock`, and runs `uv sync --extra dev` for the launcher and all five
      benchmarks in one go.
- [ ] Run each benchmark's own test suite once envs exist, to catch a bad
      resolution before you're mid-SLURM-job:
      ```bash
      for d in FaithEval-reproduce TruthfulQA-reproduce HaluEval-reproduce RAGTruth-reproduce harness-eval; do
        (cd "$d" && ./.venv/bin/python -m pytest -q)
      done
      ```

## 4. Known install risks — verify these specifically

- [ ] **`harness-eval` / lm_eval 0.4.12**: pinned against transformers 4.41,
      but documented as "metadata-clean but unverified at runtime" — 0.4.12's
      `HFLM` targets much newer transformers. Run `harness-eval`'s tests and
      a real smoke eval; if `HFLM` breaks, fall back by editing
      `harness-eval/pyproject-HPC.toml` to `lm_eval[hf]==0.4.8`, delete its
      `uv.lock`, and re-sync (see `harness-eval/ARCHITECTURE.md` "HPC").
- [ ] **`rouge-score`** (pulled by lm_eval for the `truthfulqa_gen` subtask,
      part of the default `truthfulqa` tag) is **sdist-only** — confirm the
      mirror serves source distributions, not just wheels, or drop
      `truthfulqa_gen` by not requesting it (lm_eval tags aren't
      subtask-filterable here without touching `harness.tasks`, so the
      practical fix if the mirror can't build it is to test this early).
- [ ] BLEURT/TensorFlow and legacy BLEU/ROUGE metrics are **deliberately
      excluded** from `TruthfulQA-reproduce`'s HPC build (`legacy-metrics`
      extra). Don't set `truthfulqa.metrics` to include `bleu`/`rouge`/`bleurt`
      on the cluster unless you separately provision that stack.
- [ ] No vLLM, no TGI anywhere in the HPC builds (vLLM needs torch ≥2.4,
      which conflicts with the pinned 2.2.2) — this is intentional and
      already reflected in every benchmark's model loader (HF backend only).

## 5. Pre-download everything into the HF cache (on a login node, with internet)

Compute nodes are offline, so anything `from_pretrained`/`load_dataset` would
normally fetch must already be cached. Set one shared cache dir first (put it
on project/scratch storage, matching wherever `SP-DPO-Base`/`topollm` already
point `HF_HOME`, not `$HOME`):

```bash
export HF_HOME=/gpfs/project/$USER/hf_cache   # adjust to this cluster's convention
```

Per-benchmark, only fetch what you'll actually run:

- [ ] **The model(s) you're evaluating** — your `SP-DPO-Base` checkpoints.
      These are local paths, not Hub ids, so there's nothing to
      `huggingface-cli download`; just make sure the checkpoint directories
      (with their own tokenizer files — see `harness-eval/ARCHITECTURE.md`
      "design decision 2" on why adapters must keep their own tokenizer) are
      already on the cluster filesystem the compute node can see, and pointed
      to via `conf/model/<name>.yaml` (`model.id`).
- [ ] **FaithEval-reproduce** — only the task(s) in `faitheval.tasks`:
      ```bash
      huggingface-cli download Salesforce/FaithEval-unanswerable-v1.0 --repo-type dataset
      huggingface-cli download Salesforce/FaithEval-inconsistent-v1.0 --repo-type dataset
      huggingface-cli download Salesforce/FaithEval-counterfactual-v1.0 --repo-type dataset
      ```
- [ ] **TruthfulQA-reproduce** — the question CSV is already committed in the
      repo (no download). Default `truthfulqa.metrics: [mc]` needs nothing
      extra. Only if you turn on `judge`/`info` metrics:
      ```bash
      huggingface-cli download allenai/truthfulqa-truth-judge-llama2-7B
      huggingface-cli download allenai/truthfulqa-info-judge-llama2-7B
      ```
      (two 7B Llama-2 models — large; skip unless you need those metrics).
- [ ] **HaluEval-reproduce** — its 35K eval samples are committed locally
      under `HaluEval-reproduce/data/*.json`; nothing to download, the judge
      *is* the evaluated model (`backend: hf`).
- [ ] **RAGTruth-reproduce** — its dataset is committed locally
      (`RAGTruth-reproduce/dataset/{source_info,response}.jsonl`). Only the
      detector needs fetching, unless you supply your own or run
      `ragtruth.gold_f1=true` against an existing detector's output:
      ```bash
      huggingface-cli download CodingLL/RAGTruth_Eval
      ```
- [ ] **harness-eval** — lm_eval resolves TruthfulQA (and the multilingual
      variants) straight from the HF `datasets` hub at run time; the exact
      dataset repo ids live inside the *installed* `lm_eval` package's task
      YAMLs, not in this repo, so don't hand-guess them. Instead, prime the
      cache by actually running a tiny CPU smoke test on the login node with
      internet still enabled:
      ```bash
      cd harness-eval && ./.venv/bin/python src/run_eval.py \
          --model-id Qwen/Qwen2.5-0.5B-Instruct --tasks truthfulqa \
          --device -1 --limit 2 --output-dir /tmp/harness-smoke
      ```
      This populates `$HF_HOME` organically; compute nodes then reuse it.
      Default `harness.tasks: [truthfulqa]` only needs the English dataset —
      only do this for `truthfulqa_multilingual` (31 okapi languages) or
      `truthfulqa-multi` (5 HiTZ languages) if you actually intend to run
      those tags, since each fans out into dozens of dataset repos.
- [ ] Any **base model** a LoRA/PEFT adapter checkpoint needs (`base_model_id`
      in `conf/model/`), if it isn't already cached from training.

## 6. Offline env vars (compute nodes)

Export these in the SLURM job, after step 5 has populated `$HF_HOME`:

```bash
export HF_HOME=/gpfs/project/$USER/hf_cache   # same path used to pre-download
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

If any of these fire on a cache miss, the job will fail fast with a clear
"couldn't reach the Hub" error rather than hanging — that's useful for
catching a missed download in step 5.

## 7. Point the launcher at your checkpoint(s)

- [ ] Copy `conf/model/_template.yaml` to `conf/model/<name>.yaml` and fill
      in `id` (and `base_model_id`/`tokenizer_id` for a LoRA adapter), or
      just override on the CLI: `./run_all.sh model.id=/path/to/checkpoint`.
- [ ] Decide `dtype` per the actual GPU: `bf16` needs Ampere+ (A100); on
      V100 nodes use `model.dtype=float16` (same tradeoff `SP-DPO-Base`
      documents for training).
- [ ] Fix — and record — the protocol choices that materially move scores
      across every model you'll compare: `truthfulqa.prompt_style`
      (`chat` vs `completion`) and `harness.apply_chat_template`
      (`false` = leaderboard-comparable, `true` = matches the other four
      benchmarks). Both are written into each run's output for provenance,
      but decide *before* running the whole checkpoint set so results are
      comparable to each other.

## 8. Write the SLURM job script

None exists in this repo yet — write one. Rough shape:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=eval-suite
#SBATCH --gres=gpu:1
#SBATCH --time=...
#SBATCH --output=logs/%x-%j.out

module load Python/3.12.3 uv/0.10.2 CUDA/12.6.1

export HF_HOME=/gpfs/project/$USER/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd /gpfs/project/$USER/Eval_master
./run_all.sh model.id=/path/to/checkpoint model.dtype=bfloat16
```

- [ ] Check whether all five benchmarks comfortably share one GPU allocation
      sequentially (the launcher runs them one after another, each loading
      its own model into memory and releasing it) or whether you'd rather
      split into per-benchmark jobs/array tasks for parallelism and shorter
      individual walltimes.
- [ ] Decide `--time` per benchmark set — `harness` with multilingual tags
      fans out into 60+ tasks and will dominate the walltime budget if
      enabled.

## 9. Smoke-test before the real run

- [ ] Dry run on the login node first (no GPU/queue needed) to sanity-check
      every generated command:
      ```bash
      ./run_all.sh model.dtype=float32 model.device_map=cpu num_samples=5 dry_run=true
      ```
- [ ] Then a small **real** interactive-session GPU test, one benchmark at a
      time, before committing to a full batch job:
      ```bash
      srun --gres=gpu:1 --pty bash
      ./run_all.sh run='[faitheval]' num_samples=5
      ```
      Remember `harness.num_samples` is lm_eval's `--limit`, applied **per
      task** — even a small value fans out fast once multilingual tags are
      in play.
- [ ] Once one benchmark works end-to-end offline, run the full `run='[...]'`
      list for real.

## 10. After it runs

- [ ] `outputs/` is git-ignored — copy results off scratch storage (it's
      often purged on a rolling window) before the run ages out.
- [ ] Keep a note of exactly which `prompt_style` / `apply_chat_template` /
      dtype you used per run, since those aren't otherwise obvious from the
      output files alone in isolation from `summary.json`/`run_config.csv`.
