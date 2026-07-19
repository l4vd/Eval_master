# harness-eval

TruthfulQA evaluation through [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(`lm_eval`) — English **and** multilingual, using the harness the way public
leaderboards do.

This is a **second, independent** TruthfulQA number next to `TruthfulQA-reproduce`
(which runs the original authors' scripts). The harness is the community standard
for reporting TruthfulQA, so both are kept; expect their MC1/MC2 to differ, because
the prompt and scoring plumbing are not identical. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start

```bash
# From the Eval_master launcher (preferred): harness only, tiny model, CPU smoke test
./run_all.sh run='[harness]' model.dtype=float32 num_samples=5 harness.tasks='[truthfulqa_mc1]'

# Your own checkpoint or LoRA adapter, full English TruthfulQA
./run_all.sh run='[harness]' model.id=/path/to/final_checkpoint

# Standalone (inside this folder's venv)
python src/run_eval.py --model-id Qwen/Qwen2.5-0.5B-Instruct --tasks truthfulqa --device -1
```

## Tasks

The three benches are lm_eval **tags**; each fans out to per-subtask scores (no
aggregate). Pass any mix to `--tasks` / `harness.tasks`; glob patterns work.

| name | expands to | metrics |
| --- | --- | --- |
| `truthfulqa` | `truthfulqa_mc1`, `truthfulqa_mc2`, `truthfulqa_gen` (English) | acc, bleu/rouge |
| `truthfulqa_multilingual` | `truthfulqa_<lang>_mc1` / `_mc2`, 31 okapi languages | acc |
| `truthfulqa-multi` | `truthfulqa-multi_{mc1,mc2,gen}_{en,es,ca,eu,gl}` (HiTZ) | acc, bleu |

Discover what the installed lm_eval actually registers (no model is loaded):

```bash
python src/run_eval.py --list-tasks truthfulqa
```

## Key flags

| flag | launcher key | meaning |
| --- | --- | --- |
| `--model-id` | `model.id` | Hub id, local full-model dir, or PEFT/LoRA adapter |
| `--base-model-id` | `model.base_model_id` | base for a LoRA adapter (else read from its config) |
| `--tokenizer-id` | `model.tokenizer_id` | tokenizer, if not saved with the checkpoint |
| `--tasks` | `harness.tasks` | lm_eval task/tag names (glob ok) |
| `--limit` | `num_samples` | first N docs **per task** (non-comparable to published scores) |
| `--num-fewshot` | `harness.num_fewshot` | few-shot count (null = task default; TruthfulQA is 0-shot) |
| `--batch-size` | `harness.batch_size` | int, or `auto` / `auto:N` |
| `--device` | `model.device_index` | `-1`=CPU, `0`=cuda:0, or a device string |
| `--dtype` | `model.dtype` | `bfloat16` \| `float16` \| `float32` |
| `--apply-chat-template` | `harness.apply_chat_template` | see **Prompt format** |
| `--fewshot-as-multiturn` | `harness.fewshot_as_multiturn` | requires `--apply-chat-template` |
| `--log-samples` | `harness.log_samples` | write per-document `samples.jsonl` |

## Prompt format (report which you used)

`--apply-chat-template` is **off** by default: the published, completion-style
protocol that leaderboard numbers use — and lm_eval's own default. Turning it on
renders prompts with the model's chat template (comparable to
`TruthfulQA-reproduce`'s `prompt_style: chat` and to the suite's other benchmarks).

The choice **materially moves the scores** (the launcher's README-runner.md reports
MC2 0.06 completion vs 0.26 chat on a Qwen2.5-0.5B smoke test), so keep it fixed
across the models you compare. The resolved value is recorded in `summary.json`.

## Outputs

Written to `--output-dir` (the launcher points this at `outputs/<date>/<time>/harness/`):

- `summary.json` — flat one-row-per-(task, metric, filter) results plus a
  provenance header (`model_args`, `resolved_tasks`, `apply_chat_template`,
  `limit`, `lm_eval_version`, …).
- `samples.jsonl` — one record per evaluated document (with `--log-samples`),
  UTF-8, non-Latin text kept literal.
- `lm_eval_results.json` — lm_eval's raw result dict, minus the per-sample payload.

## Install

Each benchmark owns its virtualenv (incompatible stacks). From the launcher:

```bash
./setup_envs_local.sh --venv-root "$LOCALAPPDATA/eval-venvs"   # laptop / Windows
./setup_envs_HPC.sh                                            # cluster (pyproject-HPC.toml)
```

Or standalone with `uv`:

```bash
uv sync --extra dev            # honours the committed uv.lock
uv run pytest -q               # offline unit tests (task tests need lm_eval, installed here)
```

Requires Python **≥ 3.10** (a deviation from the sibling benchmarks; see
ARCHITECTURE.md — lm_eval 0.4.10+ needs it, and `>=3.9` silently backtracks to an
old lm_eval).

## Install (HPC)

`setup_envs_HPC.sh` swaps in `pyproject-HPC.toml` (torch 2.2.2 / transformers 4.41 /
Python 3.12) and does a fresh `uv sync`. The three task groups need **no** extra
metrics deps and pull **no** TensorFlow (BLEURT is disabled upstream). Pre-download
the TruthfulQA datasets into the HF cache; the compute nodes have no internet.
⚠️ lm_eval 0.4.12 on transformers 4.41 is metadata-clean but unverified at runtime —
fall back to `lm_eval==0.4.8` if it breaks. See ARCHITECTURE.md "HPC".

## Credits & citations

This module is only a thin adapter. All the evaluation machinery — task
definitions, prompting, and metrics — belongs to **EleutherAI's
lm-evaluation-harness** and to the authors of the TruthfulQA benchmarks it runs.
When you report results, cite **those** projects, not this wrapper. This module
targets lm_eval **v0.4.12**.

**lm-evaluation-harness** — the framework (MIT-licensed), Gao et al.:

```bibtex
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = jul,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```

**TruthfulQA** (`truthfulqa`) — the original benchmark, Lin, Hilton & Evans:

```bibtex
@inproceedings{lin-etal-2022-truthfulqa,
  title     = {{T}ruthful{QA}: Measuring How Models Mimic Human Falsehoods},
  author    = {Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle = {Proc. 60th Annual Meeting of the ACL (Volume 1: Long Papers)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2109.07958}
}
```

**Okapi** (`truthfulqa_multilingual`, 31 machine-translated languages) — Lai et al.:

```bibtex
@article{lai2023okapi,
  title   = {Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback},
  author  = {Lai, Viet Dac and Nguyen, Chien Van and Ngo, Nghia Trung and Nguyen, Thuat and Dernoncourt, Franck and Rossi, Ryan A. and Nguyen, Thien Huu},
  journal = {arXiv preprint arXiv:2307.16039},
  year    = {2023}
}
```

**TruthfulQA-multi** (`truthfulqa-multi`, 5 human-translated languages; Apache-2.0) —
HiTZ Center, Calvo Figueras et al.:

```bibtex
@article{calvo2025truthknowsnolanguage,
  title   = {Truth Knows No Language: Evaluating Truthfulness Beyond English},
  author  = {Calvo Figueras, Blanca and Sagarzazu, Eneko and Etxaniz, Julen and Barnes, Jeremy and Gamallo, Pablo and De Dios Flores, Iria and Agerri, Rodrigo},
  journal = {arXiv preprint arXiv:2502.09387},
  year    = {2025}
}
```

The default smoke-test model is [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
(Qwen Team, Alibaba). Consult each project's own repository for its license and
terms of use.
