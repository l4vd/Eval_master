# harness-eval — an lm-evaluation-harness TruthfulQA module

## Context

`Eval_master/` runs four benchmarks (FaithEval, TruthfulQA, HaluEval, RAGTruth) against one
model through a Hydra launcher. Its TruthfulQA numbers come from the **original authors'
scripts** (`TruthfulQA-reproduce/`), which is not how the field reports TruthfulQA: the
project's own theory notes state that _"modern papers almost never run the original author
scripts"_ and that TruthfulQA is _"standardly executed through … EleutherAI's
lm-evaluation-harness"_.

`harness-eval/` adds that missing, **leaderboard-comparable** number, and extends TruthfulQA
to multilingual settings the current suite cannot reach at all. It is a **parallel module**
to `TruthfulQA-reproduce`, not a replacement — the two answer different questions:

|                        | protocol                                       | comparable to                  |
| ---------------------- | ---------------------------------------------- | ------------------------------ |
| `TruthfulQA-reproduce` | `prompt_style: chat` (default)                 | the suite's other 3 benchmarks |
| `harness-eval`         | lm_eval defaults, `apply_chat_template: false` | published / leaderboard scores |

Expect their MC1/MC2 to differ. That is the point, and both get reported.

**Outcome:** `./run_all.sh run='[harness]' model.id=/path/to/final_checkpoint` produces
`outputs/<date>/<time>/harness/summary.json` with harness-protocol TruthfulQA scores, using
the same LoRA/checkpoint interface as every sibling.

**Decisions taken:** default `tasks: [truthfulqa]` (English; multilingual one override away) ·
`apply_chat_template: false` (lm_eval's own flag _and_ default — use the framework as upstream
intends) · launcher key `harness`.

**Verification status:** every lm_eval claim below was verified against upstream source at
tags `v0.4.5` / `v0.4.12` / `main`. `lm_eval` is not installed anywhere on this machine.
Items marked ⚠️ are unverified and must be checked during implementation.

---

## Step 0 — Blocking pre-code check (~5 min, do this first)

Our caps (`datasets>=2.20,<4`) resolve to **datasets 3.6.0** locally, which **removed
script-based dataset loading**; the HPC twin pins `datasets<3.0`, which still has it.
If either multilingual dataset is script-based it fails locally but works on the cluster —
an inverted, very confusing failure.

Check `alexandrainst/m_truthfulqa` (okapi) and `HiTZ/truthfulqa-multi` on the Hub: parquet
or loading script? If script-based, add `dataset_kwargs` or cap `datasets<3`. English
`truthful_qa` is parquet and unaffected — so this does not block the default task set, only
the multilingual opt-in.

---

## Design

Wrap lm_eval's **Python API** (`lm_eval.simple_evaluate`), not its CLI — upstream `main` has
already refactored `__main__.py` into a new `lm_eval._cli.HarnessCLI`, so the CLI is the less
stable surface, and shelling out would mean a nested subprocess plus post-processing a
timestamped output dir.

The module's job is exactly three translations. Everything else is lm_eval's.

1. **`model.id` → lm_eval `model_args`** (the PEFT/tokenizer resolution) — `model.py`
2. **task names → validated task list** — `tasks.py`
3. **lm_eval results → repo-convention `summary.json`** — `results.py`

Three findings drive the design; each is a correctness issue, not a preference.

### Finding 1 — `requires-python` must be `>=3.10` (deviation from the `>=3.9` convention)

lm_eval's floor rose over the 0.4.x line: `0.4.5 → >=3.8`, `0.4.8 → >=3.9`, `0.4.12 → >=3.10`
(and 0.4.12 moves torch/transformers/peft/accelerate from base deps into an `hf` extra).
With `requires-python = ">=3.9"`, a resolver must satisfy the floor for _every_ interpreter
≥3.9, so it **silently backtracks to ~0.4.9.x** — you would think you pinned latest-0.4.x and
get a two-year-old resolution.

Pin `requires-python = ">=3.10"` and `lm_eval[hf]>=0.4.12,<0.5`. Harmless in practice (the
cluster is 3.12). Document the deviation in `pyproject.toml` and `ARCHITECTURE.md`.
Note: **0.4.6 never existed.**

### Finding 2 — build `model_args` as a **dict**, not a string

The "decouple from `HFLM.__init__`" rationale for the string form is false: verified in
`lm_eval/api/model.py`, both `create_from_arg_string` and `create_from_arg_obj` terminate in
`cls(**kwargs)` — identical coupling. The string only adds a lossy parse, and
`simple_parse_args_string` does a bare `.split(",")` **with no escaping**, so a checkpoint
path containing a comma silently corrupts the args.

`simple_evaluate(model_args=...)` accepts `str | dict` in every 0.4.x version. Pass the dict.
Record the string form in `summary.json` for copy-paste provenance only.

### Finding 3 — lm_eval's native `peft=` breaks the sibling tokenizer contract

`HFLM._create_tokenizer` runs **before** `_create_model` and keys off `pretrained`, so
`pretrained=<base>,peft=<adapter>` loads the **base** tokenizer. But
[model.py:117](Eval_master/FaithEval-reproduce/src/faitheval/model.py#L117) is
`resolved_tokenizer_id = tokenizer_id or model_id` — the **adapter dir's** tokenizer — and its
comment states _"a checkpoint produced by this project's training pipeline always has its own
tokenizer saved alongside it (adapter or full)"_.

Left alone, DPO adapters would be scored with a different tokenizer here than in the other
four benchmarks — exactly the comparison the thesis rests on. So when `model.id` is an
adapter, pass `tokenizer=<adapter dir>` explicitly **if** it holds tokenizer files, else fall
back to the base. This is the module's core value-add and the top unit-test target.

### Version-drift guard: pass every behavior-bearing kwarg explicitly

`simple_evaluate`'s `fewshot_as_multiturn` default **flipped `False` → `True`** between 0.4.5
and 0.4.12, and the `ValueError` guarding multiturn-without-chat-template was **deleted**. Same
call, different prompt framing, on precisely the axis README-runner.md says moves MC2 from 0.06
to 0.26.

Guard by discipline, not a shim — every kwarg we need exists across the whole 0.4.5–0.4.12
band: (a) never rely on an lm_eval default; (b) `simple_evaluate` is `@positional_deprecated`,
so always call by keyword; (c) add the `inspect.signature` test below; (d) commit `uv.lock`.
Do **not** pass `confirm_run_unsafe_code` / `samples` / `metadata` (0.4.12-only, and unneeded —
no code-execution tasks here).

### All three task names are **tags**, not groups

Verified against v0.4.12 YAML source — the okapi README is stale and disagrees; trust the YAML.

| config name               | expands to                                                          | count |
| ------------------------- | ------------------------------------------------------------------- | ----- |
| `truthfulqa`              | `truthfulqa_mc1`, `truthfulqa_mc2`, `truthfulqa_gen`                | 3     |
| `truthfulqa_multilingual` | `truthfulqa_<lang>_mc1` / `_mc2`, 31 okapi langs                    | 62    |
| `truthfulqa-multi`        | `truthfulqa-multi_{mc1,mc2,gen}_{en,es,ca,eu,gl}` (note the hyphen) | 15    |

Tags **do not aggregate** ⇒ `results["groups"]` will be **absent** and there is no
cross-language aggregate. Normalize defensively (`.get(..., {})`); ⚠️ whether tags populate
`group_subtasks` is unverified.

Never hardcode these lists — carry names in config, validate via `TaskManager`, and ship
`--list-tasks <substring>` for discovery. Open/closed: a new benchmark is a new YAML name, no
code change.

### Dependencies — no extras needed

`truthfulqa_gen` needs `sacrebleu` + `rouge-score`, **both base deps**; BLEURT is commented
out upstream, so **no TensorFlow** (unlike `TruthfulQA-reproduce`'s `legacy-metrics` extra).
okapi's `utils.py` imports only `re`/`datasets`/`numpy` ⇒ **`lm_eval[multilingual]` is not
needed**. `truthfulqa-multi` gen uses sacrebleu; its "LLM-as-a-Judge" is the paper's
methodology, not the harness task.

lm_eval's floors (`torch>=1.8`, `transformers>=4.1`, `datasets>=2.16`, `peft>=0.2`,
`accelerate>=0.26`) sit far below every repo cap — no conflict. Still declare
torch/transformers/accelerate/peft **ourselves** at the repo caps, so the base→`hf`-extra
migration can't drop them. New transitive weight: `evaluate`, `pytablewriter`, `sqlitedict`,
`dill`, `word2number`, `more_itertools`, `scikit-learn`, `sacrebleu`, `rouge-score`
(sdist-only → needs a build; flag for the offline HPC mirror).

---

## Files

### New: `harness-eval/` (folder exists, empty and untracked)

Follows the `FaithEval-reproduce` reference layout, minus what lm_eval owns — **no**
`data.py` / `prompting.py` / `metrics.py` / `configs/`.

```
harness-eval/
├── pyproject.toml          # requires-python=">=3.10" (deviation, comment why)
├── pyproject-HPC.toml      # cluster twin
├── uv.lock                 # committed (generated by uv sync)
├── README.md               # Quick Start / flags / Install (HPC) / own checkpoint
├── ARCHITECTURE.md         # module map; the 3 findings; "2nd independent number"
├── src/
│   ├── run_eval.py         # 3-line shim -> harness_eval.cli:main
│   └── harness_eval/
│       ├── __init__.py
│       ├── cli.py          # argparse; parse_args -> build_config -> run_evaluation; --list-tasks
│       ├── config.py       # frozen EvalConfig + __post_init__ validation
│       ├── model.py        # build_model_args()  <- core value-add
│       ├── tasks.py        # resolve_tasks() / list_tasks()  <- core value-add
│       ├── results.py      # EvalResults -> flat rows + samples.jsonl
│       └── evaluator.py    # run_evaluation(): calls simple_evaluate, writes I/O
└── tests/
    ├── test_model_args.py
    ├── test_tasks.py
    ├── test_results.py
    └── test_cli.py
```

Module notes:

- **`model.py` must not import torch or peft.** Read `adapter_config.json` with `json.load`
  instead of `PeftConfig.from_pretrained` — exactly equivalent, since `_is_peft_adapter` only
  ever matches a local dir, and it keeps the file import-light so its tests run in
  milliseconds with zero heavy deps. Port `_looks_like_local_path` / `_check_local_path_exists`
  / `_is_peft_adapter` verbatim from
  [faitheval/model.py](Eval_master/FaithEval-reproduce/src/faitheval/model.py#L22-L64).
- **`evaluator.run_evaluation(config, *, evaluate_fn=None)`** — a one-line seam; `None` →
  lazy-import `lm_eval.simple_evaluate`. Lets tests inject a fake **with lm_eval absent**, and
  preserves the repo's lazy-import convention (importing `lm_eval` drags in torch _and_
  datasets, so `--help` must not touch it; `--list-tasks` legitimately pays the cost).
- Repo conventions throughout: `from __future__ import annotations`,
  `parse_args(argv: list[str] | None = None)`, `ArgumentDefaultsHelpFormatter`,
  cross-validation via `parser.error(...)`, frozen dataclass config.
- **Omit `license = { file = ... }`** — the siblings point at a vendored upstream LICENSE;
  we vendor nothing, and a dangling file reference breaks the setuptools build.

`pyproject.toml` deps:

```toml
requires-python = ">=3.10"   # DEVIATION from the repo's >=3.9: lm_eval 0.4.10+ requires
                             # >=3.10, and ">=3.9" makes the resolver silently backtrack to 0.4.9.x.
dependencies = [
    "lm_eval[hf]>=0.4.12,<0.5",
    "torch>=2.1,<3",           # declared here so lm_eval's base->`hf`-extra move can't drop them
    "transformers>=4.44,<5",
    "accelerate>=0.33,<2",
    "datasets>=2.20,<4",
    "peft>=0.11,<1",
]
[project.optional-dependencies]
dev = ["pytest>=8.0"]
[project.scripts]
harness-eval = "harness_eval.cli:main"
```

`pyproject-HPC.toml`: copy the FaithEval twin (`torch==2.2.2`, `transformers>=4.41,<4.42`,
`datasets>=2.18,<3.0`, `pyarrow<16`, `numpy>=1.26,<2`, `requires-python=">=3.12,<3.13"`,
`[tool.uv] index-strategy = "unsafe-best-match"`) + `lm_eval[hf]==0.4.12`.
⚠️ Metadata-clean but **runtime-unverified**: 0.4.12's `HFLM` is written against far newer
transformers than 4.41. Document the fallback `lm_eval==0.4.8` (`>=3.9`, torch as a base dep,
contemporaneous with 4.41). Only the cluster can settle it.

### Modified: `run_benchmarks.py`

`FOLDERS["harness"] = ROOT / "harness-eval"` · `BUILDERS["harness"] = build_harness` · no
`CWD_SUBDIR` · add `"harness"` to the `:214` mkdir tuple for consistency (the module mkdirs
its own `--output-dir` anyway, as faitheval/ragtruth do).

```python
def build_harness(cfg: DictConfig, out: Path) -> list[list[str]]:
    b = cfg.harness
    samples = _resolve_samples(b, cfg)
    # One command for the whole task list: lm_eval loads the model once and
    # evaluates every task in a single pass.
    cmd = (
        ["src/run_eval.py", "--model-id", str(cfg.model.id)]
        + _model_common_map(cfg)
        + ["--tasks", *[str(t) for t in b.tasks]]
        + ["--dtype", str(cfg.model.dtype), "--device", str(cfg.model.device_index)]
        + _opt("--num-fewshot", b.num_fewshot)
        + _opt("--batch-size", b.batch_size)
        + _opt("--limit", samples)
        + _opt("--system-instruction", b.system_instruction)
        + (["--apply-chat-template"] if b.apply_chat_template else [])
        + (["--fewshot-as-multiturn"] if b.fewshot_as_multiturn else [])
        + (["--trust-remote-code"] if b.trust_remote_code else [])
        + (["--log-samples"] if b.log_samples else [])
        + ["--output-dir", str(out / "harness")]
        + list(b.extra_args)
    )
    return [cmd]
```

Reuses `_opt` / `_resolve_samples` / `_model_common_map` unchanged. `_opt` is safe here: it
tests `value is None or value == ""`, and `-1 == ""` / `0 == ""` are both `False`, so
`--device -1` survives. `--tasks` is `nargs="+"`; no task name starts with `-`, so the next
flag terminates it.

### New: `conf/harness/default.yaml`

```yaml
# lm-evaluation-harness — the community-standard TruthfulQA numbers.
#
# Wraps EleutherAI/lm-evaluation-harness (`lm_eval`). This is a SECOND, independent
# number next to `TruthfulQA-reproduce` (which runs the original authors' scripts):
# the harness is what public leaderboards report, so both are kept. Expect the two
# MC1/MC2 numbers to differ — the prompt and scoring plumbing are not identical.
enabled: true
python: null

# lm_eval task names, passed to `--tasks`. All three below are lm_eval *tags*, not
# *groups*: each expands to its subtasks and is reported per-subtask, with NO
# aggregate score.
#   truthfulqa              -> truthfulqa_mc1, truthfulqa_mc2, truthfulqa_gen (English)
#   truthfulqa_multilingual -> truthfulqa_<lang>_mc1 / _mc2, 31 okapi languages (62 tasks)
#   truthfulqa-multi        -> truthfulqa-multi_{mc1,mc2,gen}_{en,es,ca,eu,gl} (HiTZ)
# Glob patterns work (e.g. 'truthfulqa_de_*'). Names are validated against the
# installed lm_eval; `src/run_eval.py --list-tasks truthfulqa` prints what it knows.
tasks: [truthfulqa]

# Sample cap. Falls back to the global `num_samples`. NOTE: this is lm_eval's
# --limit, which applies PER TASK — num_samples=5 over the 62 okapi tasks is 310
# evaluations, not 5. Any non-null value makes the run non-comparable to published
# scores; it is recorded in summary.json.
num_samples: null

# Fewshot count. null = each task's own default (TruthfulQA is 0-shot; its few-shot
# QA primer is baked into the prompt upstream).
num_fewshot: null

batch_size: 1 # int, or "auto" / "auto:N" for lm_eval's autodetection

# How the question is presented to the evaluated model.
#   false (default) — the published completion-style protocol, and lm_eval's own
#                     default; this is what leaderboard numbers use and why this
#                     module exists. Comparable to `truthfulqa.prompt_style=completion`.
#   true            — render prompts with the model's own chat template. Comparable
#                     to `truthfulqa.prompt_style=chat` (its default) and to the
#                     other three benchmarks.
# This MATERIALLY moves the scores (README-runner.md "Prompt format": MC2 0.06 vs
# 0.26 on a smoke test), so keep it fixed across the models you compare. The
# resolved value is recorded in summary.json.
apply_chat_template: false

# Present few-shot examples as separate chat turns; only meaningful with
# apply_chat_template=true. Passed explicitly on EVERY run because lm_eval's own
# default for this flipped from false to true between 0.4.5 and 0.4.12.
fewshot_as_multiturn: false

system_instruction: null # optional system prompt (requires apply_chat_template)
trust_remote_code: false # for checkpoints with custom modeling code
log_samples: true # write per-document records to samples.jsonl

extra_args: [] # raw flags forwarded to src/run_eval.py
```

`num_samples: null` **must** be present, or Hydra struct mode rejects the
`harness.num_samples=5` override that `config.yaml:22` documents.

### Modified: docs & config prose

Non-optional — otherwise the repo self-contradicts.

- `conf/config.yaml`: add `- harness: default` to `defaults:` and `harness` to `run:`; the
  `num_samples` comment (`:20-24`) must add harness **and** note its `--limit` is **per task**,
  unlike the others.
- `conf/model/qwen_tiny.yaml:15` + `README-runner.md:116`: `device_index` comment
  `"for TruthfulQA (--device)"` → `"for TruthfulQA / harness (--device)"`.
- `README-runner.md`: add a **"Prompt format"** table row for harness (`apply_chat_template`,
  default `false` = published protocol) — that section is the whole reason the flag exists;
  add `harness/` to **Outputs**; add the `run` row and tree entry.
- **"the four benchmarks" → five** in `run_benchmarks.py:2-7`, `conf/config.yaml:2-3,17-18,26-30`,
  `setup_envs.sh:4-6`, `README-runner.md` (intro + install).
- `setup_envs.sh:31`: `BENCHMARKS=(… RAGTruth-reproduce harness-eval)`, last, matching `run:`
  order. `_candidate_venvs` uses `folder.name`, so the venv lands at `<venv_root>/harness-eval`.

---

## Result normalization

`EvalResults` (`lm_eval/result_schema.py` @ v0.4.12) is a `TypedDict, total=False` — **every
key is optional**. `results: dict[str, _TaskMetrics]` is the only one to count on; each
`_TaskMetrics` holds meta keys (`name`, `alias`, `sample_len`, `sample_count`) plus dynamic
`"metric,filter"` pairs like `"acc,none"` / `"acc_stderr,none"`.

```python
_META_KEYS = {"alias", "name", "sample_len", "sample_count"}

for kind, section in (("task", res.get("results", {})), ("group", res.get("groups", {}))):
    for task_name, metrics in section.items():
        for key, value in metrics.items():
            if key in _META_KEYS:
                continue
            metric, _, filt = key.partition(",")      # "acc,none" -> ("acc", "none")
            if metric.endswith("_stderr"):
                continue                              # consumed as its base metric's stderr
            stderr = metrics.get(f"{metric}_stderr,{filt}")
            rows.append({
                "task": task_name, "kind": kind, "alias": metrics.get("alias"),
                "metric": metric, "filter": filt, "value": value,
                "stderr": None if stderr in (None, "N/A") else stderr,
                "higher_is_better": res.get("higher_is_better", {}).get(task_name, {}).get(metric),
                "num_fewshot": res.get("n-shot", {}).get(task_name),
                "version": res.get("versions", {}).get(task_name),
                "n_samples": res.get("n-samples", {}).get(task_name),
            })
```

Subtleties: `partition(",")` not `split` · the `_stderr` suffix pairing · aggregate stderr is
the literal string `"N/A"` · group-member `alias` arrives **indented** (`" - truthfulqa_mc1"`)
so key on `task_name`, never `alias` · the hyphenated keys `"n-shot"` / `"n-samples"`.

**🚩 Serialization:** results contain numpy scalars → `json.dumps` raises
`TypeError: Object of type float32 is not JSON serializable`. lm_eval's own `__main__` passes
`default=handle_non_serializable`; use `json.dumps(..., indent=2, default=str)`.

Mine lm_eval's own provenance keys — free and authoritative: `lm_eval_version`,
`transformers_version`, `git_hash`, `date`, `chat_template_sha`, `n-shot`, `config`.

Outputs (repo convention):

- `harness/summary.json` — `indent=2`, `encoding="utf-8"`; flat rows + a provenance header
  (`apply_chat_template`, `fewshot_as_multiturn`, `limit`, `num_fewshot`, `resolved_tasks`,
  `model_args` string, `lm_eval_version`).
- `harness/samples.jsonl` — `ensure_ascii=False`, one record per doc, `task` injected.
- `harness/lm_eval_results.json` — the raw dict minus `samples`.

---

## Tests

pytest, module-level `def test_*`, bare asserts, no conftest.py. **Every test runs offline
with no model or dataset download**; only `test_tasks.py` needs lm_eval installed.

**`tests/test_model_args.py`** — pure logic, no heavy imports, no mocking:
`test_hub_id_maps_to_pretrained_only` · `test_hub_id_with_slash_is_not_treated_as_local_path` ·
`test_local_full_model_dir_maps_to_pretrained_only` (`tmp_path` + `config.json`) ·
`test_adapter_dir_maps_base_to_pretrained_and_adapter_to_peft` (`tmp_path/adapter_config.json`
= `{"base_model_name_or_path": "base/model"}`) · `test_explicit_base_model_id_overrides_adapter_config` ·
`test_adapter_without_base_anywhere_raises` (`ValueError` naming `--base-model-id`) ·
**`test_adapter_tokenizer_defaults_to_adapter_dir_not_base`** (regression for Finding 3) ·
`test_adapter_without_tokenizer_files_falls_back_to_base` · `test_explicit_tokenizer_id_wins` ·
`test_missing_local_path_raises_file_not_found` (not a Hub 404) · `test_none_values_are_omitted` ·
`test_cache_dir_included_when_given` · `test_model_args_string_rejects_value_containing_comma`
(documents the `simple_parse_args_string` hazard at our boundary).

**`tests/test_tasks.py`** — `pytest.importorskip("lm_eval")`; `TaskManager()` indexes local
YAML only, no download:
`test_resolve_tasks_accepts_known_tag` · `test_resolve_tasks_rejects_unknown_name_with_suggestion`
(`"truthfulqaa"` → `ValueError` containing `truthfulqa`, via `difflib.get_close_matches`) ·
`test_resolve_tasks_expands_glob` ·
**`test_three_truthfulqa_tags_are_known_to_installed_lm_eval`** — the canary:
`{"truthfulqa", "truthfulqa_multilingual", "truthfulqa-multi"} <= set(tm.all_tags)`; catches an
upstream rename we cannot see offline ·
**`test_simple_evaluate_accepts_every_kwarg_we_pass`** — pure
`inspect.signature(lm_eval.simple_evaluate).parameters`, asserts our kwarg set is a subset.
Zero cost, catches version drift on upgrade.

**`tests/test_results.py`** — hand-written fixture dicts, no lm_eval import, no mocking:
`test_flatten_splits_metric_and_filter_keys` · `test_flatten_ignores_non_metric_keys` ·
`test_flatten_pairs_stderr_with_its_metric` · `test_flatten_sets_stderr_none_when_absent` ·
`test_flatten_coerces_na_stderr_to_none` · **`test_missing_groups_key_is_not_an_error`**
(encodes the tags-don't-aggregate finding) · `test_group_subtasks_recorded_when_present` ·
`test_flatten_uses_task_key_not_indented_alias` · `test_summary_records_provenance_fields` ·
`test_samples_jsonl_written_one_record_per_doc` ·
**`test_samples_jsonl_preserves_non_ascii`** (an Arabic doc stays literal — matters for the 31
okapi languages) · `test_summary_serializes_numpy_floats`.

**`tests/test_cli.py`**:
`test_parse_args_defaults_apply_chat_template_false` (locks the published-protocol default) ·
**`test_fewshot_as_multiturn_without_chat_template_errors`** — `parser.error(...)`, mirroring
faitheval's `--temperature requires --do-sample`; this **reinstates at our layer the
`ValueError` lm_eval deleted after 0.4.5**, giving identical behavior across the version band ·
`test_list_tasks_mode_does_not_require_model_id` · `test_device_index_minus_one_maps_to_cpu` /
`test_device_index_zero_maps_to_cuda0` · `test_help_does_not_import_lm_eval`
(`assert "lm_eval" not in sys.modules`; enforces the lazy-import convention) ·
**`test_run_evaluation_passes_expected_kwargs`** — the only test needing a fake; uses the
`evaluate_fn=` seam (not monkeypatch) with a stub capturing `**kwargs` and returning a canned
`EvalResults`. Asserts `apply_chat_template` / `fewshot_as_multiturn` / `limit` are passed
**explicitly**, and that `summary.json` + `samples.jsonl` land in `tmp_path`. Runs with lm_eval
absent.

---

## Verification

```bash
# 1. Unit tests, no downloads (from harness-eval/)
"$LOCALAPPDATA/eval-venvs/harness-eval/Scripts/python.exe" -m pytest tests/ -v

# 2. Env builds and the lock resolves as intended
./setup_envs.sh --venv-root "$LOCALAPPDATA/eval-venvs"
"$LOCALAPPDATA/eval-venvs/harness-eval/Scripts/python.exe" -c \
    "import lm_eval; print(lm_eval.__version__)"      # MUST be 0.4.12+, not 0.4.9.x (Finding 1)

# 3. Task names are real in the installed version
python src/run_eval.py --list-tasks truthfulqa

# 4. Launcher wiring — the commands, without running them
export VENV_ROOT="$LOCALAPPDATA/eval-venvs"
python run_benchmarks.py run='[harness]' dry_run=true
python run_benchmarks.py dry_run=true                 # all five still compose

# 5. End-to-end smoke test (CPU, tiny model, 5 docs/task)
./run_all.sh run='[harness]' model.dtype=float32 num_samples=5 \
    harness.tasks='[truthfulqa_mc1]'
#    -> outputs/<date>/<time>/harness/{summary.json,samples.jsonl,lm_eval_results.json}
#    -> assert summary.json has an acc row and records apply_chat_template=false

# 6. LoRA path — the reason Finding 3 exists
./run_all.sh run='[harness]' model.id=/path/to/final_checkpoint num_samples=5
#    -> assert lm_eval_results.json / logs show tokenizer == the adapter dir, NOT the base

# 7. Multilingual opt-in (gated on Step 0)
./run_all.sh run='[harness]' num_samples=2 harness.tasks='[truthfulqa_de_mc1]'
```

Step 5's number is **not** comparable to a published score (`--limit` is set); drop
`num_samples` for a real run.
