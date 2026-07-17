import argparse
import os
import sys
from . import models
from . import metrics
from . import prompting
from . import utilities
from .configs import ENGINE_MAP
import warnings
import pandas as pd


def format_frame(results):

    # Keep only the numeric score columns. This tested `dtype != 'O'`, which assumed
    # text columns are always object dtype; modern pandas gives them StringDtype, so
    # they slipped through and broke the .mean() below.
    results = results[[x for x in results.columns
                       if (x != 'Context') and pd.api.types.is_numeric_dtype(results[x])]]

    new_cols = []
    for col in results.columns:
        split = col.split(' ')
        new_cols.append((split[0], ' '.join(split[1:])))
    results.columns = pd.MultiIndex.from_tuples(new_cols)

    return results


def data_to_dict(results):

    model_names = list(results.columns.get_level_values(0).unique())
    data_dict = {model: results[model] for model in model_names}

    for mdl in data_dict:
        for diff in ['bleu', 'rouge1', 'BLEURT']:
            if '{0} diff'.format(diff) in data_dict[mdl].columns:
                data_dict[mdl]['{0} acc'.format(diff)] = (data_dict[mdl]['{0} diff'.format(diff)] > 0).astype(int)

    return data_dict


def main():

    # Model answers routinely contain characters the Windows console codepage
    # (cp1252) cannot encode. Printing one then raises UnicodeEncodeError, which the
    # broad `except` around each model aborts the whole generation loop on -- losing
    # every remaining answer. Degrade unencodable characters instead.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='replace')

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['neo-small'])
    parser.add_argument('--metrics', nargs='+', default=['bleu'])
    parser.add_argument('--preset', type=str, default='qa')
    parser.add_argument('--input_path', type=str, default='questions.csv')
    parser.add_argument('--output_path', type=str, default='answers.csv')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--gptj_path', type=str)
    # --- Run your own checkpoint / LoRA adapter (see truthfulqa/hf_local.py) ---
    parser.add_argument('--model_path', type=str, default=None,
                        help="Hugging Face Hub id, local full-model path, or PEFT/LoRA adapter "
                             "checkpoint to evaluate. When set, a synthetic 'local' model key is "
                             "evaluated instead of --models.")
    parser.add_argument('--base_model_id', type=str, default=None,
                        help="Base model id/path for a --model_path that is a LoRA adapter, if the "
                             "base recorded in the adapter config isn't resolvable on this machine.")
    parser.add_argument('--tokenizer_id', type=str, default=None,
                        help="Tokenizer id/path, if not saved alongside --model_path.")
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['bfloat16', 'float16', 'float32'],
                        help="dtype for --model_path (default float32 is CPU-safe).")
    parser.add_argument('--prompt_style', type=str, default='completion',
                        choices=list(prompting.VALID_STYLES),
                        help="How to present the few-shot examples: 'completion' is the "
                             "original raw Q:/A: string; 'chat' renders them through the "
                             "model's own chat template; 'auto' picks chat when the "
                             "tokenizer has a template. Default 'completion' (upstream).")
    # --- GPT-judge / GPT-info (`--metrics judge info`; see judge_local.py) ---
    parser.add_argument('--judge_backend', type=str, default='local', choices=['local', 'openai'],
                        help="'local' runs the authors' successor judge models in-process "
                             "(no API key). 'openai' is the legacy fine-tuned-GPT-3 path.")
    parser.add_argument('--truth_judge_id', type=str, default='allenai/truthfulqa-truth-judge-llama2-7B',
                        help="Truth judge: an HF id/path (--judge_backend local) or the name of "
                             "your fine-tuned GPT-judge engine (--judge_backend openai).")
    parser.add_argument('--info_judge_id', type=str, default='allenai/truthfulqa-info-judge-llama2-7B',
                        help="Informativeness judge; see --truth_judge_id.")
    args = parser.parse_args()

    questions = utilities.load_questions(filename=args.input_path)

    # set torch device
    if args.device >= 0:
        torch_device = "cuda:{0}".format(args.device)
    else:
        torch_device = "cpu"

    # Load a user-supplied checkpoint / LoRA adapter once and route it through the
    # autoregressive (run_answers) and MC (run_probs) branches under the 'local' key.
    local_model = None
    local_tokenizer = None
    prompt_style = args.prompt_style
    if args.model_path:
        from .hf_local import load_local_model
        local_model, local_tokenizer = load_local_model(
            args.model_path,
            base_model_id=args.base_model_id,
            tokenizer_id=args.tokenizer_id,
            cache_dir=args.cache_dir,
            device=torch_device,
            dtype=args.dtype,
        )
        args.models = ['local']
        # Resolve once, up front: 'chat' needs a chat template and a Q:/A:-shaped
        # preset, and the resolved value is recorded alongside the results.
        prompt_style = prompting.resolve_style(args.prompt_style, local_tokenizer, args.preset)
        print("Prompt style: {0} (requested: {1}); preset: {2}".format(
            prompt_style, args.prompt_style, args.preset))

    check_m = [x for x in args.metrics if x in ['judge', 'info']]

    # GPT-judge / GPT-info: load the local judges, or set up the legacy OpenAI path.
    judges = {}
    if len(check_m) and args.judge_backend == 'local':
        from .judge_local import LocalJudge
        judge_ids = {'judge': args.truth_judge_id, 'info': args.info_judge_id}
        for metric in check_m:
            print("Loading {0} judge: {1}".format(metric, judge_ids[metric]))
            judges[metric] = LocalJudge(
                judge_ids[metric],
                cache_dir=args.cache_dir,
                device=torch_device,
                dtype=args.dtype,
            )

    # check for GPT-3 models and set API key
    check = [x for x in args.models if x in ['ada', 'babbage', 'curie', 'davinci']]
    if len(check) or (len(check_m) and args.judge_backend == 'openai'):  # OpenAI API access required
        import openai  # lazy: the OpenAI reproduction path is optional
        # Read the key from the environment: this runs as a subprocess under the
        # central launcher, where an interactive input() prompt would deadlock.
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise SystemExit(
                "The OpenAI path needs OPENAI_API_KEY to be set. Use --judge_backend local "
                "to run the judges in-process instead (no API key needed)."
            )
        openai.api_key = api_key
        judge_name = args.truth_judge_id
        info_name = args.info_judge_id

    # populate frame with model answers
    for mdl in args.models:
        print("Running {0}!".format(mdl))

        # gpt-3
        if mdl in ['ada', 'babbage', 'curie', 'davinci']:  # gpt-3 models
            try:
                models.run_GPT3(questions, mdl, mdl, args.preset)
                utilities.save_questions(questions, args.output_path)
                if 'mc' in args.metrics:
                    models.run_probs_GPT3(questions, mdl, mdl, preset=args.preset)
                    utilities.save_questions(questions, args.output_path)
            except Exception as err:
                print(err)

        # local checkpoint / LoRA adapter (loaded via --model_path)
        if mdl == 'local':
            try:
                models.run_answers(questions, 'local', 'local', args.preset,
                                   model=local_model, tokenizer=local_tokenizer, device=torch_device,
                                   prompt_style=prompt_style)
                utilities.save_questions(questions, args.output_path)
                if 'mc' in args.metrics:
                    models.run_probs(questions, 'local', 'local', preset=args.preset,
                                     model=local_model, tokenizer=local_tokenizer, device=torch_device,
                                     prompt_style=prompt_style)
                    utilities.save_questions(questions, args.output_path)
            except Exception as err:
                print(err)

        # gpt-2
        if mdl in ['gpt2', 'gpt2-xl']:
            try:
                models.run_answers(questions, mdl, mdl, args.preset, device=torch_device, cache_dir=args.cache_dir)
                utilities.save_questions(questions, args.output_path)
                if 'mc' in args.metrics:
                    models.run_probs(questions, mdl, mdl, preset=args.preset, device=torch_device, cache_dir=args.cache_dir)
                    utilities.save_questions(questions, args.output_path)
            except Exception as err:
                print(err)

        # gpt-neo
        if mdl in ['neo-small', 'neo-med', 'neo-large']:
            try:
                models.run_answers(questions, ENGINE_MAP[mdl], mdl, args.preset,
                                   device=torch_device, cache_dir=args.cache_dir)
                utilities.save_questions(questions, args.output_path)
                if 'mc' in args.metrics:
                    models.run_probs(questions, ENGINE_MAP[mdl], mdl, preset=args.preset, device=torch_device,
                                     cache_dir=args.cache_dir)
                    utilities.save_questions(questions, args.output_path)
            except Exception as err:
                print(err)

        # gpt-j
        if mdl == 'gptj':
            if not args.gptj_path:  # model checkpoint missing
                warnings.warn("No GPT-J model checkpoint, skipping!", stacklevel=2)
                break
            try:
                models.run_GPTJ(questions, mdl, args.preset, device=torch_device, cache_dir=args.cache_dir,
                                check_path=args.gptj_path)
                utilities.save_questions(questions, args.output_path)
                if 'mc' in args.metrics:
                    models.run_probs_GPTJ(questions, mdl, args.preset, device=torch_device, cache_dir=args.cache_dir,
                                          check_path=args.gptj_path)
                    utilities.save_questions(questions, args.output_path)
            except Exception as err:
                print(err)

        # unifiedqa
        if mdl in ['uqa-small', 'uqa-base', 'uqa-large', 'uqa-3b']:
            try:
                models.run_UnifQA(questions, ENGINE_MAP[mdl], mdl, args.preset, device=torch_device, cache_dir=args.cache_dir)
                utilities.save_questions(questions, args.output_path)
                if 'mc' in args.metrics:
                    models.run_probs_T5(questions, ENGINE_MAP[mdl], mdl, args.preset, device=torch_device, cache_dir=args.cache_dir)
                    utilities.save_questions(questions, args.output_path)
            except Exception as err:
                print(err)

    # run metrics
    for model_key in args.models:

        if model_key not in questions.columns:
            warnings.warn("Answers missing for {0}!".format(model_key), stacklevel=2)
            continue

        for metric in args.metrics:
            if metric == 'mc':
                continue
            if metric == 'bleurt':
                try:
                    questions = metrics.run_BLEURT(model_key, questions, cache_dir=args.cache_dir)
                    utilities.save_questions(questions, args.output_path)
                except Exception as err:
                    print(err)
            elif metric in ['bleu', 'rouge']:
                try:
                    questions = metrics.run_bleu_and_rouge(model_key, questions)
                    utilities.save_questions(questions, args.output_path)
                except Exception as err:
                    print(err)
            elif metric in ['judge', 'info']:
                tag = 'GPT-judge' if metric == 'judge' else 'GPT-info'
                is_info = metric == 'info'
                try:
                    if args.judge_backend == 'local':
                        questions = metrics.run_end2end_local(model_key, tag, judges[metric],
                                                              questions, info=is_info)
                    else:
                        engine = info_name if is_info else judge_name
                        questions = metrics.run_end2end_GPT3(model_key, tag, engine,
                                                             questions, info=is_info)
                    utilities.save_questions(questions, args.output_path)
                except Exception as err:
                    print(err)
            else:
                warnings.warn("Metric {0} not known, skipping!".format(metric), stacklevel=2)

    # save all
    utilities.save_questions(questions, args.output_path)

    # format and print basic results
    results = format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2',
                                              'bleu acc',
                                              'rouge1 acc',
                                              'BLEURT acc',
                                              'GPT-judge acc',
                                              'GPT-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')
    # Keep the summary next to the answers it summarizes; writing it to the cwd
    # would drop it in the repo root and overwrite it on every run.
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    results.to_csv(os.path.join(out_dir, 'summary.csv'))

    # Record what produced these numbers: the prompt format and judge engine are
    # both configurable, so a bare score is not interpretable without them.
    if args.model_path:
        run_config = pd.DataFrame([{
            'model_path': args.model_path,
            'tokenizer_id': args.tokenizer_id or args.model_path,
            'base_model_id': args.base_model_id,
            'preset': args.preset,
            'prompt_style_requested': args.prompt_style,
            'prompt_style_resolved': prompt_style,
            'metrics': ','.join(args.metrics),
            'judge_backend': args.judge_backend if check_m else None,
            'truth_judge_id': args.truth_judge_id if 'judge' in check_m else None,
            'info_judge_id': args.info_judge_id if 'info' in check_m else None,
            'dtype': args.dtype,
        }])
        run_config.to_csv(os.path.join(out_dir, 'run_config.csv'), index=False)


if __name__ == '__main__':
    main()