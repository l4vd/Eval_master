import os
import random
import time
import json
import argparse

# `openai` and `tiktoken` are imported lazily inside the OpenAI-backed helpers so
# the local HuggingFace judge backend (--backend hf) runs without either package
# and without an OpenAI API key. The key is read from the OPENAI_API_KEY env var
# when the OpenAI backend is actually used (see _ensure_openai).


def _ensure_openai():
    """Import openai and set the API key from OPENAI_API_KEY (OpenAI backend only)."""
    import openai
    if not getattr(openai, "api_key", None):
        openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    return openai


def get_qa_response(model, question, answer, instruction, backend="openai", generator=None):
    message = [
        {"role": "system", "content":"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Question#: " + question +
                                    "\n#Answer#: " + answer +
                                    "\n#Your Judgement#: "}
    ]
    prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    if backend == "hf":
        return generator.generate(message, prompt)
    openai = _ensure_openai()
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            else:
                res = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    return response


def get_dialogue_response(model, dialog, response, instruction, backend="openai", generator=None):
    message = [
        {"role": "system", "content": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Dialogue History#: " + dialog +
                                    "\n#Response#: " + response +
                                    "\n#Your Judgement#: "}
    ]
    prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    if backend == "hf":
        return generator.generate(message, prompt)
    openai = _ensure_openai()
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            else:
                res = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    return response


def num_tokens_from_message(message, model="davinci"):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split()[:-1])
    prompt = prompt1 + prompt2
    return prompt


def get_summarization_response(model, document, summary, instruction, backend="openai", generator=None):
    message = [
        {"role": "system", "content": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Document#: " + document +
                                    "\n#Summary#: " + summary +
                                    "\n#Your Judgement#: "}
    ]
    prompt1 = instruction + "\n\n#Document#: " + document
    prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    if backend == "hf":
        # Untruncated: `truncate_message` is a workaround for davinci's 2033-token
        # window and needs tiktoken (an openai-path dependency).
        return generator.generate(message, prompt1 + prompt2)
    if model == "davinci":
        prompt = truncate_message(prompt1, prompt2)
    else:
        prompt = prompt1 + prompt2
    openai = _ensure_openai()
    while True:
        try:
            if model == "gpt-3.5-turbo":
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=0.0,
                )
                response = res['choices'][0]['message']['content']
            else:
                res = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0
                )
                response = res["choices"][0]['text'].strip()
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    return response


def evaluation_qa_dataset(model, file, instruction, output_path, backend="openai", generator=None, num_samples=None):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        n = len(data) if num_samples is None else min(num_samples, len(data))
        correct = 0
        incorrect = 0
        for i in range(n):
            knowledge = data[i]["knowledge"]
            question = data[i]["question"]
            hallucinated_answer = data[i]["hallucinated_answer"]
            right_answer = data[i]["right_answer"]

            if random.random() > 0.5:
                answer = hallucinated_answer
                ground_truth = "Yes"
            else:
                answer = right_answer
                ground_truth = "No"

            ans = get_qa_response(model, question, answer, instruction, backend=backend, generator=generator)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
                incorrect += 1

            assert(gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        accuracy = correct / n if n else 0.0
        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, accuracy))
        return {"num_examples": n, "num_correct": correct, "num_incorrect": incorrect, "accuracy": accuracy}


def evaluation_dialogue_dataset(model, file, instruction, output_path, backend="openai", generator=None, num_samples=None):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        n = len(data) if num_samples is None else min(num_samples, len(data))
        correct = 0
        incorrect = 0
        for i in range(n):
            knowledge = data[i]["knowledge"]
            dialog = data[i]["dialogue_history"]
            hallucinated_response = data[i]["hallucinated_response"]
            right_response = data[i]["right_response"]

            if random.random() > 0.5:
                response = hallucinated_response
                ground_truth = "Yes"
            else:
                response = right_response
                ground_truth = "No"

            ans = get_dialogue_response(model, dialog, response, instruction, backend=backend, generator=generator)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        accuracy = correct / n if n else 0.0
        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, accuracy))
        return {"num_examples": n, "num_correct": correct, "num_incorrect": incorrect, "accuracy": accuracy}


def evaluation_summarization_dataset(model, file, instruction, output_path, backend="openai", generator=None, num_samples=None):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        n = len(data) if num_samples is None else min(num_samples, len(data))
        correct = 0
        incorrect = 0
        for i in range(n):

            document = data[i]["document"]
            hallucinated_summary = data[i]["hallucinated_summary"]
            right_summary = data[i]["right_summary"]

            if random.random() > 0.5:
                summary = hallucinated_summary
                ground_truth = "Yes"
            else:
                summary = right_summary
                ground_truth = "No"

            ans = get_summarization_response(model, document, summary, instruction, backend=backend, generator=generator)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
            assert (gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        accuracy = correct / n if n else 0.0
        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, accuracy))
        return {"num_examples": n, "num_correct": correct, "num_incorrect": incorrect, "accuracy": accuracy}


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hallucination Generation")

    parser.add_argument("--task", default="qa", help="qa, dialogue, or summarization")
    parser.add_argument("--model", default="davinci", help="model name (OpenAI backend) or run label")
    parser.add_argument("--backend", default=None, choices=["openai", "hf"],
                        help="Judge backend. Defaults to 'hf' when --model-path is set, else 'openai'.")
    # --- Local HuggingFace judge: run your own checkpoint / LoRA adapter (see hf_local.py) ---
    parser.add_argument("--model-path", dest="model_path", default=None,
                        help="Hugging Face Hub id, local full-model path, or PEFT/LoRA adapter to use "
                             "as the Yes/No hallucination judge (--backend hf).")
    parser.add_argument("--base-model-id", dest="base_model_id", default=None,
                        help="Base model for a --model-path that is a LoRA adapter, if not resolvable "
                             "from the adapter config.")
    parser.add_argument("--tokenizer-id", dest="tokenizer_id", default=None,
                        help="Tokenizer id/path, if not saved alongside --model-path.")
    parser.add_argument("--cache-dir", dest="cache_dir", default=None,
                        help="Hugging Face cache directory.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="dtype for the local HF judge model.")
    parser.add_argument("--device-map", dest="device_map", default="auto",
                        help="device_map passed to from_pretrained for the HF judge.")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=16,
                        help="Max new tokens for the HF judge (a Yes/No answer is short).")
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=None,
                        help="Evaluate only the first N examples (useful for smoke tests).")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Directory for the per-sample results and summary JSON. Defaults to the "
                             "in-repo '<task>/' folder (legacy behaviour) when unset.")
    args = parser.parse_args()

    backend = args.backend or ("hf" if args.model_path else "openai")

    generator = None
    if backend == "hf":
        if not args.model_path:
            raise ValueError("--backend hf requires --model-path")
        from hf_local import HFChatGenerator
        generator = HFChatGenerator(
            model_id=args.model_path,
            base_model_id=args.base_model_id,
            tokenizer_id=args.tokenizer_id,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )

    instruction_file = "{}/{}_evaluation_instruction.txt".format(args.task, args.task)
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()

    model = args.model
    label = args.model_path.replace("/", "_").replace("\\", "_") if (backend == "hf" and args.model_path) else args.model

    # Where the per-sample results (and summary) are written. With --output-dir the
    # artifacts land in the run's unified output tree (alongside the other
    # benchmarks); without it, the legacy in-repo '<task>/' location is kept.
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_dir = args.output_dir
    else:
        results_dir = args.task
    output_path = os.path.join(results_dir, "{}_{}_results.json".format(args.task, label))

    # Truncate any results from a previous run: the per-sample writes below append,
    # so without this a re-run would accumulate stale rows on top of the old file.
    open(output_path, 'w', encoding='utf-8').close()

    data = "../data/{}_data.json".format(args.task)

    kwargs = dict(backend=backend, generator=generator, num_samples=args.num_samples)
    if args.task == "qa":
        stats = evaluation_qa_dataset(model, data, instruction, output_path, **kwargs)
    elif args.task == "dialogue":
        stats = evaluation_dialogue_dataset(model, data, instruction, output_path, **kwargs)
    elif args.task == "summarization":
        stats = evaluation_summarization_dataset(model, data, instruction, output_path, **kwargs)
    else:
        raise ValueError("The task must be qa, dialogue, or summarization!")

    # Persist the headline accuracy so the stored run is self-contained (the other
    # four benchmarks all write a summary; HaluEval previously only printed it).
    summary = {"task": args.task, "model": label, "backend": backend}
    summary.update(stats or {})
    summary_path = os.path.join(results_dir, "{}_{}_summary.json".format(args.task, label))
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("Summary written to {}".format(summary_path))
