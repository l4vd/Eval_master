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


def build_qa_request(question, answer, instruction):
    """The (chat messages, flat completion prompt) pair for one QA judgement.

    Split out of `get_qa_response` so the batched HF path builds byte-identical
    prompts without going through the one-at-a-time generate call.
    """
    message = [
        {"role": "system", "content":"You are a huallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provided MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Question#: " + question +
                                    "\n#Answer#: " + answer +
                                    "\n#Your Judgement#: "}
    ]
    prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    return message, prompt


def get_qa_response(model, question, answer, instruction, backend="openai", generator=None):
    message, prompt = build_qa_request(question, answer, instruction)
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


def build_dialogue_request(dialog, response, instruction):
    """The (chat messages, flat completion prompt) pair for one dialogue judgement."""
    message = [
        {"role": "system", "content": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Dialogue History#: " + dialog +
                                    "\n#Response#: " + response +
                                    "\n#Your Judgement#: "}
    ]
    prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    return message, prompt


def get_dialogue_response(model, dialog, response, instruction, backend="openai", generator=None):
    message, prompt = build_dialogue_request(dialog, response, instruction)
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


def build_summarization_request(document, summary, instruction):
    """The (chat messages, flat completion prompt) pair for one summary judgement.

    Untruncated: `truncate_message` is a workaround for davinci's 2033-token window
    and needs tiktoken (an openai-path dependency).
    """
    message = [
        {"role": "system", "content": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""},
        {"role": "user", "content": instruction +
                                    "\n\n#Document#: " + document +
                                    "\n#Summary#: " + summary +
                                    "\n#Your Judgement#: "}
    ]
    prompt1 = instruction + "\n\n#Document#: " + document
    prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    return message, prompt1 + prompt2


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


SORT_ORDERS = ("desc", "asc", "none")


class _Tally:
    """Row-level bookkeeping shared by the three tasks.

    `correct`, `incorrect` and `accuracy` are produced EXACTLY as the original
    HaluEval script produces them: a row whose judgement could not be parsed counts
    as incorrect and stays in the denominator. Do not "fix" that here — it is the
    published protocol, and changing it would put this fork's numbers on a different
    scale from the paper's.

    Everything else on this class is additional reporting derived from the same
    judgements, so it costs nothing and changes no score. It exists because
    `accuracy` alone cannot distinguish the three ways a judge fails:

      - it judges wrongly            -> accuracy drops, `format_compliance` stays 1.0
      - it never emits a verdict     -> accuracy drops, `format_compliance` drops
      - it emits one constant answer -> accuracy sits near the label base rate,
                                        while `tpr` or `tnr` is 0.0

    The third case is the dangerous one: a judge that answers "No" to everything
    scores ~0.50 and looks merely mediocre.
    """

    def __init__(self):
        self.correct = 0
        self.incorrect = 0
        self.failed = 0
        self.judged_yes = 0
        self.gt_total = {"Yes": 0, "No": 0}
        self.gt_correct = {"Yes": 0, "No": 0}

    def record_failure(self, ground_truth):
        """An unparseable judgement: incorrect, exactly as upstream counts it."""
        self.gt_total[ground_truth] += 1
        self.failed += 1
        self.incorrect += 1

    def record(self, ground_truth, judgement):
        """A parsed "Yes"/"No" judgement."""
        self.gt_total[ground_truth] += 1
        if judgement == "Yes":
            self.judged_yes += 1
        if ground_truth == judgement:
            self.correct += 1
            self.gt_correct[ground_truth] += 1
        else:
            self.incorrect += 1

    def stats(self, n):
        parsed = n - self.failed
        return {
            # --- the published metric, unchanged ---
            "num_examples": n,
            "num_correct": self.correct,
            "num_incorrect": self.incorrect,
            "accuracy": self.correct / n if n else 0.0,
            # --- diagnostics (do not affect the above) ---
            "num_failed": self.failed,
            "format_compliance": parsed / n if n else 0.0,
            "num_ground_truth_yes": self.gt_total["Yes"],
            "num_ground_truth_no": self.gt_total["No"],
            # Recall on each class. A constant judge pins one of these to 0.0.
            "tpr": _ratio(self.gt_correct["Yes"], self.gt_total["Yes"]),
            "tnr": _ratio(self.gt_correct["No"], self.gt_total["No"]),
            # Share of PARSED rows judged "Yes" — 0.0 or 1.0 means degenerate.
            "judged_yes_rate": _ratio(self.judged_yes, parsed),
        }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _seed_labels(seed):
    """Pin the ground-truth coin flips (see REPRODUCIBILITY.md).

    Upstream never seeds, so every run scores a different random 50/50 partition of
    "shown the hallucinated output vs. the correct one" and two arms are compared
    against two different label draws. Seeding does not change the *distribution* of
    the metric — the published number is itself one draw from it — it just makes the
    draw reproducible and makes arms paired. `None` keeps the upstream behaviour.
    """
    if seed is not None:
        random.seed(seed)


def _resolve_batch_size(backend, generator):
    """Prompts per forward pass. Only the local HF judge batches; OpenAI stays serial."""
    if backend != "hf":
        return 1
    return max(1, getattr(generator, "batch_size", 1))


def _batch_order(pending, backend, generator, sort_by_length):
    """Indices of `pending` in the order the judge should see them.

    Sorting by judge-prompt token length groups similarly-sized prompts into one padded
    forward pass, which is where this benchmark's wall clock actually goes: each split is
    10,000 rows generating 16 tokens apiece, so the cost is dominated by the prompt — and in
    file order a batch is padded to its single longest row. `summarization` carries whole
    CNN/DM documents, so that padding is most of the compute there.

    Descending (not ascending) puts the peak-memory batch FIRST, so a CUDA OOM surfaces in
    the first minute rather than after an hour of successful work.

    The caller has already drawn every ground truth in ascending index order, so this
    reordering moves only WHEN a row is judged, never WHICH answer it is shown. Identity for
    the OpenAI backend, which is serial and unbatched (see `_resolve_batch_size`).
    """
    order = list(range(len(pending)))
    if backend != "hf" or sort_by_length == "none" or not pending:
        return order
    # The built request is the last element of every task's pending tuple.
    lengths = generator.prompt_token_lengths([p[-1] for p in pending])
    return sorted(order, key=lambda i: lengths[i], reverse=sort_by_length == "desc")


def evaluation_qa_dataset(model, file, instruction, output_path, backend="openai", generator=None, num_samples=None, sort_by_length="desc", seed=None):
    _seed_labels(seed)
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        n = len(data) if num_samples is None else min(num_samples, len(data))
        batch_size = _resolve_batch_size(backend, generator)
        tally = _Tally()

        # Phase 1 — draw every item's ground truth and build its prompt, over the WHOLE
        # split in ascending index order. Doing this up front (rather than per chunk) is
        # what lets phase 1.5 reorder execution without disturbing the labels: random() is
        # still called exactly once per index in ascending order, so the Yes/No assignment
        # is identical to the unbatched loop for a given seed no matter how batches are cut.
        pending = []
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

            pending.append((i, knowledge, question, answer, ground_truth,
                            build_qa_request(question, answer, instruction)))

        # Phase 1.5 — execution order (longest prompts first), see _batch_order.
        order = _batch_order(pending, backend, generator, sort_by_length)

        for start in range(0, n, batch_size):
            chunk = [pending[i] for i in order[start:min(start + batch_size, n)]]

            # Phase 2 — one padded forward pass for the whole chunk.
            if backend == "hf":
                answers = generator.generate_many([request for *_, request in chunk])
            else:
                answers = [get_qa_response(model, question, answer, instruction,
                                           backend=backend, generator=generator)
                           for _, _, question, answer, _, _ in chunk]

            # Phase 3 — score and write, in execution order (each row carries its `index`).
            for (i, knowledge, question, answer, ground_truth, _), raw in zip(chunk, answers):
                ans = raw.replace(".", "")

                if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                    gen = {"index": i, "knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!", "raw_judgement": raw}
                    dump_jsonl(gen, output_path, append=True)
                    tally.record_failure(ground_truth)
                    print('sample {} fails......'.format(i))
                    continue
                elif "Yes" in ans:
                    if ans != "Yes":
                        ans = "Yes"
                    gen = {"index": i, "knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans, "raw_judgement": raw}
                elif "No" in ans:
                    if ans != "No":
                        ans = "No"
                    gen = {"index": i, "knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans, "raw_judgement": raw}
                else:
                    gen = None

                assert(gen is not None)

                tally.record(ground_truth, ans)

                print('sample {} success......'.format(i))
                dump_jsonl(gen, output_path, append=True)

        stats = tally.stats(n)
        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(
            tally.correct, tally.incorrect, stats["accuracy"]))
        _print_diagnostics(stats)
        return stats


def evaluation_dialogue_dataset(model, file, instruction, output_path, backend="openai", generator=None, num_samples=None, sort_by_length="desc", seed=None):
    _seed_labels(seed)
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        n = len(data) if num_samples is None else min(num_samples, len(data))
        batch_size = _resolve_batch_size(backend, generator)
        tally = _Tally()

        # Phase 1 — draw ground truths and build prompts over the whole split, ascending
        # index order (see evaluation_qa_dataset for why this must precede the reordering).
        pending = []
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

            pending.append((i, knowledge, dialog, response, ground_truth,
                            build_dialogue_request(dialog, response, instruction)))

        # Phase 1.5 — execution order (longest prompts first), see _batch_order.
        order = _batch_order(pending, backend, generator, sort_by_length)

        for start in range(0, n, batch_size):
            chunk = [pending[i] for i in order[start:min(start + batch_size, n)]]

            # Phase 2 — one padded forward pass for the whole chunk.
            if backend == "hf":
                answers = generator.generate_many([request for *_, request in chunk])
            else:
                answers = [get_dialogue_response(model, dialog, response, instruction,
                                                 backend=backend, generator=generator)
                           for _, _, dialog, response, _, _ in chunk]

            # Phase 3 — score and write, in execution order (each row carries its `index`).
            for (i, knowledge, dialog, response, ground_truth, _), raw in zip(chunk, answers):
                ans = raw.replace(".", "")

                if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                    gen = {"index": i, "knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!", "raw_judgement": raw}
                    dump_jsonl(gen, output_path, append=True)
                    tally.record_failure(ground_truth)
                    print('sample {} fails......'.format(i))
                    continue
                elif "Yes" in ans:
                    if ans != "Yes":
                        ans = "Yes"
                    gen = {"index": i, "knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans, "raw_judgement": raw}
                elif "No" in ans:
                    if ans != "No":
                        ans = "No"
                    gen = {"index": i, "knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": ans, "raw_judgement": raw}
                else:
                    gen = None
                assert (gen is not None)

                tally.record(ground_truth, ans)

                print('sample {} success......'.format(i))
                dump_jsonl(gen, output_path, append=True)

        stats = tally.stats(n)
        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(
            tally.correct, tally.incorrect, stats["accuracy"]))
        _print_diagnostics(stats)
        return stats


def evaluation_summarization_dataset(model, file, instruction, output_path, backend="openai", generator=None, num_samples=None, sort_by_length="desc", seed=None):
    _seed_labels(seed)
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        n = len(data) if num_samples is None else min(num_samples, len(data))
        batch_size = _resolve_batch_size(backend, generator)
        tally = _Tally()

        # Phase 1 — draw ground truths and build prompts over the whole split, ascending
        # index order (see evaluation_qa_dataset for why this must precede the reordering).
        # These documents are the longest prompts in HaluEval, so this is the split most
        # likely to need a smaller --batch-size to stay inside GPU memory — and the one
        # length-sorted batching helps most. Holding all n built prompts costs roughly the
        # size of the documents again on top of `data`, which is already fully resident.
        pending = []
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

            pending.append((i, document, summary, ground_truth,
                            build_summarization_request(document, summary, instruction)))

        # Phase 1.5 — execution order (longest prompts first), see _batch_order.
        order = _batch_order(pending, backend, generator, sort_by_length)

        for start in range(0, n, batch_size):
            chunk = [pending[i] for i in order[start:min(start + batch_size, n)]]

            # Phase 2 — one padded forward pass for the whole chunk.
            if backend == "hf":
                answers = generator.generate_many([request for *_, request in chunk])
            else:
                answers = [get_summarization_response(model, document, summary, instruction,
                                                      backend=backend, generator=generator)
                           for _, document, summary, _, _ in chunk]

            # Phase 3 — score and write, in execution order (each row carries its `index`).
            for (i, document, summary, ground_truth, _), raw in zip(chunk, answers):
                ans = raw.replace(".", "")

                if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                    gen = {"index": i, "document": document, "summary": summary, "ground_truth": ground_truth, "judgement": "failed!", "raw_judgement": raw}
                    dump_jsonl(gen, output_path, append=True)
                    tally.record_failure(ground_truth)
                    print('sample {} fails......'.format(i))
                    continue
                elif "Yes" in ans:
                    if ans != "Yes":
                        ans = "Yes"
                    gen = {"index": i, "document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans, "raw_judgement": raw}
                elif "No" in ans:
                    if ans != "No":
                        ans = "No"
                    gen = {"index": i, "document": document, "summary": summary, "ground_truth": ground_truth, "judgement": ans, "raw_judgement": raw}
                else:
                    gen = None
                assert (gen is not None)

                tally.record(ground_truth, ans)

                print('sample {} success......'.format(i))
                dump_jsonl(gen, output_path, append=True)

        stats = tally.stats(n)
        print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(
            tally.correct, tally.incorrect, stats["accuracy"]))
        _print_diagnostics(stats)
        return stats


def _run_label(model_path):
    """A filename-safe label for a `--model-path`, which is usually a long local path.

    Slashes have always been flattened to underscores; the rest of the reserved set
    matters on Windows, where a drive letter's ":" survives into the filename and NTFS
    reads `qa_c:_Users_...json` as an alternate data stream on a file called `qa_c` —
    the run then completes and reports success while writing a 0-byte results file.
    Linux paths contain none of these characters, so artifact names on the cluster (and
    their comparability with earlier runs) are unchanged.
    """
    label = model_path.replace("/", "_").replace("\\", "_")
    for reserved in ':*?"<>|':
        label = label.replace(reserved, "_")
    return label


def _print_diagnostics(stats):
    """One line that says WHY the accuracy above came out the way it did.

    Printed next to the headline number because the three failure modes are
    indistinguishable from accuracy alone (see `_Tally`).
    """
    print('  format compliance: {}/{} ({:.1%} parsed, {} failed)'.format(
        stats["num_examples"] - stats["num_failed"], stats["num_examples"],
        stats["format_compliance"], stats["num_failed"]))
    print('  TPR (hallucination caught): {:.3f} | TNR (correct output cleared): {:.3f} '
          '| judged "Yes" on {:.1%} of parsed rows'.format(
              stats["tpr"], stats["tnr"], stats["judged_yes_rate"]))


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
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None,
                        help="Max new tokens for the HF judge. Default (unset) picks the value "
                             "matching the OpenAI endpoint each prompt format stands in for: 16 "
                             "for the flat/completion format (openai.Completion's own default) "
                             "and 128 for the chat format (openai.ChatCompletion was called "
                             "without a cap, so a verbose-but-valid judgement was never "
                             "truncated). Too small a budget truncates a judge mid-verdict and "
                             "the row is then scored as 'failed!', i.e. WRONG — see hf_local's "
                             "DEFAULT_MAX_NEW_TOKENS.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1,
                        help="Judge prompts per forward pass (--backend hf only). >1 is much "
                             "faster on a GPU; lower it if you hit CUDA OOM on summarization. "
                             "Keep it fixed across models you intend to compare.")
    parser.add_argument("--sort-by-length", dest="sort_by_length", default="desc",
                        choices=list(SORT_ORDERS),
                        help="Order judge prompts by token length before batching "
                             "(--backend hf only). The pipeline pads each batch to its own "
                             "longest prompt, so grouping similar lengths together removes "
                             "most of the padding waste — the biggest runtime lever on "
                             "summarization. 'desc' runs the peak-memory batch first, so a "
                             "CUDA OOM shows up immediately. Reordering changes batch "
                             "composition, which can flip a greedy argmax tie; keep it FIXED "
                             "across models you compare (it is recorded in summary.json).")
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=None,
                        help="Evaluate only the first N examples (useful for smoke tests). Leave "
                             "unset for a run comparable to the published numbers, which score "
                             "the whole 10,000-row split.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the ground-truth coin flip that decides whether each row "
                             "is shown its hallucinated or its correct output. Unset reproduces "
                             "upstream's unseeded behaviour. Setting it does not change the "
                             "metric's distribution, but it makes runs reproducible and makes "
                             "compared models PAIRED (same label draw). Use the same seed for "
                             "every arm you compare; it is recorded in summary.json.")
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
            batch_size=args.batch_size,
        )

    instruction_file = "{}/{}_evaluation_instruction.txt".format(args.task, args.task)
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()

    model = args.model
    label = _run_label(args.model_path) if (backend == "hf" and args.model_path) else args.model

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

    kwargs = dict(backend=backend, generator=generator, num_samples=args.num_samples,
                  sort_by_length=args.sort_by_length, seed=args.seed)
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
    # Every setting that can move the number is recorded alongside it, so two
    # summaries can be checked for comparability without digging up the launch command.
    summary = {"task": args.task, "model": label, "backend": backend,
               "batch_size": args.batch_size if backend == "hf" else 1,
               "sort_by_length": args.sort_by_length if backend == "hf" else "none",
               "seed": args.seed,
               "num_samples_requested": args.num_samples,
               "max_new_tokens": generator.max_new_tokens if generator is not None else None,
               "prompt_format": generator.prompt_format if generator is not None else "openai"}
    summary.update(stats or {})
    summary_path = os.path.join(results_dir, "{}_{}_summary.json".format(args.task, label))
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print("Summary written to {}".format(summary_path))
