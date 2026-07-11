# RAGTruth
![](image/fig1.jpg)

RAGTruth is a word-level hallucination corpus in various tasks within the Retrieval-augmented generation (RAG) setting both for **training** and **evaluating**.

RAG has become a main technique for alleviating hallucinations in large language models (LLMs). Despite the integration of RAG, LLMs may still present unsupported or contradictory claims to the retrieved contents. In order to develop effective hallucination prevention strategies under RAG, it is important to create benchmark datasets that can measure the extent of hallucination. RAGTruth comprises nearly 18,000 naturally generated responses from diverse LLMs using RAG. These responses have undergone meticulous manual annotations at both the individual cases and word levels, incorporating evaluations of hallucination intensity.

> **This fork adds a serverless, two-stage evaluation pipeline** so you can point
> RAGTruth at **your own generation model / LoRA** without running a TGI Docker
> server. See [**Evaluation**](#evaluation) below (jump to
> [Run it with your own model](#run-it-with-your-own-generation-model--lora)) and
> [`ARCHITECTURE.md`](./ARCHITECTURE.md). The original TGI-based baseline is
> retained and documented too.

## Updates
1. [2024/06] We released our training and evaluation code. Model weight can be found [here](https://github.com/CodingLL/RAGTruth_Eval/tree/master)
2. [2024/02] We updated the data: we included more annotated hallucinations and added one new meta, `implicit_true`.
3. [2024/01] We released the RAGTruth corpus.
## Dataset
Given that each piece of source information elicits six distinct responses from various Language Models, we publish the source information and its corresponding responses individually.

1. **response.jsonl**

| Field Name  | Field Value | Description                                 |
| ----------- | ----------- | ------------------------------------------- |
| id          | String       | The index of the response.                |
| source_id         | String     | The index of the source information.  |
| model   | String      | The model that generates the response.  |
| temperature           | Float      | The temperature when generating response.          |
| labels           |List[Dict]      | The hallucination spans. `text` is the hallucination itself; `start` & `end` are the position of hallucination span in the response; `label_type` is the type of the hallucination span. `due_to_null` means whether the hallucination is caused by null value. `implicit_true` means this span is correct while the info is not mentioned in the context. `meta` is the comment of the hallucination given by annotators.      |
| split           | String      |  `train` or `test`           |
| quality          | String      | Indicate whether this answer has quality issues. `incorrect_refusal` means the model mistakenly refuses to answer despite the presence of the relevant context. `truncated` means the response is truncated unexpectedly.    |
| response      | String      | The response of LLMs given a certain instruction.  |

Sample:
```json
{
  "id": "1472",
  "source_id": "11316",
  "model": "mistral-7B-instruct",
  "temperature": 0.925,
  "labels": [
    {
      "start": 219,
      "end": 229,
      "text": "Gaza Strip",
      "meta": "HIGH INTRO OF NEW INFO\nIt is not mentioned in the original source that Gaza Strip is included in the occupied Palestinian territory.",
      "label_type": "Evident Baseless Info"
    }
  ],
  "split": "train",
  "quality": "good",
  "response": "The Palestinian Authority has officially become the 123rd member of the International Criminal Court (ICC), giving the court jurisdiction over alleged crimes in Palestinian territories. This includes East Jerusalem and Gaza Strip, which are occupied by Israel. The signing of Rome Statute by Palestinians in January 2021 had already established ICC's jurisdiction over alleged crimes committed \"since June 13, 2014\" in these areas. Now, the court can open a preliminary investigation or formal investigation into the situation in Palestinian territories, potentially leading to war crimes probes against Israeli individuals. However, this could also lead to counter-charges against Palestinians. The ICC welcomed Palestine's accession, while Israel and the US, who are not ICC members, opposed the move."
}
```

2. **source_info.jsonl**

| Field Name  | Field Value | Description                                 |
| ----------- | ----------- | ------------------------------------------- |
| source_id         | String     | The index of the source information.  |
| task_type           | String      | The task type of the data, including QA, Data2txt and Summary         |
| source | String      |  The source of the original content.    |
| source_info | String or Dict     | Base content under RAG setting. For summarization tasks, the value of this field is of the string type; for data-to-text and QA (question-answering) tasks, the value of this field is of the dict type.    |
| prompt | String | The prompt we used to generate responses. For Llama and Mistral models, we used `<s>[INST] {prompt} [/INST]`. |

QA sample:
```json
{
  "source_id": "14312",
  "task_type": "QA",
  "source": "MARCO",
  "source_info": {
    "question": "how to prepare beets and beet greens",
    "passages": "passage 1:Procedures: 1  Preheat oven to 350 degrees Fahrenheit. 2  Wash beets thoroughly, leaving skins on. 3  Place beets in a small baking dish or roasting pan, toss with 2 tablespoons of coconut oil, cover and bake for 45 to 60 minutes or until tender.  For the greens: heat remaining coconut oil in a skillet over medium-low heat.\n\npassage 2:Serve with red wine vinegar or butter and salt and pepper. For the greens: heat remaining coconut oil in a skillet over medium-low heat. Add garlic and onion and cook for one minute. Tear the beet greens into 2 to 3 inch pieces, and add to skillet, stirring until wilted and tender. Season with salt and pepper.\n\npassage 3:Directions See How It's Made. 1  Wash the greens thoroughly several times in deep water. Cook in very little boiling salted water until just tender, a few minutes. 2  Submit a Correction.\n\n"
  },
  "prompt": "Briefly answer the following question:\nhow to prepare beets and beet greens\nBear in mind that your response should be strictly based on the following ten passages:\npassage 1:Procedures: 1  Preheat oven to 350 degrees Fahrenheit. 2  Wash beets thoroughly, leaving skins on. 3  Place beets in a small baking dish or roasting pan, toss with 2 tablespoons of coconut oil, cover and bake for 45 to 60 minutes or until tender.  For the greens: heat remaining coconut oil in a skillet over medium-low heat.\n\npassage 2:Serve with red wine vinegar or butter and salt and pepper. For the greens: heat remaining coconut oil in a skillet over medium-low heat. Add garlic and onion and cook for one minute. Tear the beet greens into 2 to 3 inch pieces, and add to skillet, stirring until wilted and tender. Season with salt and pepper.\n\npassage 3:Directions See How It's Made. 1  Wash the greens thoroughly several times in deep water. Cook in very little boiling salted water until just tender, a few minutes. 2  Submit a Correction.\n\nIn case the passages do not contain the necessary information to answer the question, please reply with: \"Unable to answer based on given passages.\"\noutput:"
}
```

Data2txt sample:
```json
{
  "source_id": "13661",
  "task_type": "Data2txt",
  "source": "Yelp",
  "source_info": {
    "name": "Subway",
    "address": "1940 Cliff Dr, Ste B-13",
    "city": "Santa Barbara",
    "state": "CA",
    "categories": "Restaurants, Sandwiches, Salad, Fast Food",
    "hours": {
      "Monday": "9:0-22:30",
      "Tuesday": "9:0-22:30",
      "Wednesday": "9:0-22:30",
      "Thursday": "9:0-22:30",
      "Friday": "9:0-22:30",
      "Saturday": "9:0-22:30",
      "Sunday": "11:0-22:0"
    },
    "attributes": {
      "BusinessParking": {
        "garage": false,
        "street": false,
        "validated": false,
        "lot": true,
        "valet": false
      },
      "RestaurantsReservations": false,
      "OutdoorSeating": true,
      "WiFi": "no",
      "RestaurantsTakeOut": true,
      "RestaurantsGoodForGroups": true,
      "Music": null,
      "Ambience": {
        "touristy": false,
        "hipster": false,
        "romantic": null,
        "divey": null,
        "intimate": null,
        "trendy": null,
        "upscale": null,
        "classy": null,
        "casual": null
      }
    },
    "business_stars": 3.0,
    "review_info": [
      {
        "review_stars": 1.0,
        "review_date": "2020-05-11 02:07:36",
        "review_text": "My husband and I came in earlier today for lunch after I ordered my sandwich my husband ordered a club and didn't think anything about it while the girl made it because he assumed she knew what went on a club.  Once we got in the car I looked at the receipt and realized she made him a turkey sandwich so we went back in to ask her to add the other meat and to refund us and recharge the correct price since a club is a little more. She was very rude about it and told us she wasn't going to do anything about and proceeded to call us liars and say he asked for a turkey sub. I told her she didn't have to be so rude so she told me to \"get the f**k out b***h\" and if I had a problem with it I could \"call her f***ing manager\". Also she proceeded to cuss us as we walked out of the store. It was quite unacceptable and inappropriate of an employee to be this unprofessional and aggressive."
      },
      {
        "review_stars": 3.0,
        "review_date": "2020-03-02 20:05:55",
        "review_text": "Small store, personnel not very well organized, store is only moderately clean. \nStaff is friendly sometimes, other times they will only barely recognize you."
      },
      {
        "review_stars": 5.0,
        "review_date": "2019-07-10 01:49:07",
        "review_text": "Nice and clean location. Toppings look fresh and well stocked. Joaquin and Odalis are always helpful and friendly."
      }
    ]
  },
  "prompt": "Instruction:\nWrite an objective overview about the following local business based only on the provided structured data in the JSON format. You should include details and cover the information mentioned in the customers' review. The overview should be 100 - 200 words. Don't make up information. Structured data:\n{'name': 'Subway', 'address': '1940 Cliff Dr, Ste B-13', 'city': 'Santa Barbara', 'state': 'CA', 'categories': 'Restaurants, Sandwiches, Salad, Fast Food', 'hours': {'Monday': '9:0-22:30', 'Tuesday': '9:0-22:30', 'Wednesday': '9:0-22:30', 'Thursday': '9:0-22:30', 'Friday': '9:0-22:30', 'Saturday': '9:0-22:30', 'Sunday': '11:0-22:0'}, 'attributes': {'BusinessParking': {'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}, 'RestaurantsReservations': False, 'OutdoorSeating': True, 'WiFi': 'no', 'RestaurantsTakeOut': True, 'RestaurantsGoodForGroups': True, 'Music': None, 'Ambience': {'touristy': False, 'hipster': False, 'romantic': None, 'divey': None, 'intimate': None, 'trendy': None, 'upscale': None, 'classy': None, 'casual': None}}, 'business_stars': 3.0, 'review_info': [{'review_stars': 1.0, 'review_date': '2020-05-11 02:07:36', 'review_text': 'My husband and I came in earlier today for lunch after I ordered my sandwich my husband ordered a club and didn\\'t think anything about it while the girl made it because he assumed she knew what went on a club.  Once we got in the car I looked at the receipt and realized she made him a turkey sandwich so we went back in to ask her to add the other meat and to refund us and recharge the correct price since a club is a little more. She was very rude about it and told us she wasn\\'t going to do anything about and proceeded to call us liars and say he asked for a turkey sub. I told her she didn\\'t have to be so rude so she told me to \"get the f**k out b***h\" and if I had a problem with it I could \"call her f***ing manager\". Also she proceeded to cuss us as we walked out of the store. It was quite unacceptable and inappropriate of an employee to be this unprofessional and aggressive.'}, {'review_stars': 3.0, 'review_date': '2020-03-02 20:05:55', 'review_text': 'Small store, personnel not very well organized, store is only moderately clean. \\nStaff is friendly sometimes, other times they will only barely recognize you.'}, {'review_stars': 5.0, 'review_date': '2019-07-10 01:49:07', 'review_text': 'Nice and clean location. Toppings look fresh and well stocked. Joaquin and Odalis are always helpful and friendly.'}]}\nOverview:"
}
```

Summary sample:
```json
{
  "source_id": "11316",
  "task_type": "Summary",
  "source": "CNN/DM",
  "source_info": "The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed \"in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014.\" Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a move toward greater justice. \"As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice,\" he said, according to an ICC news release. \"Indeed, today brings us closer to our shared goals of justice and peace.\" Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. \"As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly,\" she said. Rights group Human Rights Watch welcomed the development. \"Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court's treaty should speak out to welcome its membership,\" said Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts to undermine international justice, not Palestine's decision to join a treaty to which over 100 countries around the world are members.\" In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it \"strongly\" disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC,\" the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. \"We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,\" it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as \"Palestine.\" While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would \"conduct its analysis in full independence and impartiality.\" The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.\n",
  "prompt": "Summarize the following news within 141 words:\nThe Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed \"in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014.\" Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a move toward greater justice. \"As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice,\" he said, according to an ICC news release. \"Indeed, today brings us closer to our shared goals of justice and peace.\" Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. \"As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly,\" she said. Rights group Human Rights Watch welcomed the development. \"Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court's treaty should speak out to welcome its membership,\" said Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts to undermine international justice, not Palestine's decision to join a treaty to which over 100 countries around the world are members.\" In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it \"strongly\" disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC,\" the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. \"We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,\" it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as \"Palestine.\" While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would \"conduct its analysis in full independence and impartiality.\" The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.\n\noutput:"
}
```

## Data Statistics
#### Descriptive Statistics Devided by Task

Task                  | Instances | Responses | Hallucination Responses | Hallucination Spans |
--- | --- | --- | --- | --- |
Summarization(CNN/DM)      | 628  | 3768   | 1165 | 1474 |
Summarization(Recent News) | 315  | 1890   | 521  | 598 |
Question Answering         | 989  | 5934   | 1724 | 2927 |
Data-to-text               | 1033 | 6198  | 4254 | 9290 |
Overall                    | 2965 | 17790 | 7664 | 14289 |


#### Descriptive Statistics Devided by LLMs

Model | Hallucination Responses | Hallucination Spans |
--- | --- | --- |
GPT-3.5-turbo-0613  | 401  | 533  |
GPT-4-0613          | 406  | 485  |
Llama-2-7B-chat     | 1832 | 3302 |
Llama-2-13B-chat    | 1677 | 3799 |
Llama-2-70B-chat    | 1395 | 2608 |
Mistral-7B-Instruct | 1953 | 3562 |

## Evaluation

There are **two** ways to evaluate in this repo:

1. The **original baseline** (train your own detector, serve it via a TGI Docker
   server, predict + score) — retained for reproduction.
2. A **new serverless, two-stage in-process pipeline** — no server, no Docker;
   run RAGTruth end-to-end against **your own generation model** plus a
   hallucination detector. This is the recommended path and where "run it with
   your own model" lives.

### How RAGTruth evaluation works

RAGTruth is a *detection* corpus: given a RAG `(reference, response)` pair, a
**detector model** outputs the hallucinated spans as
`{"hallucination list": [...]}`. Two things can therefore be measured:

- **Hallucination rate of a generation model** — have your model produce a RAG
  response for each source item, then have the detector flag it. The fraction of
  responses flagged (overall and per task type) characterizes *your* model.
- **Detector quality (paper reproduction)** — run the detector on the corpus's
  *original* responses and compare its predictions to the gold `labels`
  (precision / recall / F1).

### Original baseline pipeline (TGI, retained)

The upstream flow, in [`baseline/`](./baseline) (see also
[`baseline/readme.md`](./baseline/readme.md)):

1. **Prepare** — `python baseline/prepare_dataset.py` merges `response.jsonl`
   (good-quality) with `source_info.jsonl`, splits train/dev by `source_id`, and
   writes `train.jsonl` / `dev.jsonl` / `test.jsonl`.
2. **Train** a detector — `baseline/train.py` fine-tunes a Llama-2 model to emit
   the `{"hallucination list": [...]}` JSON for each `[INST]`-wrapped example
   (`baseline/dataset.py`).
3. **Serve via TGI** — the trained detector is served with HuggingFace
   `text-generation-inference` in Docker:
   ```bash
   docker run -d --name baseline --gpus '"device=7"' -v $PWD:/data --shm-size 1g \
       -p 8300:80 ghcr.io/huggingface/text-generation-inference:2.0.1 \
       --model-id /data/exp/baseline --dtype bfloat16 --max-total-tokens 8000 \
       --sharded false --max-input-length 4095
   ```
4. **Predict + evaluate** — `baseline/predict_and_evaluate.py` fires many async
   requests at the server and computes case-level recall/precision/F1.

#### Why `http://127.0.0.1:8300` (localhost)?

`predict_and_evaluate.py` talks to `AsyncInferenceClient(model="http://127.0.0.1:8300")`.
That is **not** a remote service — it is the TGI container running on the *same
machine*:

- `docker run ... -p 8300:80` **publishes** the container's internal port `80`
  (where TGI's HTTP server listens) to port `8300` on the **host**. So from the
  host, `127.0.0.1:8300` reaches the containerized server.
- Client (`predict_and_evaluate.py`) and server (the container) run on the **same
  node**, so the loopback address `127.0.0.1` is all that's needed — no network,
  no auth.

**Why a server at all, instead of just calling the model in-process?** Throughput.
TGI does continuous batching and token streaming, so the client can keep many
requests in flight — here an `asyncio.Semaphore(80)` lets up to **80 concurrent**
generations be serviced and batched together, which is far faster than a Python
`for` loop calling `model.generate` one prompt at a time. The cost is the
operational overhead of building an image, launching a GPU container, and
managing the port — exactly what the new pipeline removes.

> **The `--tokenizer` flag is nearly vestigial.** `predict_and_evaluate.py` builds
> the prompt as a plain string and sends it to TGI, which tokenizes server-side
> using the *served model's* tokenizer. The client-side `AutoTokenizer` loaded
> from `--tokenizer` is passed to `process_dialog_to_single_turn(..., return_prompt=True)`
> only to satisfy its signature — with `return_prompt=True` the tokenizer is never
> actually used. The upstream code even notes this ("actually we do not need
> tokenizer / just to meet the parameter requirements").

### New serverless two-stage pipeline

The new [`src/ragtruth_eval/`](./src/ragtruth_eval) package runs everything
**in-process** — no TGI, no Docker, no localhost port. Two models, two stages:

- **Stage 1 – generate** (`--stage generate`): load your generation model, produce
  a RAG response for each source item, stream to `outputs/<run>/generations.jsonl`.
- **Stage 2 – detect** (`--stage detect`): load the detector model, read the
  generations, emit `{"hallucination list": [...]}`, write
  `outputs/<run>/detections.jsonl` and `outputs/<run>/summary.json` (the
  hallucination-rate summary).

`--stage all` runs both. The detector defaults to the same decoding params the
released detector was trained/served with; malformed JSON is repaired locally
(the in-process greedy path can't just retry generation like the TGI client did).

Install:

```bash
pip install -e .            # installs the `ragtruth-eval` console script
```

#### Run it with your own generation model / LoRA

Pass your checkpoint as `--model-id` (a Hub id, a local full-model path, or a
PEFT/LoRA adapter such as a `final_checkpoint` from the sibling `SP-DPO-Base`
pipeline). The detector is `--detector-model-id` — the RAGTruth authors' released
weights [`CodingLL/RAGTruth_Eval`](https://github.com/CodingLL/RAGTruth_Eval), or
your own trained detector.

```bash
# End-to-end: your model generates, the released detector flags:
python src/run_eval.py --stage all \
    --model-id /path/to/final_checkpoint \
    --detector-model-id CodingLL/RAGTruth_Eval \
    --output-dir outputs/my-run

# equivalently, via the installed console script:
ragtruth-eval --stage all --model-id /path/to/final_checkpoint \
    --detector-model-id CodingLL/RAGTruth_Eval

# A LoRA adapter is auto-detected (adapter_config.json) and merged onto its base:
python src/run_eval.py --stage all --model-id /path/to/lora_adapter \
    --detector-model-id CodingLL/RAGTruth_Eval --num-samples 20

# Run the two stages separately (e.g. generate on one node, detect on another):
python src/run_eval.py --stage generate --model-id <gen> --output-dir outputs/my-run
python src/run_eval.py --stage detect --detector-model-id <detector> --output-dir outputs/my-run
```

Useful flags (`ragtruth-eval --help` for all):

| Flag | Purpose |
| --- | --- |
| `--stage` | `generate`, `detect`, or `all` |
| `--model-id` / `--base-model-id` / `--tokenizer-id` | generation model (path/Hub/LoRA) |
| `--detector-model-id` (+ `--detector-base-model-id` / `--detector-tokenizer-id`) | detector model |
| `--split` | filter source items to `train`/`dev`/`test` (via `response.jsonl`) |
| `--num-samples N` | use only the first N items (smoke tests) |
| `--task-types` | restrict to `QA` / `Summary` / `Data2txt` |
| `--gold-f1` | reproduction mode: score the detector vs. gold labels (P/R/F1), skips Stage 1 |
| `--output-dir` | where `generations.jsonl` / `detections.jsonl` / `summary.json` go |

A local `--model-id` / `--detector-model-id` that doesn't exist fails fast with a
clear `FileNotFoundError` instead of a confusing Hub-lookup error.

#### Reproduce the paper's detector F1 (no server)

`--gold-f1` runs the detector over the corpus's *original* responses + gold labels
and reports precision/recall/F1 per task — the same numbers
`baseline/predict_and_evaluate.py` produced, but in-process:

```bash
python src/run_eval.py --stage detect --gold-f1 --split test \
    --detector-model-id CodingLL/RAGTruth_Eval --output-dir outputs/repro
```

### Install (HPC)

On the cluster use the pinned Linux variant
[`pyproject-HPC.toml`](./pyproject-HPC.toml) (torch 2.2.2, transformers 4.41,
Python 3.12; matches the known-good `topollm` / `SP-DPO-Base` env — **no vLLM,
no TGI**). With `uv`:

```bash
module load Python/3.12.3 uv/0.10.2 CUDA/12.6.1   # adjust to available modules
cp pyproject-HPC.toml pyproject.toml              # or: uv sync --project pyproject-HPC.toml
uv sync
```

> **Mirror / offline:** compute nodes have no internet. Uncomment the
> `[[tool.uv.index]]` block in `pyproject-HPC.toml`, point it at the cluster PyPI
> mirror, and pre-download **both** the generation model and the detector
> (`CodingLL/RAGTruth_Eval`) into the HF cache on a login node; then export
> `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` on the compute
> nodes. Removing the TGI server is what makes this benchmark runnable on a stock
> SLURM GPU node.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the two-stage data flow, the two
model roles, and how the retained `baseline/` trainer relates to the new package.

## Citation

Please cite our paper if you use our dataset:
```bibtex
@misc{wu2023ragtruth,
      title={RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models}, 
      author={Yuanhao Wu and Juno Zhu and Siliang Xu and Kashun Shum and Cheng Niu and Randy Zhong and Juntong Song and Tong Zhang},
      year={2023},
      eprint={2401.00396},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ParticleMedia/RAGTruth&type=Date)](https://star-history.com/#ParticleMedia/RAGTruth&Date)
