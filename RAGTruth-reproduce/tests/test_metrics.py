from ragtruth_eval.metrics import gold_f1, hallucination_rate
from ragtruth_eval.prompts import build_detector_prompt, parse_hallucination_list


def _det(task, pred, gold=None, parse_failed=False):
    d = {"task_type": task, "pred_halu": pred, "parse_failed": parse_failed}
    if gold is not None:
        d["gold_halu"] = gold
    return d


def test_hallucination_rate_overall_and_per_task():
    detections = [
        _det("QA", True),
        _det("QA", False),
        _det("Summary", True),
        _det("Summary", True),
    ]
    result = hallucination_rate(detections)
    assert result["total"] == 4
    assert result["flagged"] == 3
    assert result["hallucination_rate"] == 0.75
    assert result["per_task"]["QA"]["hallucination_rate"] == 0.5
    assert result["per_task"]["Summary"]["hallucination_rate"] == 1.0


def test_hallucination_rate_empty_is_zero_not_error():
    result = hallucination_rate([])
    assert result["total"] == 0
    assert result["hallucination_rate"] == 0.0


def test_hallucination_rate_counts_parse_failures():
    detections = [_det("QA", False, parse_failed=True), _det("QA", True)]
    result = hallucination_rate(detections)
    assert result["parse_failures"] == 1
    assert result["flagged"] == 1


def test_gold_f1_perfect_prediction():
    detections = [
        _det("QA", True, gold=True),
        _det("QA", False, gold=False),
        _det("Summary", True, gold=True),
    ]
    result = gold_f1(detections)
    assert result["overall"]["precision"] == 1.0
    assert result["overall"]["recall"] == 1.0
    assert result["overall"]["f1"] == 1.0


def test_gold_f1_mixed():
    # tp=1 (pred T, gold T), fp=1 (pred T, gold F), fn=1 (pred F, gold T)
    detections = [
        _det("QA", True, gold=True),
        _det("QA", True, gold=False),
        _det("QA", False, gold=True),
    ]
    result = gold_f1(detections)
    assert result["overall"]["precision"] == 0.5
    assert result["overall"]["recall"] == 0.5
    assert result["overall"]["f1"] == 0.5


def test_parse_hallucination_list_clean_json():
    spans, ok = parse_hallucination_list('{"hallucination list": ["a span", "b span"]}')
    assert ok
    assert spans == ["a span", "b span"]


def test_parse_hallucination_list_empty():
    spans, ok = parse_hallucination_list('{"hallucination list": []}')
    assert ok
    assert spans == []


def test_parse_hallucination_list_with_surrounding_text():
    raw = 'Output: {"hallucination list": ["oops"]} trailing junk'
    spans, ok = parse_hallucination_list(raw)
    assert ok
    assert spans == ["oops"]


def test_parse_hallucination_list_unparseable():
    spans, ok = parse_hallucination_list("the model rambled without any json")
    assert not ok
    assert spans == []


def test_build_detector_prompt_qa_includes_question_and_inst_wrap():
    item = {
        "task_type": "QA",
        "question": "what is x?",
        "reference": "passage text",
        "response": "the answer",
    }
    prompt = build_detector_prompt(item)
    assert prompt.startswith("[INST]")
    assert prompt.rstrip().endswith("[/INST]")
    assert "what is x?" in prompt
    assert "the answer" in prompt
