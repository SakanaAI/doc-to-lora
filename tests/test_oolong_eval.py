from collections import Counter

from datasets import Dataset

from ctx_to_lora.data.preprocessing_fn import get_preprocessing_fn
from ctx_to_lora.data.processing import (
    OOLONG_MAX_CONTEXT_LEN,
    filter_dataset_by_max_value,
    stratified_subsample_dataset,
)
from ctx_to_lora.eval_utils import compute_oolong_metrics, extract_oolong_answer


def test_oolong_preprocessing_maps_core_fields_and_metadata():
    preprocess = get_preprocessing_fn("oolong-synth", is_eval=True)
    sample = {
        "id": 7,
        "dataset": "spam",
        "context_window_text": "context body",
        "question": "Give your final answer in the form 'Label: answer'.",
        "task_group": "classification",
        "task": "most_freq",
        "answer": "['ham']",
        "answer_type": "ANSWER_TYPE.LABEL",
        "input_subset": "all",
        "num_labels": 2,
        "context_window_id": 101,
    }

    processed = preprocess(sample)

    assert processed["context"] == "context body"
    assert processed["prompts"] == [sample["question"]]
    assert processed["responses"] == ["['ham']"]
    assert processed["oolong_task"] == "most_freq"
    assert processed["oolong_answer_type"] == "ANSWER_TYPE.LABEL"
    assert processed["oolong_context_window_id"] == 101


def test_extract_oolong_answer_prefers_suffix_after_colon():
    answer, confidence = extract_oolong_answer("Reasoning...\nLabel: [ham]")

    assert answer == "ham"
    assert confidence == "vhigh"


def test_compute_oolong_metrics_handles_exact_numeric_and_date_scoring():
    decoded_txts = [
        {
            "generated": "Label: ham",
            "label": "['ham']",
            "oolong_answer_type": "ANSWER_TYPE.LABEL",
        },
        {
            "generated": "Answer: 8",
            "label": "['10']",
            "oolong_answer_type": "ANSWER_TYPE.NUMERIC",
        },
        {
            "generated": "Date: 2025-02-07",
            "label": "[datetime.date(2025, 2, 7)]",
            "oolong_answer_type": "ANSWER_TYPE.DATE",
        },
    ]

    metrics, per_sample, counts = compute_oolong_metrics(decoded_txts)

    assert metrics["oolong_score"] == (1.0 + (0.75**2) + 1.0) / 3
    assert metrics["oolong_parse_rate"] == 1.0
    assert per_sample["oolong_score"] == [1.0, 0.75**2, 1.0]
    assert counts["oolong_score"] == 3


def test_stratified_subsample_dataset_balances_context_len_groups():
    ds = Dataset.from_dict(
        {
            "context_len": [1] * 800 + [2] * 50 + [3] * 800,
            "row_id": list(range(1650)),
        }
    )

    sampled = stratified_subsample_dataset(
        ds=ds,
        group_column="context_len",
        max_samples=1000,
        seed=42,
    )
    counts = Counter(sampled["context_len"])

    assert len(sampled) == 1000
    assert counts[2] == 50
    assert counts[1] == 475
    assert counts[3] == 475


def test_filter_dataset_by_max_value_drops_over_64k_context_len_rows():
    ds = Dataset.from_dict(
        {
            "context_len": [1024, OOLONG_MAX_CONTEXT_LEN, OOLONG_MAX_CONTEXT_LEN + 1],
            "row_id": [0, 1, 2],
        }
    )

    filtered = filter_dataset_by_max_value(
        ds=ds,
        column="context_len",
        max_value=OOLONG_MAX_CONTEXT_LEN,
    )

    assert filtered["context_len"] == [1024, OOLONG_MAX_CONTEXT_LEN]
