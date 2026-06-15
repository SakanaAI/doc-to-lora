from ctx_to_lora.data.preprocessing_fn import parse_clipper_user_message
from ctx_to_lora.eval_utils import compute_clipper_metrics, extract_clipper_answer


def test_parse_clipper_user_message_splits_context_and_prompt():
    user_message = (
        "You are provided with a context and a statement.\n\n"
        "<context>Book body</context>\n\n"
        "<statement>A claim.</statement>\n\n"
        "<question>Is it true or false?</question>"
    )

    context, prompt = parse_clipper_user_message(user_message)

    assert context == "Book body"
    assert "<statement>A claim.</statement>" in prompt
    assert "<context>" not in prompt


def test_extract_clipper_answer_prefers_answer_tag():
    text = "<explanation>Reasoning mentions false.</explanation><answer>TRUE</answer>"
    assert extract_clipper_answer(text) == "true"


def test_compute_clipper_metrics_reports_pair_accuracy():
    decoded_txts = [
        {
            "generated": "<answer>TRUE</answer>",
            "label": "<answer>TRUE</answer>",
            "clipper_status": "true",
            "clipper_pair_id": "pair-a",
        },
        {
            "generated": "<answer>false</answer>",
            "label": "<answer>FALSE</answer>",
            "clipper_status": "false",
            "clipper_pair_id": "pair-a",
        },
        {
            "generated": "<answer>true</answer>",
            "label": "<answer>TRUE</answer>",
            "clipper_status": "true",
            "clipper_pair_id": "pair-b",
        },
        {
            "generated": "<answer>true</answer>",
            "label": "<answer>FALSE</answer>",
            "clipper_status": "false",
            "clipper_pair_id": "pair-b",
        },
    ]

    metrics, per_sample, counts = compute_clipper_metrics(decoded_txts)

    assert metrics["clipper_accuracy"] == 0.75
    assert metrics["clipper_true_accuracy"] == 1.0
    assert metrics["clipper_false_accuracy"] == 0.5
    assert metrics["clipper_pair_accuracy"] == 0.5
    assert per_sample["clipper_pair_accuracy"] == [1.0, 1.0, 0.0, 0.0]
    assert counts["clipper_pair_accuracy"] == 2
