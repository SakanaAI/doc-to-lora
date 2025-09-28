import re
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import EvalPrediction

LENGTH_BINS = [
    # finegrain bins
    (0, 2**7 - 1),
    (2**7, 2**8 - 1),
    (2**8, 2**9 - 1),
    # coarse bins
    (0, 2**9 - 1),
    (2**9, 2**10 - 1),
    (2**10, 2**11 - 1),
    (2**11, 2**12 - 1),
    (2**12, 2**13 - 1),
    (0, 2**13 - 1),
    (2**13, 2**14 - 1),
    (2**14, 2**15 - 1),
    (2**15, float("inf")),
]


def get_length_bin(length: int):
    """Get the length bin for a given length."""
    for i, (start, end) in enumerate(LENGTH_BINS):
        if start <= length < end:
            return (start, end)


def compute_gsm8k_acc(pred_texts, label_texts):
    out = defaultdict(list)
    for pred_text, label_text in zip(pred_texts, label_texts):
        label_ans = float(label_text.split("####")[-1].strip().replace(",", ""))
        # find all the numbers (including decimals) in the string
        numbers = re.findall(r"\d+\.?\d*", pred_text)

        if not numbers:
            out["gsm8k_acc"].append(0)
            continue

        prediction = float(numbers[-1].rstrip(".").strip().replace(",", ""))

        if label_ans == prediction:
            out["gsm8k_acc"].append(1)
        else:
            out["gsm8k_acc"].append(0)

    out_mean = dict()
    for k in out:
        out_mean[k] = np.mean(out[k])
    return out_mean, out


def compute_macro_f1_score(
    pred_texts: list[str],
    label_texts: list[str],
    choices: list[list[str]],  # per-sample choices
    tasks: list[str],  # per-sample task name
):
    """
    Compute macro-averaged precision/recall/F1 for multi-class classification.
    - Parses labels/preds using:
        * Prefer token immediately following case-insensitive "Answer:" / "Answer："
        * Otherwise exact token match of any provided choice (case-insensitive, token-boundary)
    - Invalid predictions count as errors for accuracy and as FN for the true class.
    - Macro metrics exclude zero-support classes *within each task*.
    - Final macro scores are the simple average across tasks (each task has equal weight).
    Returns:
      out_mean: dict with task-averaged macro metrics + per-task metrics like macro_f1_<task_slug>
      out_details: dict with:
          - "accuracy": per-sample correctness (0/1)
          - "per_task": raw per-task metric dicts
    """
    import re
    from collections import defaultdict

    import numpy as np

    assert len(pred_texts) == len(label_texts) == len(choices) == len(tasks), (
        "All input lists must have the same length."
    )

    # --- helpers ---
    def _norm_label(text: str) -> str:
        m = re.search(r"answer\s*[:：]\s*([^\n\r]+)", text, flags=re.I)
        cand = (m.group(1) if m else text).strip().rstrip(".")
        return cand

    def _extract_choice(text: str, per_sample_choices: list[str]):
        # 1) token after "Answer:"
        m = re.search(r"answer\s*[:：]\s*([A-Za-z0-9\-\._]+)", text, flags=re.I)
        if m:
            raw = m.group(1).strip().rstrip(".")
            for ch in per_sample_choices:
                if raw.lower() == ch.lower():
                    return ch
        # 2) exact token match (case-insensitive) with boundaries
        for ch in per_sample_choices:
            pattern = rf"(?<!\w){re.escape(ch)}(?!\w)"
            if re.search(pattern, text, flags=re.I):
                return ch
        return None

    def _choice_to_idx(val: str | None, choice_list: list[str]) -> int:
        if val is None:
            return -1
        for i, ch in enumerate(choice_list):
            if val.lower() == ch.lower():
                return i
        return -1

    def _slugify(name: str) -> str:
        # lowercase, replace non-alnum with underscores, collapse repeats, trim edges
        s = re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower())
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "task"

    # Normalize ground-truth labels
    norm_labels = [_norm_label(lbl) for lbl in label_texts]

    # Group indices by task
    by_task = defaultdict(list)
    for i, tname in enumerate(tasks):
        by_task[tname].append(i)

    per_sample_correct = [0] * len(pred_texts)
    task_metrics = {}  # name -> dict

    # Compute per-task metrics
    for tname, idxs in by_task.items():
        # Union of classes across samples in this task
        class_set = []
        seen = set()
        for i in idxs:
            for ch in choices[i]:
                key = ch.lower()
                if key not in seen:
                    seen.add(key)
                    class_set.append(ch)

        y_true_idx, y_pred_idx = [], []
        for i in idxs:
            t_idx = _choice_to_idx(norm_labels[i], class_set)
            pred_choice = _extract_choice(pred_texts[i], choices[i])
            p_idx = _choice_to_idx(pred_choice, class_set)
            if p_idx == -1:
                # debug logging if desired
                print(
                    f"Extracted {pred_choice} from {pred_texts[i]} which is not in {class_set}"
                )

            if t_idx == -1:
                print(f"True label {norm_labels[i]} not in class set {class_set}")

            y_true_idx.append(t_idx)
            y_pred_idx.append(p_idx)

            per_sample_correct[i] = int(t_idx != -1 and p_idx == t_idx)

        # Per-class stats
        n_classes = len(class_set)
        f1s, precs, recs, supports = [], [], [], []
        for ci in range(n_classes):
            tp = sum(1 for t, p in zip(y_true_idx, y_pred_idx) if t == ci and p == ci)
            fp = sum(
                1
                for t, p in zip(y_true_idx, y_pred_idx)
                if p == ci and t != ci and p != -1
            )
            fn = sum(1 for t, p in zip(y_true_idx, y_pred_idx) if t == ci and p != ci)
            support = sum(1 for t in y_true_idx if t == ci)
            supports.append(support)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )

            precs.append(precision)
            recs.append(recall)
            f1s.append(f1)

        present = [i for i, s in enumerate(supports) if s > 0]
        if present:
            task_macro_f1 = float(np.mean([f1s[i] for i in present]))
            task_macro_prec = float(np.mean([precs[i] for i in present]))
            task_macro_rec = float(np.mean([recs[i] for i in present]))
        else:
            task_macro_f1 = task_macro_prec = task_macro_rec = 0.0

        task_acc = (
            float(
                np.mean(
                    [
                        1 if (t != -1 and p == t) else 0
                        for t, p in zip(y_true_idx, y_pred_idx)
                    ]
                )
            )
            if idxs
            else 0.0
        )

        # --- valid sample ratios (both label and prediction valid) ---
        valid_both = [
            1 if (t != -1 and p != -1) else 0 for t, p in zip(y_true_idx, y_pred_idx)
        ]
        task_valid_sample_ratio = float(np.mean(valid_both)) if idxs else 0.0

        task_metrics[tname] = {
            "macro_f1": task_macro_f1,
            "macro_precision": task_macro_prec,
            "macro_recall": task_macro_rec,
            "accuracy": task_acc,
            "valid_sample_ratio": task_valid_sample_ratio,
            "n_samples": len(idxs),
            "n_classes_present": len(present),
        }

    # Aggregate across tasks (equal weight per task)
    task_names = list(task_metrics.keys())
    if task_names:
        macro_f1_over_tasks = float(
            np.mean([task_metrics[t]["macro_f1"] for t in task_names])
        )
        macro_prec_over_tasks = float(
            np.mean([task_metrics[t]["macro_precision"] for t in task_names])
        )
        macro_rec_over_tasks = float(
            np.mean([task_metrics[t]["macro_recall"] for t in task_names])
        )
        acc_over_tasks = float(
            np.mean([task_metrics[t]["accuracy"] for t in task_names])
        )
        valid_ratio_task_avg = float(
            np.mean([task_metrics[t]["valid_sample_ratio"] for t in task_names])
        )
    else:
        macro_f1_over_tasks = macro_prec_over_tasks = macro_rec_over_tasks = (
            acc_over_tasks
        ) = valid_ratio_task_avg = 0.0

    out_mean = {
        # Task-averaged (equal task weights)
        "macro_f1": macro_f1_over_tasks,
        "macro_precision": macro_prec_over_tasks,
        "macro_recall": macro_rec_over_tasks,
        "accuracy_task_avg": acc_over_tasks,
        "valid_sample_ratio_task_avg": valid_ratio_task_avg,
    }

    # --- add per-task metrics to out_mean with slugified keys ---
    for tname, metrics in task_metrics.items():
        slug = _slugify(tname)
        out_mean[f"macro_f1_{slug}"] = metrics["macro_f1"]
        # out_mean[f"macro_precision_{slug}"] = metrics["macro_precision"]
        # out_mean[f"macro_recall_{slug}"] = metrics["macro_recall"]
        # out_mean[f"accuracy_{slug}"] = metrics["accuracy"]
        out_mean[f"valid_sample_ratio_{slug}"] = metrics["valid_sample_ratio"]
        # out_mean[f"n_samples_{slug}"] = metrics["n_samples"]
        # out_mean[f"n_classes_present_{slug}"] = metrics["n_classes_present"]

    out_details = {
        "accuracy": per_sample_correct,  # per-sample 0/1
    }

    return out_mean, out_details


def compute_rouge(pred_texts, label_texts):
    out = defaultdict(list)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for pred_text, label_text in zip(pred_texts, label_texts):
        scores = scorer.score(pred_text, label_text)
        for k, v in scores.items():
            out[f"{k}.f1"].append(v.fmeasure)
    out_mean = dict()
    for k in out:
        out_mean[k] = np.mean(out[k])
    return out_mean, out


@torch.inference_mode()
def compute_per_token_acc(shift_logits, shift_labels, valid_masks):
    indices = torch.where(valid_masks)
    acc = (shift_logits.argmax(-1) == shift_labels)[indices].float()
    return {
        "per_token_accs": acc.flatten().tolist(),
        "n_per_token_accs": valid_masks.sum().item(),
    }


@torch.inference_mode()
def compute_prefix_matching(shift_logits, shift_labels, valid_masks):
    lengths = valid_masks.sum(dim=1)

    is_wrong = (shift_logits.argmax(-1) != shift_labels) * valid_masks
    is_correct = (shift_logits.argmax(-1) == shift_labels) * valid_masks
    # NOTE: not reliable for multi-turn conversations
    # ie, all tokens in the following user's turn will be correct
    # still monotonically correlate with perf though
    wrong_pos = torch.argmax(is_wrong, dim=1) - torch.argmax(valid_masks, dim=1)
    perf = wrong_pos / lengths

    # if all tokens are correct, set to 1
    perf = torch.where(is_correct.sum(dim=1) == lengths, 1, perf)
    return {
        "prefix_matchings": perf.tolist(),
        "n_prefix_matchings": valid_masks.shape[0],
    }


@torch.inference_mode()
def compute_perplexity(shift_logits, shift_labels, valid_masks):
    return {"perplexities_ph": [1], "n_perplexities_ph": 1}
    # loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
    # loss = (loss * valid_masks).sum(dim=1) / valid_masks.sum(dim=1)
    # preplexities = torch.exp(loss)
    # return {
    #     "perplexities": preplexities.tolist(),
    #     "n_perplexities": valid_masks.shape[0],
    # }


class Evaluator:
    def __init__(self, metric_fns: list[Callable]):
        self.metric_fns = metric_fns
        self.reset()

    def reset(self):
        self.accum_metrics = defaultdict(lambda: list((0,)))
        self.count = defaultdict(lambda: list((0,)))

    def update(self, shift_logits, shift_labels, valid_masks, lengths=None):
        for metric_fn in self.metric_fns:
            # overall metric
            metric = metric_fn(shift_logits, shift_labels, valid_masks)
            for k, v in metric.items():
                key = k if not k.startswith("n_") else k[2:]
                if k.startswith("n_"):
                    # prefix "n_" indicates the count of the metric
                    self.count[key].append(v)
                else:
                    self.accum_metrics[key] += v
                for start, end in LENGTH_BINS:
                    key_w_len = f"{key}_len_{start}-{end}"
                    if key_w_len not in self.accum_metrics:
                        # add key here so that it shows up in the output
                        self.accum_metrics[key_w_len] = [0]
                        self.count[key_w_len] = [0]
            # split samples into length groups, calculate metric for each group
            if lengths is not None:
                for start, end in LENGTH_BINS:
                    logits, labels, masks = [], [], []

                    for logit, label, m, len in zip(
                        shift_logits, shift_labels, valid_masks, lengths
                    ):
                        if isinstance(len, torch.Tensor):
                            len = len.item()
                        if start <= len < end:
                            logits.append(logit)
                            labels.append(label)
                            masks.append(m)

                    if not logits:
                        continue

                    metric = metric_fn(
                        torch.stack(logits), torch.stack(labels), torch.stack(masks)
                    )
                    for k, v in metric.items():
                        if k.startswith("n_"):
                            key = f"{k[2:]}_len_{start}-{end}"
                            self.count[key].append(v)
                        else:
                            key = f"{k}_len_{start}-{end}"
                            self.accum_metrics[key] += v

    def compute(self):
        # Get result across entire eval set
        result = {
            k: np.sum(v) / np.sum(self.count[k]) if len(v) > 1 else "None"
            for k, v in self.accum_metrics.items()
        }
        # for k, v in self.count.items():
        #     if "_len_" in k:
        #         result[k.replace("_len_", "_num_samples_len_")] = sum(v)
        #     else:
        #         result[k + "_num_samples"] = sum(v)
        # Reset batch statistics
        self.reset()
        return result


@torch.no_grad()
def compute_metrics(
    eval_pred: EvalPrediction,
    compute_result: bool,
    evaluator: Evaluator,
) -> dict | None:
    inputs = eval_pred.inputs
    len_key = "ctx_ids_len" if "ctx_ids_len" in inputs else "input_ids_len"
    lengths = inputs[len_key]
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid_masks = torch.where(shift_labels != -100, 1, 0)
    evaluator.update(shift_logits, shift_labels, valid_masks, lengths)
    if compute_result:
        return evaluator.compute()
