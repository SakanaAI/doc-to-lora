from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import EvalPrediction


def compute_rouge(pred_texts, label_texts):
    out = defaultdict(list)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for pred_text, label_text in zip(pred_texts, label_texts):
        scores = scorer.score(pred_text, label_text)
        for k, v in scores.items():
            out[f"{k}.f1"].append(v.fmeasure)
    for k in out:
        out[k] = np.mean(out[k])
    return out


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
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
    loss = (loss * valid_masks).sum(dim=1) / valid_masks.sum(dim=1)
    preplexities = torch.exp(loss)
    return {
        "perplexities": preplexities.tolist(),
        "n_perplexities": valid_masks.shape[0],
    }


class Evaluator:
    def __init__(self, metric_fns: list[Callable]):
        self.metric_fns = metric_fns
        self.reset()

    def reset(self):
        self.accum_metrics = defaultdict(list)
        self.count = defaultdict(list)

    def update(self, shift_logits, shift_labels, valid_masks):
        for metric_fn in self.metric_fns:
            metric = metric_fn(shift_logits, shift_labels, valid_masks)
            for k, v in metric.items():
                if k.startswith("n_"):
                    self.count[k[2:]].append(v)
                else:
                    self.accum_metrics[k] += v

    def compute(self):
        # Get result across entire eval set
        result = {
            k: np.sum(v) / np.sum(self.count[k]) for k, v in self.accum_metrics.items()
        }
        # Reset batch statistics
        self.reset()
        return result


@torch.no_grad()
def compute_metrics(
    eval_pred: EvalPrediction,
    compute_result: bool,
    evaluator: Evaluator,
) -> dict | None:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid_masks = torch.where(shift_labels != -100, 1, 0)
    evaluator.update(shift_logits, shift_labels, valid_masks)
    if compute_result:
        return evaluator.compute()
