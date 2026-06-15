import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ctx_to_lora.eval_utils import eval_generation
from ctx_to_lora.lora_stats import DenseLoraMagnitudeAggregator


def _manual_dense_stats(combined_loras, scaling):
    overall = []
    by_module = {}
    for module_name, module_lora in combined_loras.items():
        module_values = []
        A = module_lora["A"]
        B = module_lora["B"]
        for batch_idx in range(A.shape[0]):
            for layer_idx in range(A.shape[1]):
                dense = (
                    torch.einsum(
                        "ro,ri->oi",
                        B[batch_idx, layer_idx],
                        A[batch_idx, layer_idx],
                    )
                    * scaling
                )
                module_values.append(dense.abs().reshape(-1))
        by_module[module_name] = torch.cat(module_values)
        overall.append(by_module[module_name])
    return torch.cat(overall), by_module


class DenseLoraMagnitudeTests(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for dense LoRA magnitude computation.",
    )
    def test_dense_lora_magnitude_aggregator_matches_manual_batched_dense_update(self):
        scaling = 1.5
        combined_loras = {
            "down_proj": {
                "A": torch.tensor(
                    [
                        [
                            [[1.0, -2.0], [0.5, 3.0]],
                            [[-1.0, 2.0], [4.0, -0.5]],
                        ],
                        [
                            [[2.0, 1.0], [1.5, -1.0]],
                            [[0.25, -3.0], [2.0, 0.75]],
                        ],
                    ]
                ),
                "B": torch.tensor(
                    [
                        [
                            [[2.0, -1.0, 0.5], [1.0, 0.0, -2.0]],
                            [[0.5, 1.5, -1.0], [2.0, -0.5, 1.0]],
                        ],
                        [
                            [[-1.0, 2.0, 0.25], [0.5, -1.5, 3.0]],
                            [[1.0, 0.0, 2.0], [-2.0, 1.0, -0.5]],
                        ],
                    ]
                ),
            },
            "up_proj": {
                "A": torch.tensor(
                    [
                        [
                            [[0.5, 1.0], [2.0, -1.0]],
                            [[1.0, -0.5], [0.25, 3.0]],
                        ],
                        [
                            [[-2.0, 0.5], [1.0, 1.5]],
                            [[0.75, -1.25], [2.5, 0.5]],
                        ],
                    ]
                ),
                "B": torch.tensor(
                    [
                        [
                            [[1.0, -2.0], [0.5, 1.5]],
                            [[-1.0, 2.0], [3.0, 0.25]],
                        ],
                        [
                            [[2.0, 1.0], [-0.5, 0.5]],
                            [[1.5, -1.0], [0.0, 2.0]],
                        ],
                    ]
                ),
            },
        }

        aggregator = DenseLoraMagnitudeAggregator(scaling=scaling, out_chunk_size=2)
        aggregator.update(combined_loras)

        overall_manual, by_module_manual = _manual_dense_stats(combined_loras, scaling)
        stats = aggregator.to_dict()

        self.assertEqual(stats["num_examples"], 2)
        self.assertEqual(stats["num_batches"], 1)
        self.assertEqual(stats["num_layer_matrices"], 4)

        self.assertAlmostEqual(
            stats["overall"]["mean"], overall_manual.mean().item(), places=6
        )
        self.assertAlmostEqual(
            stats["overall"]["min"], overall_manual.min().item(), places=6
        )
        self.assertAlmostEqual(
            stats["overall"]["max"], overall_manual.max().item(), places=6
        )
        self.assertAlmostEqual(
            stats["overall"]["std"],
            overall_manual.std(unbiased=False).item(),
            places=6,
        )

        for module_name, module_values in by_module_manual.items():
            module_stats = stats["by_module"][module_name]
            self.assertAlmostEqual(
                module_stats["mean"], module_values.mean().item(), places=6
            )
            self.assertAlmostEqual(
                module_stats["min"], module_values.min().item(), places=6
            )
            self.assertAlmostEqual(
                module_stats["max"], module_values.max().item(), places=6
            )
            self.assertAlmostEqual(
                module_stats["std"], module_values.std(unbiased=False).item(), places=6
            )

    def test_eval_generation_saves_dense_lora_magnitude_json_and_metrics(self):
        dense_stats = {
            "overall": {"count": 8, "mean": 1.25, "min": 0.0, "max": 4.0, "std": 1.0},
            "by_module": {
                "down_proj": {
                    "count": 8,
                    "mean": 1.25,
                    "min": 0.0,
                    "max": 4.0,
                    "std": 1.0,
                }
            },
            "num_examples": 1,
            "num_batches": 1,
            "num_layer_matrices": 2,
            "out_chunk_size": 8,
            "scaling": 16.0,
        }

        class FakeModel:
            def __init__(self, stats):
                self.log_dense_lora_magnitude = True
                self._stats = stats
                self.reset_calls = 0

            def reset_dense_lora_magnitude_stats(self):
                self.reset_calls += 1

            def get_dense_lora_magnitude_stats(self):
                return self._stats

        class FakeTrainer:
            def __init__(self, output_dir, stats):
                self.args = SimpleNamespace(output_dir=output_dir)
                self.model = FakeModel(stats)
                self.logged = []
                self.saved = []

            def predict(self, *args, **kwargs):
                return SimpleNamespace(predictions=[[0, 1]], metrics={})

            def log_metrics(self, split, metrics):
                self.logged.append((split, metrics))

            def save_metrics(self, split, metrics):
                self.saved.append((split, metrics))

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FakeTrainer(tmpdir, dense_stats)
            decoded = [{"generated": "answer", "label": "answer", "ctx_ids_len": 4}]
            dataset = [{"input_ids": [1], "labels": [-100, 1]}]

            with patch(
                "ctx_to_lora.eval_utils.decode_test_result", return_value=decoded
            ):
                result = eval_generation(
                    trainer,
                    tokenizer=None,
                    ctx_tokenizer=None,
                    datasets={"dummy": dataset},
                    original_datasets={},
                    answers={},
                    split="test",
                    remove_context=False,
                    gen_kwargs={"max_new_tokens": 4, "do_sample": False},
                )

            json_path = Path(tmpdir) / "test_dummy_dense_lora_magnitude.json"
            self.assertTrue(json_path.exists())
            self.assertEqual(json.loads(json_path.read_text()), dense_stats)
            self.assertEqual(trainer.model.reset_calls, 1)
            self.assertIn("test_dummy_dense_lora_magnitude_mean", result["test_dummy"])
            self.assertEqual(
                result["test_dummy"]["test_dummy_dense_lora_magnitude_mean"], 1.25
            )
            self.assertIn(
                "test_dummy_dense_lora_magnitude_down_proj_max", result["test_dummy"]
            )

    def test_run_eval_cli_forwards_log_dense_lora_magnitude_flag(self):
        script_path = Path(__file__).resolve().parents[1] / "run_eval.py"
        captured = {}

        def fake_run_eval(**kwargs):
            captured.update(kwargs)

        argv = [
            str(script_path),
            "--model_name_or_path",
            "google/gemma-2-2b-it",
            "--datasets",
            "dummy_dataset",
            "--log_dense_lora_magnitude",
        ]

        with patch("ctx_to_lora.eval_utils.run_eval", side_effect=fake_run_eval):
            old_argv = sys.argv
            try:
                sys.argv = argv
                runpy.run_path(str(script_path), run_name="__main__")
            finally:
                sys.argv = old_argv

        self.assertTrue(captured["log_dense_lora_magnitude"])
        self.assertTrue(captured["generative"])


if __name__ == "__main__":
    unittest.main()
