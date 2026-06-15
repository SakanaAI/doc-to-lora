import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ctx_to_lora.data.definitions import CTX_AFFIXES
from ctx_to_lora.eval_utils import eval_generation, run_eval
from ctx_to_lora.modeling.rag import DocToLoraRAGModel, RAGModel, RetrievedChunk

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
PREFIX = CTX_AFFIXES[MODEL_NAME]["prefix"]
SUFFIX = CTX_AFFIXES[MODEL_NAME]["suffix"]


class FakeTokenizer:
    def __init__(self, name_or_path=MODEL_NAME):
        self.name_or_path = name_or_path
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.chat_template = "fake"
        self._token_to_id = {}
        self._id_to_token = {}
        self._next_id = 1000

    def _encode_words(self, text: str) -> list[int]:
        tokens = []
        for word in text.split():
            if word not in self._token_to_id:
                token_id = self._next_id
                self._next_id += 1
                self._token_to_id[word] = token_id
                self._id_to_token[token_id] = word
            tokens.append(self._token_to_id[word])
        return tokens

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
        **kwargs,
    ):
        del add_special_tokens, kwargs
        input_ids = self._encode_words(text)
        out = {"input_ids": input_ids}
        if return_attention_mask:
            out["attention_mask"] = [1] * len(input_ids)
        if return_tensors == "pt":
            out = {k: torch.tensor([v], dtype=torch.long) for k, v in out.items()}
        return out

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        words = []
        for token_id in ids:
            if token_id in self._id_to_token:
                words.append(self._id_to_token[token_id])
        return " ".join(words)

    def apply_chat_template(
        self,
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_attention_mask=True,
        return_dict=True,
        return_tensors=None,
        add_special_tokens=False,
        **kwargs,
    ):
        del tokenize, add_special_tokens, kwargs
        messages = (
            conversations[0]
            if conversations and isinstance(conversations[0], list)
            else conversations
        )
        user_content = [m["content"] for m in messages if m["role"] == "user"][-1]
        token_ids = PREFIX + self._encode_words(user_content)
        if add_generation_prompt:
            token_ids += SUFFIX
        out = {"input_ids": [token_ids]}
        if return_attention_mask:
            out["attention_mask"] = [[1] * len(token_ids)]
        if return_tensors == "pt":
            out = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        if return_dict:
            return out
        return out["input_ids"]


class FakeBaseModel(torch.nn.Module):
    def __init__(self, name_or_path=MODEL_NAME):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.name_or_path = name_or_path
        self.config = SimpleNamespace(
            name_or_path=name_or_path,
            _name_or_path=name_or_path,
            max_position_embeddings=4096,
        )
        self.generation_config = SimpleNamespace(pad_token_id=0)
        self.last_generate_kwargs = None

    def generate(self, *args, **kwargs):
        del args
        self.last_generate_kwargs = kwargs
        return kwargs["input_ids"]


class FakeDocToLoraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = FakeBaseModel()
        self.generation_config = self.base_model.generation_config
        self.last_generate_kwargs = None

    def generate(self, *args, **kwargs):
        del args
        self.last_generate_kwargs = dict(kwargs)
        return kwargs["input_ids"]


def build_chat_tensor(tokenizer: FakeTokenizer, text: str) -> dict[str, torch.Tensor]:
    token_ids = PREFIX + tokenizer._encode_words(text) + SUFFIX
    return {
        "input_ids": torch.tensor([token_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
    }


class RAGModelTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.base_model = FakeBaseModel()
        self.rag = RAGModel(
            model=self.base_model,
            tokenizer=self.tokenizer,
            chunk_size=4,
            chunk_overlap=2,
            top_k=2,
            max_retrieved_tokens=8,
        )

    def test_chunk_context_respects_overlap(self):
        chunks = self.rag._chunk_context("a b c d e f g h i j")
        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in chunks],
            [(0, 4), (2, 6), (4, 8), (6, 10)],
        )
        self.assertEqual(chunks[1].text, "c d e f")
        self.assertEqual(chunks[-1].text, "g h i j")

    def test_bm25_prefers_relevant_chunk(self):
        chunks = [
            RetrievedChunk(
                "apple orange", self.tokenizer._encode_words("apple orange"), 0, 2, 0.0
            ),
            RetrievedChunk(
                "banana pear", self.tokenizer._encode_words("banana pear"), 2, 4, 0.0
            ),
        ]
        ranked = self.rag._score_chunks_bm25("apple", chunks)
        self.assertEqual(ranked[0].text, "apple orange")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_generate_rebuilds_prompt_from_selected_chunks(self):
        rag = RAGModel(
            model=self.base_model,
            tokenizer=self.tokenizer,
            chunk_size=3,
            chunk_overlap=0,
            top_k=1,
            max_retrieved_tokens=3,
        )
        question_inputs = build_chat_tensor(self.tokenizer, "answer")
        context_inputs = build_chat_tensor(
            self.tokenizer,
            "cat dog fish bird apple answer kiwi mango noise noise",
        )

        rag.generate(
            input_ids=question_inputs["input_ids"],
            attention_mask=question_inputs["attention_mask"],
            ctx_ids=context_inputs["input_ids"],
            ctx_attn_mask=context_inputs["attention_mask"],
            max_new_tokens=16,
        )

        generated_input_ids = self.base_model.last_generate_kwargs["input_ids"][0]
        generated_attention_mask = self.base_model.last_generate_kwargs[
            "attention_mask"
        ][0]
        rebuilt_prompt, _ = rag._decode_question(
            generated_input_ids,
            generated_attention_mask,
        )

        self.assertIn("bird apple answer", rebuilt_prompt)
        self.assertIn("Question:", rebuilt_prompt)
        self.assertNotIn("cat dog fish", rebuilt_prompt)
        self.assertEqual(len(rag.get_retrieval_history()), 1)
        self.assertEqual(
            rag.get_retrieval_history()[0]["rag_selected_chunks"][0]["text"],
            "bird apple answer",
        )

    def test_hybrid_generate_preserves_ctx_and_rewrites_prompt_input(self):
        doc_model = FakeDocToLoraModel()
        hybrid = DocToLoraRAGModel(
            model=doc_model,
            tokenizer=self.tokenizer,
            chunk_size=3,
            chunk_overlap=0,
            top_k=1,
            max_retrieved_tokens=3,
        )
        question_inputs = build_chat_tensor(self.tokenizer, "answer")
        context_inputs = build_chat_tensor(
            self.tokenizer,
            "cat dog fish bird apple answer kiwi mango noise noise",
        )

        hybrid.generate(
            input_ids=question_inputs["input_ids"],
            attention_mask=question_inputs["attention_mask"],
            ctx_ids=context_inputs["input_ids"],
            ctx_attn_mask=context_inputs["attention_mask"],
            n_ctx_chunks=torch.tensor([1], dtype=torch.int32),
            max_new_tokens=8,
        )

        forwarded = doc_model.last_generate_kwargs
        self.assertTrue(torch.equal(forwarded["ctx_ids"], context_inputs["input_ids"]))
        self.assertTrue(
            torch.equal(forwarded["ctx_attn_mask"], context_inputs["attention_mask"])
        )
        self.assertTrue(
            torch.equal(forwarded["n_ctx_chunks"], torch.tensor([1], dtype=torch.int32))
        )
        rebuilt_prompt, _ = hybrid._decode_question(
            forwarded["input_ids"][0],
            forwarded["attention_mask"][0],
        )
        self.assertIn("bird apple answer", rebuilt_prompt)
        self.assertEqual(
            hybrid.get_retrieval_history()[0]["rag_selected_chunks"][0]["text"],
            "bird apple answer",
        )

    def test_run_eval_rejects_eval_chunking_with_rag(self):
        with self.assertRaisesRegex(ValueError, "retrieval chunking"):
            run_eval(
                model_name_or_path="google/gemma-2-2b-it",
                datasets=["dummy_dataset"],
                use_rag=True,
                max_ctx_chunk_len=128,
                eval_batch_size=1,
            )

    def test_run_eval_rejects_hybrid_without_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "only available"):
            run_eval(
                model_name_or_path="google/gemma-2-2b-it",
                datasets=["dummy_dataset"],
                use_hybrid_rag=True,
                eval_batch_size=1,
            )

    def test_eval_generation_saves_retrieval_metadata(self):
        class FakeModel:
            def __init__(self):
                self._history = [
                    {
                        "rag_question": "who?",
                        "rag_selected_chunks": [
                            {"text": "retrieved text", "score": 1.0}
                        ],
                    }
                ]

            def reset_retrieval_history(self):
                self._history = list(self._history)

            def get_retrieval_history(self):
                return self._history

        class FakeTrainer:
            def __init__(self, output_dir):
                self.args = SimpleNamespace(output_dir=output_dir)
                self.model = FakeModel()

            def predict(self, *args, **kwargs):
                del args, kwargs
                return SimpleNamespace(predictions=[[0, 1]], metrics={})

            def log_metrics(self, split, metrics):
                del split, metrics

            def save_metrics(self, split, metrics):
                del split, metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FakeTrainer(tmpdir)
            decoded = [{"generated": "answer", "label": "answer", "ctx_ids_len": 4}]
            dataset = [{"input_ids": [1], "labels": [-100, 1]}]

            with patch(
                "ctx_to_lora.eval_utils.decode_test_result", return_value=decoded
            ):
                eval_generation(
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

            jsonl_path = Path(tmpdir) / "test_dummy_generated_text.jsonl"
            lines = jsonl_path.read_text().splitlines()
            payload = json.loads(lines[0])
            self.assertEqual(payload["rag_question"], "who?")
            self.assertEqual(
                payload["rag_selected_chunks"][0]["text"], "retrieved text"
            )

    def test_run_eval_cli_forwards_rag_flags(self):
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
            "--use_rag",
            "--rag_chunk_size",
            "128",
            "--rag_chunk_overlap",
            "16",
            "--rag_top_k",
            "3",
            "--rag_max_retrieved_tokens",
            "512",
        ]

        with patch("ctx_to_lora.eval_utils.run_eval", side_effect=fake_run_eval):
            old_argv = sys.argv
            try:
                sys.argv = argv
                runpy.run_path(str(script_path), run_name="__main__")
            finally:
                sys.argv = old_argv

        self.assertTrue(captured["use_rag"])
        self.assertEqual(captured["rag_chunk_size"], 128)
        self.assertEqual(captured["rag_chunk_overlap"], 16)
        self.assertEqual(captured["rag_top_k"], 3)
        self.assertEqual(captured["rag_max_retrieved_tokens"], 512)
        self.assertTrue(captured["generative"])

    def test_run_eval_cli_forwards_hybrid_rag_flag(self):
        script_path = Path(__file__).resolve().parents[1] / "run_eval.py"
        captured = {}

        def fake_run_eval(**kwargs):
            captured.update(kwargs)

        argv = [
            str(script_path),
            "--checkpoint_path",
            "train_outputs/runs/demo/checkpoint-1/pytorch_model.bin",
            "--datasets",
            "dummy_dataset",
            "--use_hybrid_rag",
            "--rag_chunk_size",
            "128",
        ]

        with patch("ctx_to_lora.eval_utils.run_eval", side_effect=fake_run_eval):
            old_argv = sys.argv
            try:
                sys.argv = argv
                runpy.run_path(str(script_path), run_name="__main__")
            finally:
                sys.argv = old_argv

        self.assertTrue(captured["use_hybrid_rag"])
        self.assertEqual(captured["rag_chunk_size"], 128)
        self.assertTrue(captured["generative"])


if __name__ == "__main__":
    unittest.main()
