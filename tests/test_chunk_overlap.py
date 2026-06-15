import inspect
import unittest

from ctx_to_lora.data.definitions import CTX_AFFIXES
from ctx_to_lora.data.processing import (
    construct_and_tokenize_ctx_qa,
    split_too_long_ctx,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


class ChunkOverlapTest(unittest.TestCase):
    def test_construct_and_tokenize_ctx_qa_accepts_overlap_kwarg(self):
        signature = inspect.signature(construct_and_tokenize_ctx_qa)
        self.assertIn("ctx_chunk_overlap", signature.parameters)

    def _strip_affixes(self, chunks):
        prefix = CTX_AFFIXES[MODEL_NAME]["prefix"]
        suffix = CTX_AFFIXES[MODEL_NAME]["suffix"]
        out = []
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                self.assertEqual(chunk[: len(prefix)], prefix)
                chunk = chunk[len(prefix) :]
            if idx < len(chunks) - 1:
                self.assertEqual(chunk[-len(suffix) :], suffix)
                chunk = chunk[: -len(suffix)]
            out.append(chunk)
        return out

    def test_eval_overlap_adds_shared_boundary_tokens(self):
        ctx_ids = list(range(10))
        out = split_too_long_ctx(
            sample={"ctx_ids": ctx_ids},
            model_name_or_path=MODEL_NAME,
            num_chunk_probs=None,
            max_chunk_len=4,
            min_chunk_len=-1,
            max_num_split=None,
            is_train=False,
            chunk_overlap=2,
        )

        stripped_chunks = self._strip_affixes(out["ctx_ids"])

        self.assertEqual(out["n_ctx_chunks"], 4)
        self.assertEqual(
            stripped_chunks,
            [
                [0, 1, 2, 3],
                [2, 3, 4, 5],
                [4, 5, 6, 7],
                [6, 7, 8, 9],
            ],
        )

    def test_zero_overlap_preserves_existing_balanced_split(self):
        ctx_ids = list(range(10))
        out = split_too_long_ctx(
            sample={"ctx_ids": ctx_ids},
            model_name_or_path=MODEL_NAME,
            num_chunk_probs=None,
            max_chunk_len=4,
            min_chunk_len=-1,
            max_num_split=None,
            is_train=False,
            chunk_overlap=0,
        )

        stripped_chunks = self._strip_affixes(out["ctx_ids"])

        self.assertEqual(out["n_ctx_chunks"], 3)
        self.assertEqual(stripped_chunks, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_overlap_is_rejected_for_train(self):
        with self.assertRaisesRegex(ValueError, "eval splits"):
            split_too_long_ctx(
                sample={"ctx_ids": list(range(8))},
                model_name_or_path=MODEL_NAME,
                num_chunk_probs=None,
                max_chunk_len=4,
                min_chunk_len=-1,
                max_num_split=None,
                is_train=True,
                chunk_overlap=1,
            )


if __name__ == "__main__":
    unittest.main()
