import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ctx_to_lora.data.definitions import CTX_AFFIXES

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class RetrievedChunk:
    text: str
    token_ids: list[int]
    start: int
    end: int
    score: float


class _RAGMixin:
    def _init_rag(
        self,
        tokenizer,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        top_k: int = 4,
        max_retrieved_tokens: int = 1536,
        retrieval_mode: str = "bm25",
        prompt_instruction: str | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("`chunk_size` must be > 0.")
        if chunk_overlap < 0:
            raise ValueError("`chunk_overlap` must be >= 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("`chunk_overlap` must be smaller than `chunk_size`.")
        if top_k <= 0:
            raise ValueError("`top_k` must be > 0.")
        if max_retrieved_tokens <= 0:
            raise ValueError("`max_retrieved_tokens` must be > 0.")
        if retrieval_mode != "bm25":
            raise ValueError(
                f"Unsupported `retrieval_mode`: {retrieval_mode}. Only 'bm25' is supported."
            )

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.max_retrieved_tokens = max_retrieved_tokens
        self.retrieval_mode = retrieval_mode
        self.prompt_instruction = (
            prompt_instruction
            or "Answer the question using only the retrieved context below. "
            "If the answer is not supported by the retrieved context, say you don't know."
        )

        model_name = getattr(self.base_model, "name_or_path", None)
        if model_name is None and hasattr(self.base_model, "config"):
            model_name = getattr(
                self.base_model.config,
                "name_or_path",
                getattr(self.base_model.config, "_name_or_path", None),
            )
        if model_name not in CTX_AFFIXES:
            raise ValueError(
                f"No context affixes configured for model '{model_name}'. "
                "RAG wrappers only support models listed in `CTX_AFFIXES`."
            )

        affixes = CTX_AFFIXES[model_name]
        self.prefix = affixes["prefix"]
        self.suffix = affixes["suffix"]
        self.len_prefix = len(self.prefix)
        self.len_suffix = len(self.suffix)
        self.last_retrieval = None
        self._retrieval_history = []

    @property
    def generation_config(self):
        return self.generation_model.generation_config

    @property
    def device(self):
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def reset_retrieval_history(self) -> None:
        self.last_retrieval = None
        self._retrieval_history = []

    def get_retrieval_history(self) -> list[dict[str, Any]]:
        return list(self._retrieval_history)

    def _strip_padding_and_affixes(
        self, token_ids: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> list[int]:
        if attn_mask is not None:
            valid = token_ids[attn_mask.bool()].tolist()
        else:
            valid = token_ids.tolist()

        if len(valid) >= self.len_prefix and valid[: self.len_prefix] == self.prefix:
            valid = valid[self.len_prefix :]
        if len(valid) >= self.len_suffix and valid[-self.len_suffix :] == self.suffix:
            valid = valid[: -self.len_suffix]
        return valid

    def _decode_question(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[str, list[int]]:
        question_ids = self._strip_padding_and_affixes(input_ids, attention_mask)
        question = self.tokenizer.decode(question_ids, skip_special_tokens=True).strip()
        return question, question_ids

    def _decode_context(
        self,
        ctx_ids: torch.Tensor,
        ctx_attn_mask: torch.Tensor | None = None,
    ) -> tuple[str, list[int]]:
        if ctx_ids.ndim == 1:
            ctx_ids = ctx_ids.unsqueeze(0)
        if ctx_attn_mask is None:
            ctx_attn_mask = torch.ones_like(ctx_ids)
        elif ctx_attn_mask.ndim == 1:
            ctx_attn_mask = ctx_attn_mask.unsqueeze(0)

        full_ids = []
        for chunk_ids, chunk_mask in zip(ctx_ids, ctx_attn_mask):
            full_ids.extend(self._strip_padding_and_affixes(chunk_ids, chunk_mask))
        context = self.tokenizer.decode(full_ids, skip_special_tokens=True).strip()
        return context, full_ids

    def _tokenize_text(self, text: str) -> list[int]:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

    def _chunk_context(self, context_text: str) -> list[RetrievedChunk]:
        context_ids = self._tokenize_text(context_text)
        if not context_ids:
            return []
        if len(context_ids) <= self.chunk_size:
            return [
                RetrievedChunk(
                    text=context_text.strip(),
                    token_ids=context_ids,
                    start=0,
                    end=len(context_ids),
                    score=0.0,
                )
            ]

        stride = self.chunk_size - self.chunk_overlap
        chunks = []
        for start in range(0, len(context_ids), stride):
            end = min(start + self.chunk_size, len(context_ids))
            chunk_ids = context_ids[start:end]
            if not chunk_ids:
                break
            chunk_text = self.tokenizer.decode(
                chunk_ids, skip_special_tokens=True
            ).strip()
            chunks.append(
                RetrievedChunk(
                    text=chunk_text,
                    token_ids=chunk_ids,
                    start=start,
                    end=end,
                    score=0.0,
                )
            )
            if end == len(context_ids):
                break
        return chunks

    def _lexical_tokenize(self, text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    def _score_chunks_bm25(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        query_terms = self._lexical_tokenize(query)
        if not chunks:
            return []
        if not query_terms:
            return [
                RetrievedChunk(**{**asdict(chunk), "score": 0.0})
                for chunk in chunks[: self.top_k]
            ]

        tokenized_chunks = [self._lexical_tokenize(chunk.text) for chunk in chunks]
        num_chunks = len(tokenized_chunks)
        avgdl = sum(len(chunk_terms) for chunk_terms in tokenized_chunks) / max(
            num_chunks, 1
        )
        doc_freq = Counter()
        for chunk_terms in tokenized_chunks:
            for term in set(chunk_terms):
                doc_freq[term] += 1

        k1 = 1.5
        b = 0.75
        ranked = []
        for idx, (chunk, chunk_terms) in enumerate(zip(chunks, tokenized_chunks)):
            term_freq = Counter(chunk_terms)
            doc_len = len(chunk_terms)
            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if freq == 0:
                    continue
                df = doc_freq[term]
                idf = math.log(1.0 + (num_chunks - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1.0 - b + b * doc_len / max(avgdl, 1e-8))
                score += idf * (freq * (k1 + 1.0)) / denom
            score += 1e-4 / (1 + idx)
            ranked.append(RetrievedChunk(**{**asdict(chunk), "score": float(score)}))

        ranked.sort(key=lambda chunk: (-chunk.score, chunk.start, chunk.end))
        return ranked

    def _merge_overlapping_chunks(
        self, context_ids: list[int], chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        chunks = sorted(chunks, key=lambda chunk: (chunk.start, chunk.end))
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            prev = merged[-1]
            if chunk.start <= prev.end:
                start = prev.start
                end = max(prev.end, chunk.end)
                merged_ids = context_ids[start:end]
                merged[-1] = RetrievedChunk(
                    text=self.tokenizer.decode(
                        merged_ids, skip_special_tokens=True
                    ).strip(),
                    token_ids=merged_ids,
                    start=start,
                    end=end,
                    score=max(prev.score, chunk.score),
                )
                continue
            merged.append(chunk)
        return merged

    def _select_chunks_with_budget(
        self,
        ranked_chunks: list[RetrievedChunk],
        context_ids: list[int],
    ) -> list[RetrievedChunk]:
        if not ranked_chunks:
            return []

        selected = []
        budget_used = 0
        for chunk in ranked_chunks:
            if len(selected) >= self.top_k:
                break
            chunk_len = len(chunk.token_ids)
            if selected and budget_used + chunk_len > self.max_retrieved_tokens:
                continue
            if chunk_len > self.max_retrieved_tokens:
                truncated_ids = chunk.token_ids[: self.max_retrieved_tokens]
                selected.append(
                    RetrievedChunk(
                        text=self.tokenizer.decode(
                            truncated_ids, skip_special_tokens=True
                        ).strip(),
                        token_ids=truncated_ids,
                        start=chunk.start,
                        end=chunk.start + len(truncated_ids),
                        score=chunk.score,
                    )
                )
                budget_used = len(truncated_ids)
                break

            selected.append(chunk)
            budget_used += chunk_len

        if not selected:
            top_chunk = ranked_chunks[0]
            truncated_ids = top_chunk.token_ids[: self.max_retrieved_tokens]
            selected = [
                RetrievedChunk(
                    text=self.tokenizer.decode(
                        truncated_ids, skip_special_tokens=True
                    ).strip(),
                    token_ids=truncated_ids,
                    start=top_chunk.start,
                    end=top_chunk.start + len(truncated_ids),
                    score=top_chunk.score,
                )
            ]

        return self._merge_overlapping_chunks(context_ids, selected)

    def _build_prompt_text(
        self,
        question: str,
        selected_chunks: list[RetrievedChunk],
    ) -> str:
        sections = [self.prompt_instruction.strip()]
        if selected_chunks:
            sections.append("Retrieved context:")
            for idx, chunk in enumerate(selected_chunks, start=1):
                sections.append(f"[{idx}] {chunk.text.strip()}")
        sections.append(f"Question:\n{question.strip()}")
        return "\n\n".join(section for section in sections if section.strip())

    def _build_rag_inputs(
        self,
        question: str,
        selected_chunks: list[RetrievedChunk],
    ) -> dict[str, torch.Tensor]:
        prompt_text = self._build_prompt_text(question, selected_chunks)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_text},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_attention_mask=True,
            return_dict=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {k: v.to(self.device) for k, v in tokenized.items()}

    def _retrieve(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        context_ids: list[int],
    ) -> tuple[list[RetrievedChunk], bool]:
        if len(context_ids) <= self.max_retrieved_tokens:
            full_context = RetrievedChunk(
                text=self.tokenizer.decode(
                    context_ids, skip_special_tokens=True
                ).strip(),
                token_ids=context_ids,
                start=0,
                end=len(context_ids),
                score=0.0,
            )
            return [full_context], True

        ranked = self._score_chunks_bm25(question, chunks)
        return self._select_chunks_with_budget(ranked, context_ids), False

    def _record_retrieval(
        self,
        question: str,
        context_token_count: int,
        prompt_token_count: int,
        used_full_context: bool,
        selected_chunks: list[RetrievedChunk],
    ) -> None:
        retrieval = {
            "rag_question": question,
            "rag_context_token_count": context_token_count,
            "rag_prompt_token_count": prompt_token_count,
            "rag_used_full_context": used_full_context,
            "rag_chunk_size": self.chunk_size,
            "rag_chunk_overlap": self.chunk_overlap,
            "rag_top_k": self.top_k,
            "rag_max_retrieved_tokens": self.max_retrieved_tokens,
            "rag_retrieval_mode": self.retrieval_mode,
            "rag_selected_chunks": [asdict(chunk) for chunk in selected_chunks],
        }
        self.last_retrieval = retrieval
        self._retrieval_history.append(retrieval)

    def _prepare_rag_generation_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ctx_ids: torch.Tensor,
        ctx_attn_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        question, _ = self._decode_question(input_ids[0], attention_mask[0])
        context_text, context_ids = self._decode_context(ctx_ids, ctx_attn_mask)
        chunks = self._chunk_context(context_text)
        selected_chunks, used_full_context = self._retrieve(
            question, chunks, context_ids
        )
        rag_inputs = self._build_rag_inputs(question, selected_chunks)
        prompt_token_count = int(rag_inputs["attention_mask"].sum().item())
        self._record_retrieval(
            question=question,
            context_token_count=len(context_ids),
            prompt_token_count=prompt_token_count,
            used_full_context=used_full_context,
            selected_chunks=selected_chunks,
        )
        return rag_inputs


class RAGModel(nn.Module, _RAGMixin):
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        top_k: int = 4,
        max_retrieved_tokens: int = 1536,
        retrieval_mode: str = "bm25",
        prompt_instruction: str | None = None,
    ):
        super().__init__()
        self.base_model = model
        self.generation_model = model
        self._init_rag(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            max_retrieved_tokens=max_retrieved_tokens,
            retrieval_mode=retrieval_mode,
            prompt_instruction=prompt_instruction,
        )

    def generate(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError("RAGModel only supports batch_size=1 during evaluation.")

        attention_mask = kwargs.get("attention_mask")
        ctx_ids = kwargs.get("ctx_ids")
        ctx_attn_mask = kwargs.get("ctx_attn_mask")
        if ctx_ids is None:
            raise ValueError("RAGModel requires `ctx_ids`.")

        rag_inputs = self._prepare_rag_generation_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ctx_ids=ctx_ids,
            ctx_attn_mask=ctx_attn_mask,
        )

        for key in [
            "ctx_ids",
            "ctx_attn_mask",
            "n_ctx_chunks",
            "input_ids",
            "attention_mask",
        ]:
            kwargs.pop(key, None)

        return self.base_model.generate(*args, **rag_inputs, **kwargs)


class DocToLoraRAGModel(nn.Module, _RAGMixin):
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        top_k: int = 4,
        max_retrieved_tokens: int = 1536,
        retrieval_mode: str = "bm25",
        prompt_instruction: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.base_model = model.base_model
        self.generation_model = model
        self._init_rag(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            max_retrieved_tokens=max_retrieved_tokens,
            retrieval_mode=retrieval_mode,
            prompt_instruction=prompt_instruction,
        )

    def generate(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError(
                "DocToLoraRAGModel only supports batch_size=1 during evaluation."
            )

        attention_mask = kwargs.get("attention_mask")
        ctx_ids = kwargs.get("ctx_ids")
        ctx_attn_mask = kwargs.get("ctx_attn_mask")
        if ctx_ids is None:
            raise ValueError("DocToLoraRAGModel requires `ctx_ids`.")

        rag_inputs = self._prepare_rag_generation_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ctx_ids=ctx_ids,
            ctx_attn_mask=ctx_attn_mask,
        )

        kwargs = dict(kwargs)
        kwargs["input_ids"] = rag_inputs["input_ids"]
        kwargs["attention_mask"] = rag_inputs["attention_mask"]
        return self.model.generate(*args, **kwargs)
