import os

import torch
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"  # vLLM V1 does not support logits processors.


class LogitsSpy:
    def __init__(self):
        self.processed_logits: list[torch.Tensor] = []

    def __call__(self, token_ids: list[int], logits: torch.Tensor):
        self.processed_logits.append(logits)
        return logits


if __name__ == "__main__":
    llm = LLM(model="google/gemma-2-2b-it", trust_remote_code=True)
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a robot learning to love.",
    ]
    logits_spy = LogitsSpy()
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=5, logits_processors=[logits_spy]
    )
    outputs = llm.generate(prompts, sampling_params)
    logits = logits_spy.processed_logits
    print(f"Logits: {logits}")
    print(len(logits))
    print(logits[0].shape)
    print(logits[1].shape)
    print(logits[2].shape)
    print(logits[3].shape)
    print(logits[4].shape)

    breakpoint()
