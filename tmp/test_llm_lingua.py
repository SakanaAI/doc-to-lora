from llmlingua import PromptCompressor

with open("src/ctx_to_lora/modeling/test_ctx.txt") as f:
    prompt = f.read()

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,  # Whether to use llmlingua-2
)
compressed_prompt = llm_lingua.compress_prompt(
    prompt, rate=0.25, force_tokens=["\n", "?"]
)
print(compressed_prompt)

## Or use LLMLingua-2-small model
# llm_lingua = PromptCompressor(
#     model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
#     use_llmlingua2=True,  # Whether to use llmlingua-2
# )
