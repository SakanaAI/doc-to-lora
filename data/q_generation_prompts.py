import random

SYSTEM_PROMPT = (
    "You are a creative and helpful assistant. "
    "You will be given a context and you need to generate questions from the given context. "
    "**DO NOT** hallucinate or make up information."
)

PROMPT_TEMPLATE = (
    "### Context ###\n{context}\n\n"
    "### Instruction ###\n{instruction}\n\n"
    "### Rules ###\n"
    "Phrases like 'based on the provided context', 'according to the context', etc., must not to appear in your response.\n\n"
    # "2. The questions should not overlap. They should be diverse, covering many aspects of the context.\n"
    # "3. Do not give away too much information in the questions. For example, ask 'Who is X?' instead of 'Who is X that did Y?' when Y is clear from the context.\n"
    # "4. Ignore the text formatting of the context, e.g., bold, italic, underline, etc.\n"
    # "5. Ignore typos, spacing, and grammatical errors in the context.\n"
    # "6. Always use proper grammar and punctuation.\n"
    # "7. Try to use different question forms and styles.\n"
    "### Output Format ###\n"
    "The question/instruction/task/request should be in the following format:\n\n"
    "Message: {{question}}\n\n"
    "Use this output template regardless of the type of the output."
)

# taken from https://github.com/HazyResearch/cartridges/blob/2ac563d79c2f3367a9e780a7bb0b3cf3039a8d50/cartridges/data/resources.py#L195
# def get_structuring_seed_prompt() -> str:
#     DATA_FORMATS = [
#         "JSON",
#         "YAML",
#         "TOML",
#         "INI",
#         "XML",
#         "plain text",
#     ]

#     data_format = random.choice(DATA_FORMATS)

#     EXAMPLES = [
#         dedent(f"""
#             Can you structure the information in the context in the following format: {data_format}? Be sure to include precise information like any dates, times, names, and numerical values.
#         """).strip(),
#     ]

#     example = random.choice(EXAMPLES)

#     return dedent(f"""
#         Please generate a single chat message instructing an LLM to structure the information in {data_format}. The message can follow the following template, filling in details from the context:

#         '{example}'
#     """).strip()


# def get_summarization_seed_prompt() -> str:
#     prompts = [
#         dedent("""
#             Please generate a single chat message instructing an LLM to summarize part of the context.
#             Make sure the instruction is very explicit about the section of the context that you want to summarize.
#             Include details (ids, names, titles, dates, etc.) that make it clear what you are asking about.
#         """).strip(),
#         dedent("""
#             Please generate a single chat message instructing an LLM to summarize a section.
#             Make sure the instruction is explicit about the section that should be summarized and the document it is from.
#         """).strip(),
#     ]
#     prompt = random.choice(prompts)
#     return prompt


def get_question_seed_prompt() -> str:
    prompts = [
        (
            "Generate a question for an LLM that will test its knowledge of the information in the context above. "
            "In your question be sure to include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "Generate a message for an LLM that will test its knowledge of the information in the context above. "
            "Be sure to include details (ids, names, titles, dates, etc.) in the question so that it can be answered without access to the context (i.e. closed-book setting). "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "You are helping to quiz a user about the information in the context. "
            "Please generate a question about the subsection of the context above. "
            "Be sure to include details (ids, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
            "Answer only with the question, do not include any other text."
        ),
    ]
    prompt = random.choice(prompts)
    return prompt


def get_use_case_seed_prompt() -> str:
    prompt = (
        "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within the provided context. "
        "Consider how a user might want to apply this information in a real-world scenario, not just recall it. Put yourself into someone else's shoes and think how you might use the information. "
        "After considering potential use cases, your task will be to generate an instruction or task that reflects one of these downstream applications. "
        "This instruction or task should be something a user, who has access to this context, might ask when trying to accomplish their specific goal. "
        "Output only a single instruction or task. Do NOT include any other text or explanation."
    )
    return prompt


def get_creative_seed_prompt() -> str:
    prompt = (
        "You are having a creative open-ended conversation inspired by the information in the context. "
        "Please generate an open question for your conversation partner to start off the discussion. "
        "Answer only with the question, do not include any other text."
    )
    return prompt


def get_generic_seed_prompt() -> str:
    return "Please generate a single chat message to begin a conversation about the information in the context. Make an open-ended request or provide a task."
