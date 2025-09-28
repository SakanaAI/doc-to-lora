import logging
import random
from collections.abc import Callable
from typing import Any

from ctx_to_lora.data.definitions import CLOSED_QA_INTX_TEMPLATES, EVAL_INTX_TEMPLATES
from ctx_to_lora.utils import concat_list

logger = logging.getLogger()


def closed_qa_prompting(prompt: str):
    template = random.choice(CLOSED_QA_INTX_TEMPLATES)
    return template.format(input=prompt)


def chat_to_str(messages: list[dict[str, str]]):
    return "Below is the chat history from the current user.\n\n" + "\n\n".join(
        [
            "Message from: {role}\n{content}".format(
                **{**m, "role": m["role"].capitalize()}
            )
            for m in messages
        ]
    )


def get_preprocessing_fn(
    ds_name: str,
    is_eval: bool,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Get preprocessing function for a specific dataset.

    Args:
        ds_name: Name of the dataset

    Returns:
        A preprocessing function that takes and returns a dictionary
    """
    f = lambda x: x
    if ds_name.startswith("self_gen") or ds_name.endswith("_compact"):
        # already processed data, do nothing
        return f

    if "fw_qa_v2" in ds_name:

        def f(sample):
            # get questions/answers from all levels in the ds
            q_cols = [col for col in sample.keys() if col.startswith("prompts_level")]
            r_cols = [col for col in sample.keys() if col.startswith("responses_level")]
            questions = concat_list([sample[col] for col in q_cols])
            responses = concat_list([sample[col] for col in r_cols])
            min_len = min(len(questions), len(responses))

            if min_len == 0:
                return {
                    "context": None,
                    "prompts": None,
                    "responses": None,
                }

            return {
                "context": sample["context"],
                "prompts": questions[:min_len],
                "responses": responses[:min_len],
            }

    elif ds_name.startswith("ctx_qa"):

        def f(sample):
            ctx = sample["context"]
            # remove unintended prompt
            ctx = ctx.split("\n...\n\n### Code ###\n")[-1]
            sample["context"] = ctx
            return sample

    elif ds_name == "facts":

        def f(sample):
            prompt = sample["system_instruction"] + sample["user_request"]
            return {
                "context": sample["context_document"],
                "prompt": prompt,
                "response": "PLACEHOLDER. PLEASE GENERATE SELF-GEN DATA.",
            }

    elif ds_name.startswith("longbench"):

        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["input"],
                "response": sample["answers"][0],
            }

    elif ds_name == "negative_nq":

        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["prompt"],
                "response": sample["answer"],
            }

    elif ds_name == "triviaqa_retrieved":
        # only used for eval
        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["prompt"],
                "response": sample["answer"],
            }

    elif ds_name == "pwc" or ds_name == "pwc_tiny":
        # original pwc
        def f(sample):
            return {
                "context": sample["input"],
                "prompt": sample["prompt"],
                "response": sample["answer"],
            }

    elif ds_name.startswith("hotpot_qa"):

        def f(sample):
            txt = ""
            for p in sample["context"]["sentences"]:
                txt += " " + "".join(p)

            q = sample["question"]
            prompt = closed_qa_prompting(q) if not is_eval else q

            return {
                "context": txt.strip(),
                "prompt": prompt,
                "response": sample["answer"],
            }

    elif ds_name == "squad":
        # original squad
        def f(sample):
            q = sample["question"]
            prompt = closed_qa_prompting(q) if not is_eval else q
            return {
                "context": sample["context"],
                "prompt": prompt,
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "squad_assistant_ctx":

        def f(sample):
            return {
                "context": "You are a useful AI assistant.",
                "prompt": sample["context"] + "\n\n" + sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "squad_negative":
        with open("data/gutenburg_sample.txt") as f:
            gutenburg_sample = f.read()

        def f(sample):
            return {
                "context": gutenburg_sample,
                "prompt": sample["context"] + "\n\n" + sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "squad_negative_no_passage":
        with open("data/gutenburg_sample.txt") as f:
            gutenburg_sample = f.read()

        def f(sample):
            return {
                "context": gutenburg_sample,
                "prompt": sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "squad_assistant_ctx_no_passage":

        def f(sample):
            return {
                "context": "You are a useful AI assistant.",
                "prompt": sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "drop":

        def f(sample):
            q = sample["question"]
            prompt = closed_qa_prompting(q) if not is_eval else q
            return {
                "context": sample["passage"],
                "prompt": prompt,
                "response": sample["answers_spans"]["spans"][0],
            }

    elif ds_name == "ropes":
        ctx_template = "{background}\n{situation}"

        def f(sample):
            response = sample["answers"]["text"][0]
            bg_txt = sample["background"]
            situation_txt = sample["situation"]
            ctx = ctx_template.format(background=bg_txt, situation=situation_txt)
            q = sample["question"]
            q = closed_qa_prompting(q) if not is_eval else q
            return {"context": ctx, "prompt": q, "response": response}

    elif ds_name in ["narrativeqa", "quoref", "tatqa"]:  # , "ropes"]:

        def f(sample):
            response = sample["answers"][0]
            if isinstance(response, list):
                response = response[0]
            q = sample["messages"][-1]["content"]
            return {
                "context": sample["document"],
                "prompt": q,
                "response": response,
            }

    elif ds_name == "synthetic_convqa":

        def f(sample):
            response = sample["answers"][0]
            if isinstance(response, list):
                response = response[0]
            return {
                "context": sample["document"],
                "prompt": sample["messages"][-1]["content"],
                "response": response,
            }

    elif ds_name == "booksum":
        # TODO: sample from SUMM_PROMPTS once improved data gen branch is merged
        prompt_templates = [
            "Summarization the provided text.",
            "# Summary",
            "### Summary",
            "Summary of the text",
        ]
        # TODO: use these templates for training
        # analysis_templates = []
        # summary_templates = []

        def f(sample):
            return {
                "context": sample["chapter"],
                "prompt": "Summarize the provided text.",
                "response": sample["summary_text"],
            }

    elif ds_name == "gov_report":

        def f(sample):
            return {
                "context": sample["report"],
                "prompt": "Summarize the provided text.",
                "response": sample["summary"],
            }

    elif "wikitext" in ds_name:

        def f(sample):
            return {
                "context": sample["page"],
                "prompt": "PLAECHOLDER",
                "response": "PLAECHOLDER",
            }

    elif ds_name == "openmathintx-2":

        def f(sample):
            return {
                "context": sample["problem"],
                "prompt": sample["problem"],
                "response": sample["generated_solution"],
            }

    elif ds_name == "gsm8k":
        instruction_prompt = (
            "Provide the reasoning steps before giving the final integer answer."
        )

        def f(sample):
            question = sample["question"]
            last_dot_index = question.rfind(".")

            context = question[: last_dot_index + 1]
            prompt = question[last_dot_index + 1 :].lstrip()

            return {
                "prompt": prompt + "\n" + instruction_prompt,
                "context": context,
                "response": sample["answer"],
            }
            # return {
            #     "context": sample["question"],
            #     "prompt": sample["question"] + "\n" + instruction_prompt,
            #     "response": sample["answer"],
            # }
    elif ds_name == "gsm8k_negative":
        with open("data/gutenburg_sample.txt") as f:
            gutenburg_sample = f.read()

        def f(sample):
            return {
                "prompt": sample["question"],
                "context": gutenburg_sample,
                "response": sample["answer"],
            }

    elif ds_name == "gsm8k_assistant_ctx":

        def f(sample):
            return {
                "prompt": sample["question"],
                "context": "You are a useful AI assistant.",
                "response": sample["answer"],
            }
            # return {
            #     "context": sample["question"],
            #     "prompt": sample["question"] + "\n" + instruction_prompt,
            #     "response": sample["answer"],
            # }

    elif ds_name == "gsm8k_fewshot":
        N_FEW_SHOT_EXAMPLES = 5
        few_shot_examples = [
            {
                "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "target": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
            },
            {
                "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                "target": "Weng earns 12/60 = 0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.\n#### 10",
            },
            {
                "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
                "target": "In the beginning, Betty has only 100 / 2 = 50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = 5 more.\n#### 5",
            },
            {
                "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
                "target": "Maila read 12 x 2 = 24 pages today.\nSo she was able to read a total of 12 + 24 = 36 pages since yesterday.\nThere are 120 - 36 = 84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.\n#### 42",
            },
            {
                "question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?",
                "target": "He writes each friend 3*2=6 pages a week\nSo he writes 6*2=12 pages every week\nThat means he writes 12*52=624 pages a year\n#### 624",
            },
        ]
        few_shot_examples_txt = ""
        for i, example in enumerate(few_shot_examples[:N_FEW_SHOT_EXAMPLES]):
            few_shot_examples_txt += f"# Example {i + 1}:\nQuestion: {example['question']}\nSolution: {example['target']}\n\n"

        def f(sample):
            instruction_prompt = (
                "Provide the reasoning steps before giving the final integer answer."
            )

            question = sample["question"]
            # Find the last sentence as prompt (removing leading space)

            return {
                "prompt": question + "\n" + instruction_prompt,
                "context": few_shot_examples_txt,
                "response": sample["answer"],
            }
    elif ds_name == "metaicl":

        def f(sample):
            prompt = f"You are a classifier. Reply must be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {sample['options']['options']}.\nInput:\n{sample['input']}"
            context = "Examples\n"
            for example in sample["examples"]:
                context += f"Input: {example['input']}\nAnswer: {example['output']}\n\n"

            return {
                "context": context,
                "prompt": prompt,
                "response": sample["output"],
            }

    elif "opencoder-edu" in ds_name:

        def f(sample):
            return {
                "context": sample["instruction"],
                "prompt": sample["instruction"],
                "response": "```python\n" + sample["code"].strip() + "\n```",
            }

    elif "openhermes" == ds_name:
        # system prompt is from a set of predefined prompts → query
        # user prompt is the content → ctx
        def f(sample):
            messages = sample["messages"]
            if messages[0]["role"] != "system":
                return {"context": None, "prompt": None, "response": None}
            ctx = messages[1]["content"]  # user input
            q = messages[0]["content"]  # system intx
            response = messages[2]["content"]
            return {"context": ctx, "prompt": q, "response": response}

    elif "smol-magpie-ultra" == ds_name or "systemchats" == ds_name:
        # ctx → all history up to the last turn
        # q → last user input
        def f(sample):
            chat_history = chat_to_str(sample["messages"][:-2])
            q = sample["messages"][-2]["content"]
            response = sample["messages"][-1]["content"]
            return {"context": chat_history, "prompt": q, "response": response}

    elif "smol-rewrite" == ds_name or "smol-summarize" == ds_name:
        # user content is a paragraph of text → ctx
        # system prompt is a paraphrase/summarize instruction → q
        def f(sample):
            ctx = sample["messages"][1]["content"]
            q = sample["messages"][0]["content"]
            response = sample["messages"][2]["content"]
            return {"context": ctx, "prompt": q, "response": response}

    elif "ctx-q-gsm" == ds_name:

        def f(sample):
            ctx = sample["stateful_gsm_symbolic_context"]
            q = sample["stateful_gsm_symbolic_question"]
            response = sample["answer"]
            return {"context": ctx, "prompt": q, "response": response}

    elif ds_name.startswith("automathtext"):

        def f(sample):
            return {
                "context": sample["text"],
                "prompt": "PLACEHOLDER. PLEASE GENERATE QUESTIONS FOR THIS DATASET.",
                "response": "PLACEHOLDER PLEASE GENERATE QUESTIONS FOR THIS DATASET.",
            }

    elif "codeparrot" == ds_name:

        def f(sample):
            return {
                "context": sample["content"],
                "prompt": "PLACEHOLDER. PLEASE GENERATE QUESTIONS FOR THIS DATASET.",
                "response": "PLACEHOLDER. PLEASE GENERATE QUESTIONS FOR THIS DATASET.",
            }

    elif "programming_books" == ds_name:

        def f(sample):
            return {
                "context": sample["markdown"],
                "prompt": "PLACEHOLDER. PLEASE GENERATE QUESTIONS FOR THIS DATASET.",
                "response": "PLACEHOLDER. PLEASE GENERATE QUESTIONS FOR THIS DATASET.",
            }

    if is_eval and (ds_name in EVAL_INTX_TEMPLATES):
        prompt_template = EVAL_INTX_TEMPLATES[ds_name]

        def eval_intx_decorator(f):
            def g(sample):
                sample = f(sample)
                assert "prompt" in sample, (
                    f"Expected 'prompt' in sample, got {sample.keys()}"
                )
                sample["prompt"] = prompt_template.format(input=sample["prompt"])
                return sample

            return g

        f = eval_intx_decorator(f)

    def maybe_convert_to_list(f):
        def g(sample):
            sample = f(sample)
            if "prompt" in sample:
                sample["prompts"] = [sample.pop("prompt")]
            if "response" in sample:
                sample["responses"] = [sample.pop("response")]
            return sample

        return g

    f = maybe_convert_to_list(f)

    if "self_gen" not in ds_name:

        def strip_response(f):
            def g(sample):
                sample = f(sample)
                if "responses" in sample and bool(sample["responses"]):
                    sample["responses"] = [
                        r.strip() if isinstance(r, str) else r
                        for r in sample["responses"]
                    ]
                return sample

            return g

        f = strip_response(f)

    return f
