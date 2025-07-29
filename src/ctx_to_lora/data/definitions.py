from glob import glob

IGNORE_INDEX = -100

FW_QA_PATHS = [
    f"data/raw_datasets/fw_qa/{i:05d}.parquet" for i in [0, 1, 6, 7, 8, 10, 22, 30, 35]
]

TRANSFORMED_DATA_DIR = "data/processed_datasets"
RAW_DATA_DIR = "data/raw_datasets/"
SELF_GEN_DATA_DIR = f"{RAW_DATA_DIR}/self_gen/"

# see https://huggingface.co/datasets/YuxinJiang/LTE_train_data/viewer/default/train?row=97&views%5B%5D=train&sql_row=2
# based on https://arxiv.org/pdf/2402.11905
SELF_GEN_SYSTEM_MSG = (
    "### SYSTEM INSTRUCTION ###\n"
    "You are an honest and helpful assistant.\n"
    "You must use the information provided in the context for responding to the question.\n"
    "**DO NOT** hallucinate or make up information.\n"
    "### END OF SYSTEM INSTRUCTION ###"
)


REPEAT_PROMPTS = [
    "Repeat the text.",
    "Repeat the text above.",
    "Repeat the above text.",
    "Repeat exactly what was written.",
    "Duplicate the text.",
    "Echo the text. Do not output other words.",
    "Repeat the content provided.",
    "Copy the content. Reply with just the content.",
    "Repeat what you see above. Avoid explanation.",
    "Replicate the given text precisely.",
    "Repeat the entire passage.",
    "Duplicate exactly the provided text.",
    "Repeat the text that appears above.",
    "Reproduce exactly what appears above.",
    "Repeat the article.",
    "Repeat verbatim the given text.",
    "Output the same text as shown.",
    "Output an exact copy of all of the content.",
    "Provide a replica of the text.",
    "Output an identical copy of the text.",
    "Repeat the content without any changes.",
    "Repeat the provided text.",
    "Repeat the provided text above.",
    "Repeat precisely what is written in the text above.",
]

# approximate length of the datasets (train split)
# needed for streaming datasets
DS_LEN = {
    "fw_qa_3_intx_pretrain": 90_000_000,
    "fw_qa_3_intx_pretrain_medium": 40_000_000,
    "fw_qa_3_intx_pretrain_mini": 100_000,
    "fw_qa_3_mini": 100_000,
    "fw_qa_3_medium": 121_000_000,
    "fw_qa_3": 270_000_000,
    "fw_qa_xl": 27_000_000,
    "ctx_qa": 300_000,
    "pwc": 240_000,
    "hotpot_qa": 100_000,
    "squad": 90_000,
    "drop": 77_000,
    "narrativeqa": 40_000,
    "quoref": 11_000,
    "ropes": 11_000,
    "synthetic_convqa": 40_000,
}

LONGBENCH_TASKS = [
    "longbench/narrativeqa",
    "longbench/qasper",
    "longbench/multifieldqa_en",
    "longbench/multifieldqa_zh",
    "longbench/hotpotqa",
    "longbench/2wikimqa",
    "longbench/musique",
    "longbench/dureader",
    "longbench/gov_report",
    "longbench/qmsum",
    "longbench/multi_news",
    "longbench/vcsum",
]

LONGBENCH_E_TASKS = [
    "longbench/qasper_e",
    "longbench/multifieldqa_en_e",
    "longbench/hotpotqa_e",
    "longbench/2wikimqa_e",
    "longbench/gov_report_e",
    "longbench/multi_news_e",
]

DS_KWARGS = {
    "context_numbers_2_10": dict(
        train=dict(
            path="json",
            data_files=glob("data/raw_datasets/context_numbers_[2-9]/train.jsonl")
            + ["data/raw_datasets/context_numbers_10/train.jsonl"],
            split="train",
        ),
        validation=dict(
            path="json",
            data_files=glob("data/raw_datasets/context_numbers_[2-9]/val.jsonl")
            + ["data/raw_datasets/context_numbers_10/val.jsonl"],
            split="train",
        ),
    ),
    "hotpot_qa": dict(
        train=dict(path="hotpotqa/hotpot_qa", name="fullwiki", split="train[900:]"),
        validation=dict(
            path="hotpotqa/hotpot_qa", name="fullwiki", split="train[:900]"
        ),
        test=dict(path="hotpotqa/hotpot_qa", name="fullwiki", split="validation"),
    ),
    "hotpot_qa_tiny": dict(
        train=dict(path="hotpotqa/hotpot_qa", name="fullwiki", split="train[900:2000]"),
        validation=dict(
            path="hotpotqa/hotpot_qa", name="fullwiki", split="train[:900]"
        ),
    ),
    "pwc": dict(
        train=dict(path="sggetao/PwC", split="train[900:]"),
        validation=dict(path="sggetao/PwC", split="train[:900]"),
        test=dict(path="sggetao/PwC", split="test"),
    ),
    "pwc_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/pwc_compact/train/ds.parquet",
            # correspond to skipping to first 900 samples in the original dataset
            split="train[60:]",
        ),
    ),
    "pwc_compact_tiny": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/pwc_compact/train/ds.parquet",
            # correspond to skipping to first 900 samples in the original dataset
            split="train[60:200]",
        ),
    ),
    "pwc_tiny": dict(
        train=dict(path="sggetao/PwC", split="train[900:2000]"),
        validation=dict(path="sggetao/PwC", split="train[:900]"),
    ),
    "squad": dict(
        train=dict(path="data/raw_datasets/squad", split="train[900:]"),
        validation=dict(path="data/raw_datasets/squad", split="train[:900]"),
        test=dict(path="data/raw_datasets/squad", split="validation"),
        # train=dict(path="rajpurkar/squad", split="train[900:]"),
        # validation=dict(path="rajpurkar/squad", split="train[:900]"),
        # test=dict(path="rajpurkar/squad", split="validation"),
    ),
    "squad_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/squad_compact/train/ds.parquet",
            # correspond to skipping to first 900 samples in the original dataset
            split="train[180:]",
        ),
    ),
    "fw_qa_xl": dict(
        train=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/fw_qa_xl/*[!val].parquet"),
            split="train",
        ),
        validation=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/fw_qa_xl/*val.parquet"),
            split="train",
        ),
        # there is no explcit test for this ds
        # used only for sanity check
        test=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/fw_qa_xl/*val.parquet"),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_0_tiny": dict(
        train=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/000*level_0.parquet"
            ),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_0_small": dict(
        train=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/00[02468]*level_0.parquet"
            ),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_0": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/fw_qa_v2/min_0_to_2000/*level_0.parquet",
            split="train",
        ),
        validation=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/*level_0_val.parquet"
            ),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_1": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/fw_qa_v2/min_0_to_2000/*level_1.parquet",
            split="train",
        ),
        # validation=dict(
        #     path="parquet",
        #     data_files=glob(
        #         "data/raw_datasets/fw_qa_v2/min_0_to_2000/*level_1_val.parquet"
        #     ),
        #     split="train",
        # ),
    ),
    "fw_qa_v2_2k_len_level_1_tiny": dict(
        train=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/000*level_1.parquet"
            ),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_1_small": dict(
        train=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/00[02468]*level_1.parquet"
            ),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_2": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/fw_qa_v2/min_0_to_2000/*level_2.parquet",
            split="train",
        ),
        # validation=dict(
        #     path="parquet",
        #     data_files=glob(
        #         "data/raw_datasets/fw_qa_v2/min_0_to_2000/*level_2_val.parquet"
        #     ),
        #     split="train",
        # ),
    ),
    "fw_qa_v2_2k_len_level_2_tiny": dict(
        train=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/000*level_2.parquet"
            ),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_2_small": dict(
        train=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/00[02468]*level_2.parquet"
            ),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_3": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/fw_qa_v2/min_0_to_2000/*level_3.parquet",
            split="train",
        ),
        # validation=dict(
        #     path="parquet",
        #     data_files=glob(
        #         "data/raw_datasets/fw_qa_v2/min_0_to_2000/*level_3_val.parquet"
        #     ),
        #     split="train",
        # ),
    ),
    "fw_qa_v2_2k_len_level_3_tiny": dict(
        train=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/000*level_3.parquet"
            ),
            split="train",
        ),
    ),
    "fw_qa_v2_2k_len_level_3_small": dict(
        train=dict(
            path="parquet",
            data_files=glob(
                "data/raw_datasets/fw_qa_v2/min_0_to_2000/00[02468]*level_3.parquet"
            ),
            split="train",
        ),
    ),
    # "fw_qa_3_small": dict(
    #     train=dict(
    #         path="parquet",
    #         data_files=glob("data/raw_datasets/fw_qa_3/000*[!val].parquet"),
    #         split="train",
    #     ),
    # ),
    # "fw_qa_3_medium": dict(
    #     train=dict(
    #         path="parquet",
    #         data_files=glob("data/raw_datasets/fw_qa_3/00[0-5]*[!val].parquet"),
    #         split="train",
    #     ),
    # ),
    # "fw_qa_3": dict(
    #     train=dict(
    #         path="parquet",
    #         data_files=glob("data/raw_datasets/fw_qa_3/*[!val].parquet"),
    #         split="train",
    #     ),
    #     validation=dict(
    #         path="parquet",
    #         data_files=glob("data/raw_datasets/fw_qa_3/*val.parquet"),
    #         split="train",
    #     ),
    # ),
    # "fw_qa_3_mini_pretrain": dict(
    #     train=dict(
    #         path="parquet",
    #         data_files=glob("data/raw_datasets/fw_qa_intx_pretrain/000_00004.parquet"),
    #         split="train[:100000]",
    #     )
    # ),
    # "fw_qa_3_small_pretrain": dict(
    #     train=dict(
    #         path="parquet",
    #         data_files=glob(
    #             "data/raw_datasets/fw_qa_intx_pretrain/000_*[!val].parquet"
    #         ),
    #         split="train",
    #     )
    # ),
    # "fw_qa_3_small_aug_pretrain": dict(
    #     train=dict(
    #         path="parquet",
    #         data_files=glob("data/raw_datasets/fw_qa_aug_pretrain/000_*[!val].parquet"),
    #         split="train",
    #     )
    # ),
    # "fw_qa_3_medium_pretrain": dict(
    #     train=dict(
    #         path="parquet",
    #         data_files=glob(
    #             "data/raw_datasets/fw_qa_intx_pretrain/00[0-5]*[!val].parquet"
    #         ),
    #         split="train",
    #     )
    # ),
    # "fw_qa_3_pretrain": dict(
    #     train=dict(
    #         path="parquet",
    #         data_files=glob("data/raw_datasets/fw_qa_intx_pretrain/*[!val].parquet"),
    #         split="train",
    #     ),
    #     validation=dict(
    #         path="parquet",
    #         data_files=glob("data/raw_datasets/fw_qa_intx_pretrain/*val.parquet"),
    #         split="train",
    #     ),
    # ),
    "ctx_qa": dict(
        train=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/ctx_qa/*res.parquet"),
            split="train",
        ),
        validation=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/ctx_qa/*val.parquet"),
            split="train",
        ),
        # there is no explcit test for this ds
        # used only for sanity check
        test=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/ctx_qa/*val.parquet"),
            split="train",
        ),
    ),
    "drop": dict(
        train=dict(
            path="ucinlp/drop",
            split="train",
        ),
        validation=dict(
            path="ucinlp/drop",
            split="validation[:900]",
        ),
        test=dict(
            path="ucinlp/drop",
            split="validation",
        ),
    ),
    "drop_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/drop_compact/train/ds.parquet",
            # correspond to skipping to first 900 samples in the original dataset
            split="train",
        )
    ),
    "narrativeqa": dict(
        train=dict(
            path="nvidia/ChatQA-Training-Data",
            name="narrativeqa",
            split="train",
        ),
    ),
    "quoref": dict(
        train=dict(
            path="nvidia/ChatQA-Training-Data",
            name="quoref",
            split="train",
        ),
    ),
    # "ropes": dict(
    #     train=dict(
    #         path="nvidia/ChatQA-Training-Data",
    #         name="ropes",
    #         split="train",
    #     ),
    # ),
    "ropes": dict(
        train=dict(
            path="allenai/ropes",
            split="train",
        ),
        validation=dict(
            path="allenai/ropes",
            split="validation[:900]",
        ),
        test=dict(path="allenai/ropes", split="validation"),
    ),
    "ropes_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/ropes_compact/train/ds.parquet",
            split="train",
        )
    ),
    "synthetic_convqa": dict(
        train=dict(
            path="nvidia/ChatQA-Training-Data",
            name="synthetic_convqa",
            split="train",
        ),
    ),
    "booksum": dict(
        train=dict(
            path="kmfoda/booksum",
            split="train",
        ),
        validation=dict(
            path="kmfoda/booksum",
            split="validation",
        ),
        test=dict(
            path="kmfoda/booksum",
            split="test",
        ),
    ),
    "gov_report": dict(
        train=dict(
            path="ccdv/govreport-summarization",
            split="train",
        ),
        validation=dict(
            path="ccdv/govreport-summarization",
            split="validation",
        ),
        test=dict(
            path="ccdv/govreport-summarization",
            split="test",
        ),
    ),
    # "wikitext-2": dict(
    #     train=dict(
    #         path="EleutherAI/wikitext_document_level",
    #         name="wikitext-2-raw-v1",
    #         split="train",
    #     ),
    # ),
    # "wikitext-103": dict(
    #     train=dict(
    #         path="EleutherAI/wikitext_document_level",
    #         name="wikitext-103-raw-v1",
    #         split="train",
    #     ),
    # ),
    "gsm8k": dict(
        train=dict(
            path="openai/gsm8k",
            name="main",
            split="train[100:]",
        ),
        validation=dict(
            path="openai/gsm8k",
            name="main",
            split="train[:100]",
        ),
    ),
    "openmathintx-2": dict(
        train=dict(
            path="nvidia/OpenMathInstruct-2",
            split="train_1M[:100000]",
        ),
    ),
    "opencoder-edu": dict(
        train=dict(
            path="OpenCoder-LLM/opc-sft-stage2",
            name="educational_instruct",
            split="train[900:]",
        ),
        validation=dict(
            path="OpenCoder-LLM/opc-sft-stage2",
            name="educational_instruct",
            split="train[:900]",
        ),
    ),
    "triviaqa_retrieved": dict(
        test=dict(
            path="json",
            data_files="data/eval/triviaqa/triviaqa_retrieved.jsonl",
            split="train",
        ),
    ),
    # for negative training and evaluating
    "negative_nq": dict(
        test=dict(
            path="json",
            data_files="data/raw_datasets/negative_natural_questions/validation.jsonl",
            split="train",
        ),
    ),
}

# LongBench kwargs
for ds_name in LONGBENCH_TASKS + LONGBENCH_E_TASKS:
    DS_KWARGS[ds_name] = dict(
        test=dict(
            path="THUDM/LongBench",
            name=ds_name.split("/")[-1],
            split="test",
        )
    )


CLOSED_QA_DATASETS = {
    "longbench/narrativeqa",
    "longbench/qasper",
    "longbench/multifieldqa_en",
    "longbench/hotpotqa",
    "longbench/2wikimqa",
    "longbench/musique",
    "hotpot_qa",
    "squad",
    "triviaqa_retrieved",
    "negative_nq",
    "ropes",
    "drop",
}

MULTI_ANSWER_DATASETS = {
    "longbench/narrativeqa",
    "longbench/qasper",
    "longbench/multifieldqa_en",
    "longbench/hotpotqa",
    "longbench/2wikimqa",
    "longbench/musique",
    "squad",
    "drop",
}


for ds_name in list(CLOSED_QA_DATASETS):
    if ds_name.startswith("longbench/"):
        CLOSED_QA_DATASETS.add(f"{ds_name}_e")


for ds_name in list(MULTI_ANSWER_DATASETS):
    if ds_name.startswith("longbench/"):
        MULTI_ANSWER_DATASETS.add(f"{ds_name}_e")


# for training closed qa datasets, e.g., hotpot_qa, squad, etc.
CLOSED_QA_INTX_TEMPLATES = [
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
    "Answer without any explanation.\n\nQuestion: {input}",
    "Based on the provided text, what is the answer to the following question? Provide only the answer.\n\nQuestion: {input}",
    "Extract the answer to the question from the text. Be concise. Do not explain.\n\nQuestion: {input}",
    "What is the answer to this question, based on the context? Respond with the answer only.\n\nQuestion: {input}",
    "Provide a direct answer to the question using the given passages. Do not give any explanation.\n\nQuestion: {input}",
    "Answer the question using only information from the provided text. No extra words.\n\nQuestion: {input}",
    "From the passages, answer the question. Just the answer, please.\n\nQuestion: {input}",
    "Give the answer to the question. Do not include any other text.\n\nQuestion: {input}",
    "The answer to the question is in the text. Find it and state it clearly. No need for explanation.\n\nQuestion: {input}",
    "Concisely answer the question based on the text provided. Don't include any other words. Just the answer.\n\nQuestion: {input}",
    "Read the passages and answer the question with the minimal necessary words.\n\nQuestion: {input}",
    "What is the direct response to the question, according to the text? Avoid explanation.\n\nQuestion: {input}",
    "Please provide only the answer to the question, derived from the text.\n\nQuestion: {input}",
    "Using the provided context, answer the question. Output the answer and nothing else.\n\nQuestion: {input}",
    "Identify the answer in the text and present it without elaboration.\n\nQuestion: {input}",
    "Answer the following question based on the text. Your answer should be brief and to the point. No explanation.\n\nQuestion: {input}",
    "Based on the information given, what is the answer to the question? Only state the answer.\n\nQuestion: {input}",
    "Find the answer to the question in the provided passages and write it down. No explanations.\n\nQuestion: {input}",
    "The question is: {input}. Provide the answer based on the text, and nothing more.",
    "Question: {input}\nAnswer directly based on the text provided. No extra words.",
    "Question: {input}\nPlease provide the answer based on the text. No explanation is needed.",
]


EVAL_INTX_TEMPLATES = {
    "ropes": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "drop": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "negative_nq": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "triviaqa_retrieved": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "hotpot_qa": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "squad": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "longbench/qasper": 'Answer the question as concisely as you can, using a single phrase or sentence if possible.\nIf the question cannot be answered based on the information in the article, write "unanswerable".\nIf the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}',
    "longbench/narrativeqa": "Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}",
    "longbench/multifieldqa_en": "Answer the following question. Only output the answer and do not output any other words.\n\nQuestion: {input}",
    # "longbench/multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}",
    "longbench/hotpotqa": "Answer the following question. Only output the answer and do not output any other words.\n\nQuestion: {input}",
    "longbench/2wikimqa": "Answer the following question. Only output the answer and do not output any other words.\n\nQuestion: {input}",
    "longbench/musique": "Answer the following question. Only output the answer and do not output any other words.\n\nQuestion: {input}",
    #
    # "longbench/dureader": "请基于给定的文章回答下述问题\n\n请基于上述文章回答下面的问题。\n\n问题：{input}",
    "longbench/gov_report": "You are given a report by a government agency. Write a one-page summary of the report.",
    "longbench/qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query based on the above meeting transcript.\n\nQuery: {input}",
    "longbench/multi_news": "You are given several news passages. Write a one-page summary of all the news.",
    # "longbench/vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。",
}
for ds_name in LONGBENCH_E_TASKS:
    EVAL_INTX_TEMPLATES[ds_name] = EVAL_INTX_TEMPLATES[ds_name[:-2]]

# for ds_name in DS_KWARGS:
#     if ds_name not in EVAL_INTX_TEMPLATES:
#         EVAL_INTX_TEMPLATES[ds_name] = "{input}"
