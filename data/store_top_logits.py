import argparse
import io
import zipfile
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer


# e.g., python data/store_top_logits.py --data "temp/test_data.parquet" --model "Qwen/Qwen3-0.6B"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help='path to a parquet shard of the dataset')
    ap.add_argument('--model', required=True,
                    help='huggingface model name or checkpoint')
    ap.add_argument('--max_logits_to_store', type=int, default=100,
                    help='max number of logits kept per completion')
    ap.add_argument('--limit_samples', type=int, default=None,
                    help='process only the first n dataset rows')
    ap.add_argument('--logit_precision', default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'],
                    help='dtype for saved logits')
    ap.add_argument('--index_precision', default='int32',
                    choices=['int64', 'int32', 'int16'],
                    help='dtype for saved indices')
    args = ap.parse_args()

    float_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }
    int_map = {
        'int64': torch.int64,
        'int32': torch.int32,
        'int16': torch.int16,
    }
    l_dtype = float_map[args.logit_precision]
    i_dtype = int_map[args.index_precision]

    ds = load_dataset('parquet', data_files=args.data)['train']
    if args.limit_samples and args.limit_samples > 0:
        ds = ds.select(range(min(args.limit_samples, len(ds))))

    tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    base = Path(args.data).with_suffix('')
    model_dir = args.model.replace('/', '__')
    out_dir = base / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit_samples:
        zip_name = f'limited_logits_{args.limit_samples}.zip'
    else:
        zip_name = 'all_logits.zip'
    zip_path = out_dir / zip_name

    samp_id = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for row in tqdm(ds):
            ctx = row['context']
            qs = row['prompts_level_0']
            ans = row['responses_level_0']
            q_list = []
            a_list = []
            vals_list = []
            sel_pos_list = []
            for q, a in zip(qs, ans):
                messages = [
                    {'role': 'user', 'content': ctx + '\n' + q},
                    {'role': 'assistant', 'content': a},
                ]
                tpl = tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_special_token=False,
                    truncation=False,
                    add_generation_prompt=False,
                    return_dict=True,
                )
                tokens_to_mask = tok.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_special_token=False,
                    truncation=False,
                    add_generation_prompt=True,
                    return_dict=True,
                )
                number_of_tokens_to_mask = len(tokens_to_mask['input_ids'])
                assert tokens_to_mask['input_ids'] == tpl['input_ids'][:number_of_tokens_to_mask]

                inp = torch.tensor([tpl['input_ids']], device=device)
                with torch.no_grad():
                    logits = model(inp).logits
                logits_for_answer = logits[0, number_of_tokens_to_mask - 1:, :]
                assert logits_for_answer.shape[0] > 0
                assert logits_for_answer.shape[0] + number_of_tokens_to_mask -1 == inp.size(1)

                # just in case, logits to store is super large
                k = min(args.max_logits_to_store, logits_for_answer.shape[-1])
                vals, idxs = torch.topk(logits_for_answer, k, dim=-1)
                q_list.append(q)
                a_list.append(a)
                vals_list.append(vals.cpu())
                sel_pos_list.append(idxs.cpu())
            record = {
                'context': ctx,
                'question': q_list,
                'answer': a_list,
                'top_logits': vals_list,
                'top_positions_positions': sel_pos_list,
            }
            buf = io.BytesIO()
            torch.save(record, buf)
            samp_id += 1
            zf.writestr(f'{samp_id:07d}.pt', buf.getvalue())


if __name__ == '__main__':
    main()
