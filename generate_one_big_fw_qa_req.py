import json
from glob import glob

if __name__ == "__main__":
    res_files = glob("openai_batches/fineweb_qa_pairs_*_res.jsonl")
    prompt_files = glob("openai_batches/fineweb_qa_pairs_*[!res].jsonl")
    unprocessed_files = set(prompt_files) - set(
        [f.replace("_res", "") for f in res_files]
    )
    print(f"Found {len(unprocessed_files)} unprocessed files:\n{unprocessed_files}")
    # Concatenate "body" from each line from unprocessed files
    with open("openai_batches/fineweb_qa_pairs_large.jsonl", "w") as outfile:
        for fname in unprocessed_files:
            with open(fname) as infile:
                for line in infile:
                    # Load the JSON object from the line
                    obj = json.loads(line)
                    # Extract the "body" and write it as a new line
                    if "body" in obj:
                        outfile.write(json.dumps(obj["body"]) + "\n")

    print(f"Concatenated 'body' from {len(unprocessed_files)} files")
