import argparse
import pandas as pd

def make_test_samples():
    return [
        {
            "context": "Alice was beginning to get very tired of sitting by her sister on the bank.",
            "prompts_level_0": [
                "Who is sitting by Alice?",
                "How is Alice feeling at the start?"
            ],
            "responses_level_0": [
                "Her sister is sitting by her.",
                "She is very tired."
            ],
        },
        {
            "context": "<think>debug info</think>Then she saw a White Rabbit with pink eyes run close by her.",
            "prompts_level_0": [
                "What did Alice see run by?",
                "What color were its eyes?"
            ],
            "responses_level_0": [
                "She saw a White Rabbit.",
                "Its eyes were pink."
            ],
        },
        {
            "context": "x" * 200,
            "prompts_level_0": ["How long is this context?"],
            "responses_level_0": ["It is two hundred characters long."],
        },
    ]

def main(output_path: str):
    samples = make_test_samples()
    df = pd.DataFrame(samples)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} test rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="test_shard.parquet",
        help="Where to write the test Parquet file"
    )
    args = parser.parse_args()
    main(args.output)
