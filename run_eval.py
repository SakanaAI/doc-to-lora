import logging

from ctx_to_lora.eval_utils import (
    run_eval,
)

logger = logging.getLogger()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Evaluate a base model from HuggingFace Hub, without loading checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help=(
            "Specific datasets to evaluate on."
            "If not provided, uses default from args.yaml"
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Eval batch size for teacher forcing",
    )
    parser.add_argument(
        "--eval_batch_size_gen",
        type=int,
        default=32,
        help="Eval batch size for generation",
    )
    parser.add_argument(
        "--max_val_samples_per_ds",
        type=int,
        default=-1,
        help=(
            "Maximum number of validation samples per dataset. "
            "If -1, uses values from checkpoint config."
        ),
    )
    parser.add_argument(
        "--max_ctx_chunk_len",
        type=int,
        default=-1,
        help="Maximum length of context chunk for evaluation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate during evaluation",
    )
    parser.add_argument(
        "--remove_context",
        action="store_true",
        help="Remove context when evaluating the base model.",
    )

    cli_args = vars(parser.parse_args())
    # setup_logging(output_dir, debug=os.getenv("DEBUG", False))

    if cli_args["model_name_or_path"]:
        assert cli_args["max_ctx_chunk_len"] <= 0, (
            f"Evaluating base model shouldn't be used with `max_ctx_chunk_len`"
        )

    eval_batch_size_gen = cli_args.pop("eval_batch_size_gen")
    eval_batch_size = cli_args.pop("eval_batch_size")
    # run_eval(
    #     **cli_args,
    #     # cli_args.checkpoint_path,
    #     # cli_args.model_name_or_path,
    #     # cli_args.eval_batch_size,
    #     # args,
    #     # split=cli_args.split,
    #     eval_batch_size=eval_batch_size,
    #     generative=False,
    # )
    run_eval(
        **cli_args,
        # cli_args.checkpoint_path,
        # cli_args.model_name_or_path,
        # cli_args.eval_batch_size_gen,
        # args,
        # split=cli_args.split,
        eval_batch_size=eval_batch_size_gen,
        generative=True,
    )
