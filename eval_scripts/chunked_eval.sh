WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad ropes drop longbench/squad_e longbench/gov_report_e test_longbench/multifieldqa_en_e --split test --max_ctx_chunk_len -1 --lora_aggregation mean

WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad ropes drop longbench/squad_e longbench/gov_report_e test_longbench/multifieldqa_en_e --split test --max_ctx_chunk_len -1 --lora_aggregation sum

WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad ropes drop longbench/squad_e longbench/gov_report_e test_longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 128 --lora_aggregation mean

WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad ropes drop longbench/squad_e longbench/gov_report_e test_longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 128 --lora_aggregation sum

WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad ropes drop longbench/squad_e longbench/gov_report_e test_longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 256 --lora_aggregation mean

WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad ropes drop longbench/squad_e longbench/gov_report_e test_longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 256 --lora_aggregation sum

WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad ropes drop longbench/squad_e longbench/gov_report_e test_longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 512 --lora_aggregation mean

WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad ropes drop longbench/squad_e longbench/gov_report_e test_longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 512 --lora_aggregation sum
