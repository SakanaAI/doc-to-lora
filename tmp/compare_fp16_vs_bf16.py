# filepath: /home/tan/research/ctx-to-lora/tmp/compare_fp16_vs_bf16.py

import torch


def compare_fp16_vs_bf16():
    """
    Compare operations between float16 and bfloat16 tensors.
    Computes difference and multiplication to see output dtypes.
    """

    # Create sample tensors with the same values but different dtypes
    values = [[1.5, 2.3, 3.7], [4.1, 5.9, 6.2]]

    # Create tensors with different dtypes
    tensor_fp16 = torch.tensor(values, dtype=torch.float16)
    tensor_bf16 = torch.tensor(values, dtype=torch.bfloat16)

    print("Original tensors:")
    print(f"FP16 tensor: {tensor_fp16}")
    print(f"FP16 dtype: {tensor_fp16.dtype}")
    print(f"BF16 tensor: {tensor_bf16}")
    print(f"BF16 dtype: {tensor_bf16.dtype}")
    print()

    # Compute difference (subtraction)
    diff_result = tensor_fp16 - tensor_bf16
    print("Difference operation (FP16 - BF16):")
    print(f"Result: {diff_result}")
    print(f"Result dtype: {diff_result.dtype}")
    print()

    # Compute multiplication
    mult_result = tensor_fp16 * tensor_bf16
    print("Multiplication operation (FP16 * BF16):")
    print(f"Result: {mult_result}")
    print(f"Result dtype: {mult_result.dtype}")
    print()

    # Test the reverse order to see if it matters
    diff_result_reverse = tensor_bf16 - tensor_fp16
    mult_result_reverse = tensor_bf16 * tensor_fp16

    print("Reverse order operations:")
    print(f"BF16 - FP16 dtype: {diff_result_reverse.dtype}")
    print(f"BF16 * FP16 dtype: {mult_result_reverse.dtype}")
    print()

    # Show the actual values to check precision differences
    print("Value comparison:")
    print(f"Original values: {values}")
    print(f"FP16 values: {tensor_fp16.tolist()}")
    print(f"BF16 values: {tensor_bf16.tolist()}")
    print(f"Difference values: {diff_result.tolist()}")
    print(f"Multiplication values: {mult_result.tolist()}")


if __name__ == "__main__":
    compare_fp16_vs_bf16()
