import torch

# Setup from your example
x = torch.rand(16, 26, 8, 2048)
a = torch.rand(1, 26, 8, 1)

# Original operation for comparison
y = x * a

# --- Einsum Solutions ---

# 1. Explicit form (best for clarity)
# We assign letters to each dimension and specify the exact output format.
y_einsum_explicit = torch.einsum("ijkl,ijkl->ijkl", x, a)

# 2. Implicit form (more concise)
# For element-wise products, you can omit the output part.
# Einsum infers the output should contain all unique indices.
y_einsum_implicit = torch.einsum("ijkl,ijkl", x, a)


# --- Verification ---
print("Original shape:", y.shape)
print("Einsum shape:", y_einsum_explicit.shape)

# Check if the results are numerically the same
print("\nAre the results the same?")
print("Explicit form match:", torch.allclose(y, y_einsum_explicit))
print("Implicit form match:", torch.allclose(y, y_einsum_implicit))

print(f"{y}=")
print(f"{y_einsum_explicit=}")
print(f"{y_einsum_implicit=}")
