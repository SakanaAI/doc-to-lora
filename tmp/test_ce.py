#!/usr/bin/env python
"""
Test cross entropy loss shape with reduction=None
"""

import torch
from torch import nn

# Create sample data
batch_size, seq_len, vocab_size = 2, 5, 1000
logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

# Flatten for cross entropy (expects 2D input)
logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
labels_flat = labels.view(-1)  # (batch_size * seq_len,)

print(f"Input shapes:")
print(f"  logits_flat: {logits_flat.shape}")
print(f"  labels_flat: {labels_flat.shape}")

loss = nn.functional.cross_entropy(logits_flat, labels_flat, reduction="none")

print(f"Output shape:")
print(f"  loss: {loss.shape}")
print(f"  loss values (first 10): {loss[:10]}")
