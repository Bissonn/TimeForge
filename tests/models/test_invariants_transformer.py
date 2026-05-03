import pytest
import torch
import numpy as np
from models.transformer import LocalAttention

pytestmark = pytest.mark.unit


def test_local_attention_causality_invariant():
    """
    CRITICAL TEST: Verifies if LocalAttention 'peeks' into the future.
    Invariant Principle: Changing the last token (t) cannot affect the output at step (t-1).
    """
    # Configuration
    batch = 1
    seq_len = 10
    embed_dim = 4
    num_heads = 1
    window_size = 4  # Local window

    layer = LocalAttention(embed_dim, num_heads, window_size, dropout=0.0)
    layer.eval()  # Disable dropout for deterministic check

    # Sequence A
    input_a = torch.randn(batch, seq_len, embed_dim)

    # Sequence B: identical to A, but with the LAST token changed
    input_b = input_a.clone()
    input_b[:, -1, :] += 100.0  # Large change at the end

    # Forward pass
    out_a, _ = layer(input_a)
    out_b, _ = layer(input_b)

    # Check prediction for time t-2 (index -2).
    # If causal masking works correctly, the result at t-2 should depend only on t-2, t-3...
    # It MUST NOT depend on t-1 (the last token in this case).

    diff = torch.abs(out_a[:, -2, :] - out_b[:, -2, :]).sum().item()

    print(f"\nOutput difference at step t-1 after changing input at t: {diff}")

    assert diff < 1e-6, \
        f"Causality violation! Future change (t) affected past (t-1). Diff: {diff}"