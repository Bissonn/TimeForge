"""
Tests for AttentionAnalyzer - Clean Architecture.

Tests verify:
1. Analyzer works with any directory (no RunContext)
2. Explicit parameter loading
3. NPZ file handling
4. ERF computation
"""

import pytest
import numpy as np
from pathlib import Path
from analysis.attention_analysis import AttentionAnalyzer


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_attention_file(path: Path, model_name: str, fold_idx: int, window_size: int, seq_len: int = 32):
    """Create a test attention NPZ file."""
    # Create dummy attention weights: (batch=2, heads=4, seq_len, seq_len)
    attention = np.random.rand(2, 4, seq_len, seq_len)

    # Normalize along last dimension (attention sums to 1)
    attention = attention / attention.sum(axis=-1, keepdims=True)

    filename = f"{model_name}_fold_{fold_idx}_w{window_size}_attention.npz"
    filepath = path / filename

    np.savez_compressed(filepath, layer_0=attention)

    return filepath


# ============================================================================
# Tests - Initialization
# ============================================================================

class TestAttentionAnalyzerInit:
    """Test AttentionAnalyzer initialization."""

    def test_analyzer_init_with_path(self, tmp_path):
        """Analyzer initializes with directory path."""
        attention_dir = tmp_path / "attention"
        attention_dir.mkdir()

        analyzer = AttentionAnalyzer(attention_dir=attention_dir)

        assert analyzer.attention_dir == attention_dir

    def test_analyzer_accepts_any_path(self, tmp_path):
        """Analyzer works with any directory path."""
        custom_dir = tmp_path / "my_custom" / "attention_logs"
        custom_dir.mkdir(parents=True)

        analyzer = AttentionAnalyzer(attention_dir=custom_dir)

        assert analyzer.attention_dir == custom_dir


# ============================================================================
# Tests - Loading
# ============================================================================

class TestAttentionLoading:
    """Test attention pattern loading."""

    def test_load_attention_patterns_explicit_params(self, tmp_path):
        """Loads attention with explicit parameters."""
        attention_dir = tmp_path / "attention"
        attention_dir.mkdir()

        # Create test file
        create_test_attention_file(attention_dir, "transformer", 0, 96)

        # Load
        analyzer = AttentionAnalyzer(attention_dir=attention_dir)
        patterns = analyzer.load_attention_patterns(
            model_name="transformer",
            window_size=96,
            fold_idx=0
        )

        assert isinstance(patterns, dict)
        assert 0 in patterns  # layer_0
        assert patterns[0].shape[0] == 2  # batch
        assert patterns[0].shape[1] == 4  # heads

    def test_load_attention_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when file doesn't exist."""
        attention_dir = tmp_path / "attention"
        attention_dir.mkdir()

        analyzer = AttentionAnalyzer(attention_dir=attention_dir)

        with pytest.raises(FileNotFoundError):
            analyzer.load_attention_patterns("nonexistent", 96, 0)


# ============================================================================
# Tests - Artifact Discovery
# ============================================================================

class TestArtifactDiscovery:
    """Test artifact discovery functionality."""

    def test_discover_all_artifacts_finds_files(self, tmp_path):
        """Discovers all attention NPZ files in directory."""
        attention_dir = tmp_path / "attention"
        attention_dir.mkdir()

        # Create multiple files
        for i in range(3):
            create_test_attention_file(attention_dir, f"model{i}", 0, 96)

        analyzer = AttentionAnalyzer(attention_dir=attention_dir)
        artifacts = analyzer.discover_all_artifacts()

        assert len(artifacts) == 3
        assert all(p.suffix == '.npz' for p in artifacts)
        assert all('attention' in p.name for p in artifacts)

    def test_discover_artifacts_empty_directory(self, tmp_path):
        """Returns empty list for empty directory."""
        attention_dir = tmp_path / "attention"
        attention_dir.mkdir()

        analyzer = AttentionAnalyzer(attention_dir=attention_dir)
        artifacts = analyzer.discover_all_artifacts()

        assert len(artifacts) == 0


# ============================================================================
# Tests - ERF Computation
# ============================================================================

class TestERFComputation:
    """Test Effective Receptive Field computation."""

    def test_compute_erf_basic(self, tmp_path):
        """Computes ERF for attention weights."""
        # Create focused attention (high values near diagonal)
        seq_len = 32
        batch, heads = 2, 4

        # Attention focused on recent timesteps (near diagonal)
        attention = np.zeros((batch, heads, seq_len, seq_len))
        for i in range(seq_len):
            for j in range(max(0, i-5), i+1):
                attention[:, :, i, j] = 1.0

        # Normalize
        attention = attention / attention.sum(axis=-1, keepdims=True)

        analyzer = AttentionAnalyzer(attention_dir=tmp_path)
        erf = analyzer.compute_effective_receptive_field(attention, threshold=0.5)

        # ERF should be small (focused on recent timesteps)
        assert erf > 0
        assert erf < seq_len

    def test_compute_erf_full_attention(self, tmp_path):
        """Computes ERF for full attention (uniform)."""
        seq_len = 32
        batch, heads = 2, 4

        # Uniform attention (attends to all positions equally)
        attention = np.ones((batch, heads, seq_len, seq_len))
        attention = attention / attention.sum(axis=-1, keepdims=True)

        analyzer = AttentionAnalyzer(attention_dir=tmp_path)
        erf = analyzer.compute_effective_receptive_field(attention, threshold=0.5)

        # ERF should be large (uses full context)
        assert 5 < erf < seq_len  # Between 5 and 32


# ============================================================================
# Tests - Integration with RunContext
# ============================================================================

class TestRunContextIntegration:
    """Test analyzer works with RunContext directories."""

    def test_analyzer_with_context_directory(self, fold_context):
        """Analyzer works with directory from RunContext."""
        # Create test data in context directory
        create_test_attention_file(
            fold_context.attention_dir,
            "transformer",
            0,
            96
        )

        # Create analyzer with context directory
        analyzer = AttentionAnalyzer(attention_dir=fold_context.attention_dir)

        # Load data
        patterns = analyzer.load_attention_patterns("transformer", 96, 0)

        assert len(patterns) > 0

    def test_analyzer_discovers_context_artifacts(self, base_context, multi_fold_contexts):
        """Analyzer discovers artifacts from multiple fold contexts."""
        # Create artifacts for each fold context
        for fold_ctx in multi_fold_contexts:
            create_test_attention_file(
                fold_ctx.attention_dir,
                fold_ctx.model_name,
                fold_ctx.fold_idx,
                fold_ctx.window_size
            )

        # Analyzer on base attention directory
        analyzer = AttentionAnalyzer(attention_dir=base_context.attention_dir)
        artifacts = analyzer.discover_all_artifacts()

        # Should find all 3 fold artifacts
        assert len(artifacts) == 3
