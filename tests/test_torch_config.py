# tests/test_torch_config.py
"""Tests for TorchConfig and configure_torch functionality."""
import pytest
import torch


class TestTorchConfig:
    """Tests for TorchConfig dataclass."""

    def test_torch_config_default_values(self):
        """TorchConfig should have sensible defaults."""
        from Auto3D.torch_config import TorchConfig

        config = TorchConfig()
        assert config.allow_tf32 is False
        assert config.cudnn_benchmark is False

    def test_torch_config_custom_values(self):
        """TorchConfig should accept custom values."""
        from Auto3D.torch_config import TorchConfig

        config = TorchConfig(allow_tf32=True, cudnn_benchmark=True)
        assert config.allow_tf32 is True
        assert config.cudnn_benchmark is True


class TestConfigureTorch:
    """Tests for configure_torch function."""

    def test_configure_torch_tf32_enabled(self):
        """Should enable TF32 when configured."""
        from Auto3D.torch_config import TorchConfig, configure_torch

        config = TorchConfig(allow_tf32=True)
        configure_torch(config)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

    def test_configure_torch_tf32_disabled(self):
        """Should disable TF32 when configured."""
        from Auto3D.torch_config import TorchConfig, configure_torch

        config = TorchConfig(allow_tf32=False)
        configure_torch(config)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False

    def test_configure_torch_cudnn_benchmark_enabled(self):
        """Should enable cuDNN benchmark when configured."""
        from Auto3D.torch_config import TorchConfig, configure_torch

        config = TorchConfig(cudnn_benchmark=True)
        configure_torch(config)
        assert torch.backends.cudnn.benchmark is True

    def test_configure_torch_cudnn_benchmark_disabled(self):
        """Should disable cuDNN benchmark when configured."""
        from Auto3D.torch_config import TorchConfig, configure_torch

        config = TorchConfig(cudnn_benchmark=False)
        configure_torch(config)
        assert torch.backends.cudnn.benchmark is False

    def test_configure_torch_with_none_uses_defaults(self):
        """Should use default config when None is passed."""
        from Auto3D.torch_config import configure_torch

        configure_torch(None)
        # Defaults should be TF32 disabled
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False

    def test_configure_torch_idempotent(self):
        """Multiple calls should produce consistent results."""
        from Auto3D.torch_config import TorchConfig, configure_torch

        config = TorchConfig(allow_tf32=True)
        configure_torch(config)
        configure_torch(config)
        assert torch.backends.cuda.matmul.allow_tf32 is True

        config = TorchConfig(allow_tf32=False)
        configure_torch(config)
        assert torch.backends.cuda.matmul.allow_tf32 is False

    def test_configure_torch_sets_fp32_precision_when_available(self):
        """On torch with the modern fp32_precision API, allow_tf32 must map to
        the precision mode ('ieee' for False, 'tf32' for True). cudnn.fp32_precision
        is the decisive check: unlike cuda.matmul, torch does NOT auto-sync it from
        the legacy allow_tf32 flag, so this fails unless configure_torch sets it."""
        from Auto3D.torch_config import TorchConfig, configure_torch

        matmul = torch.backends.cuda.matmul
        cudnn = torch.backends.cudnn
        if not hasattr(matmul, "fp32_precision") or not hasattr(cudnn, "fp32_precision"):
            pytest.skip("torch too old for fp32_precision API")

        configure_torch(TorchConfig(allow_tf32=False))
        assert matmul.fp32_precision == "ieee"
        assert cudnn.fp32_precision == "ieee"

        configure_torch(TorchConfig(allow_tf32=True))
        assert matmul.fp32_precision == "tf32"
        assert cudnn.fp32_precision == "tf32"

        configure_torch(TorchConfig(allow_tf32=False))  # restore default

    def test_configure_torch_deterministic_is_reversible(self):
        """deterministic must turn back off on a later config (was write-once).

        Previously the deterministic flags were only ever set to True, so a
        process that enabled a reproducible run could never restore fast mode.
        warn_only=True is also asserted so AIMNet2/ANI scatter ops warn instead
        of raising under deterministic mode.
        """
        from Auto3D.torch_config import TorchConfig, configure_torch

        try:
            configure_torch(TorchConfig(deterministic=True))
            assert torch.are_deterministic_algorithms_enabled() is True
            assert torch.is_deterministic_algorithms_warn_only_enabled() is True
            assert torch.backends.cudnn.deterministic is True

            configure_torch(TorchConfig(deterministic=False))
            assert torch.are_deterministic_algorithms_enabled() is False
            assert torch.backends.cudnn.deterministic is False
        finally:
            # Restore the default (non-deterministic) global state.
            configure_torch(TorchConfig(deterministic=False))


class TestAuto3DOptionsAllowTf32:
    """Tests for allow_tf32 option in Auto3DOptions."""

    def test_auto3d_options_has_allow_tf32(self):
        """Auto3DOptions should have allow_tf32 field."""
        from Auto3D.config import Auto3DOptions

        config = Auto3DOptions()
        assert hasattr(config, "allow_tf32")
        assert config.allow_tf32 is False  # Default should be False

    def test_auto3d_options_allow_tf32_can_be_set_true(self):
        """allow_tf32 should be settable to True."""
        from Auto3D.config import Auto3DOptions

        config = Auto3DOptions(allow_tf32=True)
        assert config.allow_tf32 is True

    def test_auto3d_options_allow_tf32_can_be_set_false(self):
        """allow_tf32 should be settable to False."""
        from Auto3D.config import Auto3DOptions

        config = Auto3DOptions(allow_tf32=False)
        assert config.allow_tf32 is False


class TestBatchoptNoHardcodedTF32:
    """Tests to ensure batchopt.py doesn't have hardcoded TF32 settings."""

    def test_batchopt_respects_configure_torch(self):
        """batchopt should not override TF32 settings at import."""
        from Auto3D.torch_config import TorchConfig, configure_torch

        # Set TF32 to True
        configure_torch(TorchConfig(allow_tf32=True))
        initial_tf32 = torch.backends.cuda.matmul.allow_tf32

        # Import batchopt - it should NOT change the TF32 setting
        import importlib

        import Auto3D.batch_opt.batchopt
        importlib.reload(Auto3D.batch_opt.batchopt)

        # TF32 should still be True (not overridden by batchopt)
        assert torch.backends.cuda.matmul.allow_tf32 == initial_tf32

    def test_tf32_can_be_toggled_after_batchopt_import(self):
        """TF32 settings should be changeable after batchopt is imported."""
        # Import batchopt first
        import Auto3D.batch_opt.batchopt  # noqa: F401
        from Auto3D.torch_config import TorchConfig, configure_torch

        # Then configure TF32 - this should work
        configure_torch(TorchConfig(allow_tf32=True))
        assert torch.backends.cuda.matmul.allow_tf32 is True

        configure_torch(TorchConfig(allow_tf32=False))
        assert torch.backends.cuda.matmul.allow_tf32 is False
