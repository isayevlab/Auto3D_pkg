# tests/test_optimization_engine_validation.py
"""Tests for validation in optimization_engine.py."""

from unittest.mock import MagicMock

import pytest
import torch

from Auto3D.engines.batch_opt.optimization_engine import n_steps

# Import _validate_state if it exists (will be added in implementation)
try:
    from Auto3D.engines.batch_opt.optimization_engine import _validate_state
except ImportError:
    _validate_state = None


class TestValidateState:
    """Tests for _validate_state function."""

    @pytest.fixture(autouse=True)
    def check_validate_state_exists(self):
        """Skip tests if _validate_state is not yet implemented."""
        if _validate_state is None:
            pytest.skip("_validate_state not yet implemented")

    def test_validate_state_valid_input(self):
        """_validate_state should accept valid tensors."""
        state = {
            "coord": torch.randn(2, 5, 3),  # 3D: (batch, atoms, 3)
            "numbers": torch.ones(2, 5, dtype=torch.long),  # 2D: (batch, atoms)
            "charges": torch.zeros(2),  # 1D: (batch,)
        }
        # Should not raise
        _validate_state(state)

    def test_validate_state_invalid_coord_2d(self):
        """_validate_state should raise ValueError for 2D coord."""
        state = {
            "coord": torch.randn(5, 3),  # Wrong: 2D instead of 3D
            "numbers": torch.ones(1, 5, dtype=torch.long),
            "charges": torch.zeros(1),
        }
        with pytest.raises(ValueError, match="coord.*3D"):
            _validate_state(state)

    def test_validate_state_invalid_coord_4d(self):
        """_validate_state should raise ValueError for 4D coord."""
        state = {
            "coord": torch.randn(1, 2, 5, 3),  # Wrong: 4D instead of 3D
            "numbers": torch.ones(1, 5, dtype=torch.long),
            "charges": torch.zeros(1),
        }
        with pytest.raises(ValueError, match="coord.*3D"):
            _validate_state(state)

    def test_validate_state_invalid_numbers_1d(self):
        """_validate_state should raise ValueError for 1D numbers."""
        state = {
            "coord": torch.randn(1, 5, 3),
            "numbers": torch.ones(5, dtype=torch.long),  # Wrong: 1D instead of 2D
            "charges": torch.zeros(1),
        }
        with pytest.raises(ValueError, match="numbers.*2D"):
            _validate_state(state)

    def test_validate_state_invalid_numbers_3d(self):
        """_validate_state should raise ValueError for 3D numbers."""
        state = {
            "coord": torch.randn(1, 5, 3),
            "numbers": torch.ones(1, 1, 5, dtype=torch.long),  # Wrong: 3D instead of 2D
            "charges": torch.zeros(1),
        }
        with pytest.raises(ValueError, match="numbers.*2D"):
            _validate_state(state)

    def test_validate_state_invalid_charges_0d(self):
        """_validate_state should raise ValueError for 0D charges."""
        state = {
            "coord": torch.randn(1, 5, 3),
            "numbers": torch.ones(1, 5, dtype=torch.long),
            "charges": torch.tensor(0),  # Wrong: 0D (scalar) instead of 1D
        }
        with pytest.raises(ValueError, match="charges.*1D"):
            _validate_state(state)

    def test_validate_state_invalid_charges_2d(self):
        """_validate_state should raise ValueError for 2D charges."""
        state = {
            "coord": torch.randn(1, 5, 3),
            "numbers": torch.ones(1, 5, dtype=torch.long),
            "charges": torch.zeros(1, 1),  # Wrong: 2D instead of 1D
        }
        with pytest.raises(ValueError, match="charges.*1D"):
            _validate_state(state)


class TestNStepsValidation:
    """Tests for n_steps validation behavior."""

    def test_n_steps_validates_coord_shape(self):
        """n_steps should raise ValueError for invalid coord shape."""
        invalid_state = {
            "coord": torch.randn(5, 3),  # Missing batch dimension
            "numbers": torch.ones(1, 5, dtype=torch.long),
            "charges": torch.zeros(1),
            "converged_mask": torch.zeros(1, dtype=torch.bool),
            "fmax": torch.zeros(1),
            "energy": torch.zeros(1, dtype=torch.double),
            "nn": None,
        }

        with pytest.raises(ValueError, match="coord.*3D"):
            n_steps(invalid_state, n=10, opttol=0.01, patience=100)

    def test_n_steps_validates_numbers_shape(self):
        """n_steps should raise ValueError for invalid numbers shape."""
        invalid_state = {
            "coord": torch.randn(1, 5, 3),
            "numbers": torch.ones(5, dtype=torch.long),  # Missing batch dimension
            "charges": torch.zeros(1),
            "converged_mask": torch.zeros(1, dtype=torch.bool),
            "fmax": torch.zeros(1),
            "energy": torch.zeros(1, dtype=torch.double),
            "nn": None,
        }

        with pytest.raises(ValueError, match="numbers.*2D"):
            n_steps(invalid_state, n=10, opttol=0.01, patience=100)

    def test_n_steps_validates_charges_shape(self):
        """n_steps should raise ValueError for invalid charges shape."""
        invalid_state = {
            "coord": torch.randn(1, 5, 3),
            "numbers": torch.ones(1, 5, dtype=torch.long),
            "charges": torch.zeros(1, 1),  # Should be 1D, not 2D
            "converged_mask": torch.zeros(1, dtype=torch.bool),
            "fmax": torch.zeros(1),
            "energy": torch.zeros(1, dtype=torch.double),
            "nn": None,
        }

        with pytest.raises(ValueError, match="charges.*1D"):
            n_steps(invalid_state, n=10, opttol=0.01, patience=100)

    def test_n_steps_error_is_not_assertion_error(self):
        """n_steps validation errors should be ValueError, not AssertionError."""
        invalid_state = {
            "coord": torch.randn(5, 3),  # Invalid shape
            "numbers": torch.ones(1, 5, dtype=torch.long),
            "charges": torch.zeros(1),
            "converged_mask": torch.zeros(1, dtype=torch.bool),
            "fmax": torch.zeros(1),
            "energy": torch.zeros(1, dtype=torch.double),
            "nn": None,
        }

        # Should NOT raise AssertionError, and must be raised for the coord
        # shape defect this fixture actually has -- a bare `ValueError` would
        # also pass for e.g. an unrelated numbers/charges ValueError, so pin
        # the message to the coord/3D guard this fixture is built to hit.
        with pytest.raises(ValueError, match="coord.*3D"):
            n_steps(invalid_state, n=10, opttol=0.01, patience=100)
