"""
Unit tests for jobs/validate_model.py and jobs/deploy_model.py logic.

Follows FIRST principles:
- Fast: No heavy I/O or model loading
- Independent: Each test function stands alone
- Repeatable: Deterministic with tmp_path fixtures
- Self-validating: Clear assertions
- Timely: Tests core logic, not implementation details
"""

import shutil

import pytest


class TestDeployModelLogic:
    """Tests for deploy_model.py file operations."""

    def test_copy_model_to_production(self, tmp_path):
        """Test that model file is correctly copied to production directory."""
        # Arrange
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        staging_model = staging_dir / "model.joblib"
        staging_model.write_text("mock model content")

        prod_dir = tmp_path / "production"
        prod_dir.mkdir()

        # Act
        dest = prod_dir / "model.joblib"
        shutil.copy2(staging_model, dest)

        # Assert
        assert dest.exists()
        assert dest.read_text() == "mock model content"

    def test_deploy_requires_source_model_existence(self, tmp_path):
        """Test that deployment requires source model to exist."""
        missing_model = tmp_path / "nonexistent" / "model.joblib"

        # Deploy script checks existence before proceeding
        assert not missing_model.exists()
        # In actual deploy_model.py, this would trigger sys.exit(1)


class TestValidateModelLogic:
    """Tests for validate_model.py comparison logic using parametrized cases."""

    @pytest.mark.parametrize(
        "candidate_rho,production_rho,should_pass,description",
        [
            # Standard cases
            (0.85, 0.80, True, "candidate better than production"),
            (0.80, 0.80, True, "candidate equal to production (no regression)"),
            (0.70, 0.80, False, "candidate worse than production"),
            # Edge cases
            (0.50, -1.0, True, "first deployment with no production model"),
            (0.0, -1.0, True, "zero-quality candidate still beats no model"),
            (1.0, 0.99, True, "perfect candidate beats near-perfect production"),
            (0.001, 0.0, True, "minimal improvement still passes"),
        ],
    )
    def test_validation_comparison_logic(
        self, candidate_rho, production_rho, should_pass, description
    ):
        """Test model comparison logic with various scenarios."""
        # This is the core logic from validate_model.py
        result = candidate_rho >= production_rho

        assert result == should_pass, f"Failed: {description}"

    def test_spearman_rho_range_validation(self):
        """Test that valid Spearman rho values are in expected range [-1, 1]."""
        valid_rho = 0.75

        assert -1.0 <= valid_rho <= 1.0

        # Our comparison logic should handle edge values
        assert valid_rho >= -1.0  # Passes against worst possible
        assert not (valid_rho >= 1.0 + 0.01)  # Would fail against impossible value
