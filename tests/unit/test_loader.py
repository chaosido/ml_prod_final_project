"""
Unit tests for the model loader module.
"""
import pytest
import tempfile
from pathlib import Path
import numpy as np

from asr_qe.models.loader import ModelLoader, get_model_loader


class SimpleModel:
    """A simple picklable model for testing."""
    def predict(self, X):
        return [0.5] * len(X)


@pytest.fixture
def model_file():
    """Create a temporary model file with a real picklable model."""
    import joblib
    
    model = SimpleModel()
    
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        joblib.dump(model, f.name)
        yield f.name
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


def test_loader_singleton():
    """Test that ModelLoader follows singleton pattern."""
    # Reset singleton for test isolation
    ModelLoader._instance = None
    
    loader1 = ModelLoader()
    loader2 = ModelLoader()
    
    assert loader1 is loader2


def test_load_model(model_file):
    """Test loading a model from disk."""
    # Reset singleton
    ModelLoader._instance = None
    
    loader = ModelLoader()
    loader.load(model_file)
    
    assert loader.is_loaded
    assert loader.model is not None


def test_load_missing_file():
    """Test loading from non-existent path raises error."""
    ModelLoader._instance = None
    
    loader = ModelLoader()
    
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/path/model.joblib")


def test_predict_without_load():
    """Test that predict fails if model not loaded."""
    ModelLoader._instance = None
    
    loader = ModelLoader()
    
    with pytest.raises(RuntimeError, match="not loaded"):
        loader.predict([[1, 2, 3]])


def test_predict_after_load(model_file):
    """Test prediction after loading model."""
    ModelLoader._instance = None
    
    loader = ModelLoader()
    loader.load(model_file)
    
    result = loader.predict([[1, 2, 3]])
    assert result == [0.5]


def test_get_model_loader_factory():
    """Test the factory function returns singleton."""
    # Clear the lru_cache
    get_model_loader.cache_clear()
    ModelLoader._instance = None
    
    loader1 = get_model_loader()
    loader2 = get_model_loader()
    
    assert loader1 is loader2
