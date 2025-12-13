import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton model loader using lru_cache for thread-safe lazy loading.
    """

    _instance: Optional["ModelLoader"] = None
    _model = None
    _model_path: Path = None

    def __new__(cls, model_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._model_path = None
        return cls._instance

    def load(self, model_path: str) -> None:
        """Load model from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading model from {model_path}")
        self._model = joblib.load(path)
        self._model_path = path
        logger.info("Model loaded successfully")

    @property
    def model(self):
        """Get the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def predict(self, features):
        """Make prediction using loaded model."""
        return self.model.predict(features)


@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    """
    Factory for dependency injection in the FastAPI.
    """
    return ModelLoader()
