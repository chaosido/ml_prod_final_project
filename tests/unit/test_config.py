import os
import pytest
from asr_qe.config import Settings

def test_settings_defaults(monkeypatch):
    """Verify default values are used when environment variables are not set."""
    # Ensure relevant env vars are unset for this test
    monkeypatch.delenv("REVIEW_THRESHOLD", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("ASR_QE_MODEL_PATH", raising=False)

    settings = Settings()
    
    assert settings.review_threshold == 0.4
    assert settings.log_level == "INFO"
    assert settings.model_path == "models/model.joblib"

def test_settings_env_override(monkeypatch):
    """Verify environment variables override defaults."""
    monkeypatch.setenv("REVIEW_THRESHOLD", "0.75")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ASR_QE_MODEL_PATH", "/tmp/custom_model.joblib")
    
    settings = Settings()
    
    assert settings.review_threshold == 0.75
    assert settings.log_level == "DEBUG"
    assert settings.model_path == "/tmp/custom_model.joblib"

def test_type_conversion(monkeypatch):
    """Verify numeric values are correctly converted from string env vars."""
    monkeypatch.setenv("REVIEW_THRESHOLD", "0.9")
    
    settings = Settings()
    
    assert isinstance(settings.review_threshold, float)
    assert settings.review_threshold == 0.9
