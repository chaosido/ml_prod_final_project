import os
import sys

sys.path.append(os.getcwd())
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from serving.main import app


@patch("serving.main.get_model_loader")
@patch("serving.main.get_asr_processor")
def test_health_check(mock_get_asr, mock_get_loader):
    """Verify health check endpoint works without mocks."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "version" in response.json()


@patch("serving.main.get_model_loader")
@patch("serving.main.get_asr_processor")
def test_metrics_endpoint(mock_get_asr, mock_get_loader):
    """Verify Prometheus metrics endpoint is exposed."""
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        # It should return plain text
        assert "input_audio_snr_db" in response.text
        assert "http_requests_total" in response.text


@patch("serving.main.sf.read")
@patch("serving.main.get_model_loader")
@patch("serving.main.get_asr_processor")
@patch("serving.main.AcousticFeatureExtractor")
def test_predict_flow(mock_acoustic_cls, mock_get_asr, mock_get_loader, mock_sf_read):
    """
    Verify complete prediction flow with mocked ML components.
    """
    # 1. Mock Audio Reader (sf.read)
    # Returns (data, samplerate)
    mock_audio_data = np.zeros(16000)
    mock_sf_read.return_value = (mock_audio_data, 16000)

    # 2. Mock Acoustic Features
    mock_acoustic_instance = Mock()
    mock_acoustic_cls.return_value = mock_acoustic_instance
    mock_acoustic_instance.extract.return_value = {
        "rms_db": -25.0,
        "snr_db": 15.0,
        "duration": 1.0,
    }

    # 3. Mock ASR Processor
    mock_asr_instance = Mock()
    mock_get_asr.return_value = mock_asr_instance
    mock_asr_instance.process.return_value = {
        "transcription": "mock transcription",
        "asr_confidence": 0.95,
    }

    # 4. Mock Model Loader (XGBoost)
    mock_loader_instance = Mock()
    mock_get_loader.return_value = mock_loader_instance
    mock_loader_instance.predict.return_value = np.array([0.15])

    # 5. Execute Request
    files = {"audio_file": ("test_audio.wav", b"fake_content", "audio/wav")}

    with TestClient(app) as client:
        response = client.post("/predict", files=files)

    # 6. Assertions
    assert response.status_code == 200, f"Failed with {response.text}"
    data = response.json()

    assert data["predicted_wer"] == 0.15
    assert data["transcript"] == "mock transcription"
    assert data["review_recommended"] is False

    mock_sf_read.assert_called()
    mock_acoustic_instance.extract.assert_called()
    mock_asr_instance.process.assert_called()
    mock_loader_instance.predict.assert_called()


@patch("serving.main.sf.read")
@patch("serving.main.get_model_loader")
@patch("serving.main.get_asr_processor")
@patch("serving.main.AcousticFeatureExtractor")
def test_predict_bad_quality(
    mock_acoustic_cls, mock_get_asr, mock_get_loader, mock_sf_read
):
    """Verify 'review_recommended' flag is True for high WER predictions."""
    # Setup basics
    mock_sf_read.return_value = (np.zeros(16000), 16000)
    mock_acoustic_cls.return_value.extract.return_value = {
        "snr_db": 0,
        "rms_db": -50.0,
        "duration": 1.0,
    }
    mock_get_asr.return_value.process.return_value = {
        "transcription": "",
        "asr_confidence": 0.1,
    }

    # Simulate high WER
    mock_get_loader.return_value.predict.return_value = np.array([0.85])

    files = {"audio_file": ("bad_audio.wav", b"fake", "audio/wav")}

    with TestClient(app) as client:
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert response.json()["review_recommended"] is True
