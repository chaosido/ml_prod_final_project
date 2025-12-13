import os
import sys

sys.path.append(os.getcwd())
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving.main import (
    LOW_QUALITY_TRANSCRIPT_COUNT,
    PREDICTED_WER_DISTRIBUTION,
    REVIEW_THRESHOLD,
    app,
)


@pytest.fixture
def mock_ml_components():
    """Patches all ML components to isolate metric logic."""
    with (
        patch("serving.main.sf.read") as mock_read,
        patch("serving.main.get_model_loader") as mock_loader_cls,
        patch("serving.main.get_asr_processor") as mock_asr_cls,
        patch("serving.main.AcousticFeatureExtractor") as mock_extractor_cls,
    ):

        # Setup decent defaults so we don't crash
        mock_read.return_value = (np.zeros(16000), 16000)

        mock_extractor = Mock()
        mock_extractor_cls.return_value = mock_extractor
        mock_extractor.extract.return_value = {
            "snr_db": 20,
            "rms_db": -20,
            "duration": 1.0,
        }

        mock_asr = Mock()
        mock_asr_cls.return_value = mock_asr
        mock_asr.process.return_value = {"transcription": "foo", "asr_confidence": 0.9}

        mock_loader = Mock()
        mock_loader_cls.return_value = mock_loader

        yield mock_loader  # Return the loader since we often change its prediction


def get_metric_value(metric, suffix="_total"):
    """Helper to get value from a Counter or specific sample from Histogram."""
    # Collect all samples
    samples = metric.collect()[0].samples

    # For Counters, the sample name usually ends with _total
    # For Histograms, we might look for _count, _sum, or bucket
    for s in samples:
        if s.name.endswith(suffix):
            return s.value
    return 0.0


@pytest.mark.parametrize(
    "predicted_wer, should_increment",
    [
        (0.9, True),  # 0.9 > 0.4 (default threshold) -> Batch of bad quality
        (0.1, False),  # 0.1 <= 0.4 -> Good quality
        (REVIEW_THRESHOLD + 0.01, True),  # Boundary case: just above
        (
            REVIEW_THRESHOLD,
            False,
        ),  # Boundary case: exactly at threshold (usually > check)
    ],
)
def test_low_quality_counter_logic(mock_ml_components, predicted_wer, should_increment):
    """Verify LOW_QUALITY_TRANSCRIPT_COUNT updates correctly based on threshold."""
    # 1. Get initial value
    # Counters usually track _total
    initial_count = get_metric_value(LOW_QUALITY_TRANSCRIPT_COUNT, "_total")

    # 2. Simulate Prediction
    mock_ml_components.predict.return_value = np.array([predicted_wer])

    client = TestClient(app)
    files = {"audio_file": ("test.wav", b"fake", "audio/wav")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200

    # 3. Verify Increment
    final_count = get_metric_value(LOW_QUALITY_TRANSCRIPT_COUNT, "_total")
    expected_delta = 1.0 if should_increment else 0.0

    assert final_count == initial_count + expected_delta


def test_wer_distribution_observation(mock_ml_components):
    """
    Verify PREDICTED_WER_DISTRIBUTION observes the predicted value.
    We check both the Count (number of observations) and Sum (total of observed values).
    """
    # 1. Get initial state
    initial_count = get_metric_value(PREDICTED_WER_DISTRIBUTION, "_count")
    initial_sum = get_metric_value(PREDICTED_WER_DISTRIBUTION, "_sum")

    predicted_val = 0.5
    mock_ml_components.predict.return_value = np.array([predicted_val])

    # 2. Make Request
    client = TestClient(app)
    client.post("/predict", files={"audio_file": ("test.wav", b"fake", "audio/wav")})

    # 3. Verify Updates
    final_count = get_metric_value(PREDICTED_WER_DISTRIBUTION, "_count")
    final_sum = get_metric_value(PREDICTED_WER_DISTRIBUTION, "_sum")

    # Count should increase by 1
    assert final_count == initial_count + 1
    # Sum should increase by the predicted value
    assert np.isclose(final_sum, initial_sum + predicted_val)
