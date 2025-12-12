import sys
import os
sys.path.append(os.getcwd())
import pytest
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from serving.main import app, LOW_QUALITY_TRANSCRIPT_COUNT, PREDICTED_WER_DISTRIBUTION

@pytest.fixture
def mock_ml_components():
    """Patches all ML components to isolate metric logic."""
    with patch("serving.main.sf.read") as mock_read, \
         patch("serving.main.get_model_loader") as mock_loader_cls, \
         patch("serving.main.get_asr_processor") as mock_asr_cls, \
         patch("serving.main.AcousticFeatureExtractor") as mock_extractor_cls:
        
        # Setup decent defaults so we don't crash
        mock_read.return_value = (np.zeros(16000), 16000)
        
        mock_extractor = Mock()
        mock_extractor_cls.return_value = mock_extractor
        mock_extractor.extract.return_value = {"snr_db": 20, "rms_db": -20, "duration": 1.0}
        
        mock_asr = Mock()
        mock_asr_cls.return_value = mock_asr
        mock_asr.process.return_value = {"transcription": "foo", "asr_confidence": 0.9}
        
        mock_loader = Mock()
        mock_loader_cls.return_value = mock_loader
        
        yield mock_loader  # Return the loader since we often change its prediction

def test_metric_increment_on_bad_quality(mock_ml_components):
    """Verify LOW_QUALITY_TRANSCRIPT_COUNT increments when WER > Threshold."""
    # 1. Get initial value
    before = LOW_QUALITY_TRANSCRIPT_COUNT.collect()[0].samples[0].value
    
    # 2. Simulate High WER (e.g. 0.9)
    # Note: Environment defaults threshold is 0.4
    mock_ml_components.predict.return_value = np.array([0.9])
    
    files = {"audio_file": ("bad.wav", b"fake", "audio/wav")}
    client = TestClient(app)
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    
    # 3. Verify Increment
    after = LOW_QUALITY_TRANSCRIPT_COUNT.collect()[0].samples[0].value
    assert after == before + 1, "Metric did not increment on high WER"

def test_metric_no_increment_on_good_quality(mock_ml_components):
    """Verify LOW_QUALITY_TRANSCRIPT_COUNT does NOT increment when WER is low."""
    before = LOW_QUALITY_TRANSCRIPT_COUNT.collect()[0].samples[0].value
    
    # Simulate Low WER (e.g. 0.1)
    mock_ml_components.predict.return_value = np.array([0.1])
    
    files = {"audio_file": ("good.wav", b"fake", "audio/wav")}
    client = TestClient(app)
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    
    after = LOW_QUALITY_TRANSCRIPT_COUNT.collect()[0].samples[0].value
    assert after == before, "Metric wrongly incremented on good WER"

def test_wer_distribution_observation(mock_ml_components):
    """Verify PREDICTED_WER_DISTRIBUTION observes the predicted value."""
    # This is harder to test exactly without clearing registry, 
    # but we can check if sum/count increases.
    
    initial_sum = PREDICTED_WER_DISTRIBUTION.collect()[0].samples[-1].value # Sum is usually last
    
    mock_ml_components.predict.return_value = np.array([0.5])
    
    client = TestClient(app)
    client.post("/predict", files={"audio_file": ("test.wav", b"fake", "audio/wav")})
    
    new_data = PREDICTED_WER_DISTRIBUTION.collect()[0]
    # In Histogram, samples include buckets, count, and sum.
    # Finding the 'sum' sample (sample names end in _sum)
    sum_sample = next(s for s in new_data.samples if s.name.endswith('_sum'))
    
    assert sum_sample.value > 0
