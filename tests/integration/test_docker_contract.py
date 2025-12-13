import io
import os
import time

import numpy as np
import pytest
import requests
import soundfile as sf

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


def wait_for_api(timeout=30):
    """Waits for the API to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{API_URL}/")
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


@pytest.mark.integration
def test_docker_health_contract():
    """Verify the container exposes the health check."""
    if not wait_for_api(timeout=5):
        pytest.skip("API not available - is Docker running?")

    resp = requests.get(f"{API_URL}/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.integration
def test_docker_metrics_exposed():
    """Verify Prometheus metrics are used correctly."""
    if not wait_for_api(timeout=1):
        pytest.skip("API not available - is Docker running?")

    resp = requests.get(f"{API_URL}/metrics")
    assert resp.status_code == 200
    assert "input_audio_snr_db" in resp.text
    assert "predicted_wer_bucket" in resp.text


@pytest.mark.integration
def test_docker_predict_endpoint():
    """Verify the container can actually run a prediction (Black Box)."""
    if not wait_for_api(timeout=1):
        pytest.skip("API not available - is Docker running?")

    # Create a dummy WAV file in memory
    # 1 second of silence

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(16000), 16000, format="WAV", subtype="PCM_16")
    buffer.seek(0)

    files = {"audio_file": ("silence.wav", buffer, "audio/wav")}

    resp = requests.post(f"{API_URL}/predict", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_wer" in data
    assert isinstance(data["predicted_wer"], float)
    assert "transcript" in data
