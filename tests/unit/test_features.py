from unittest.mock import Mock

import numpy as np
import pytest

from asr_qe.features.acoustic import AcousticFeatureExtractor


@pytest.fixture
def config():
    """Returns a Mock configuration object for testing."""
    mock_config = Mock()
    mock_config.frame_length = 2048
    mock_config.hop_length = 512
    mock_config.noise_ratio = 0.1
    mock_config.epsilon = 1e-9
    mock_config.silence_db = -100.0
    return mock_config


@pytest.fixture
def extractor(config):
    return AcousticFeatureExtractor(config=config)


def test_compute_rms(extractor, config):
    """Unit tests for _compute_rms method."""
    sr = 16000

    # 1. Empty Audio
    empty_audio = np.array([])

    # 2. Silent Audio
    silent_audio = np.zeros(sr)
    silent_rms = 20 * np.log10(config.epsilon)

    # 3. Sine Wave
    # 0.5 amplitude sine. RMS = 0.5/sqrt(2) = 0.3535. 20*log10(0.3535) = -9.03 dB
    t = np.linspace(0, 1, sr)
    sine_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    cases = [
        {"name": "Empty Audio", "audio": empty_audio, "expected": config.silence_db},
        {"name": "Silent Audio", "audio": silent_audio, "expected": silent_rms},
        {
            "name": "Sine Wave",
            "audio": sine_audio,
            "expected": lambda x: -9.1 < x < -8.9,
        },
    ]

    for case in cases:
        out = extractor._compute_rms(case["audio"])
        expected = case["expected"]

        if callable(expected):
            assert expected(out), f"RMS Case {case['name']} failed: {out}"
        else:
            assert np.isclose(
                out, expected, atol=1e-5
            ), f"RMS Case {case['name']} mismatch: {out} != {expected}"


def test_compute_snr(extractor, config):
    """Unit tests for _compute_snr method."""
    sr = 16000

    # 1. Short Audio (return 0.0)
    short_audio = np.ones(100)

    # 2. Pure Tone (High SNR)

    silence_pad = np.zeros(int(0.1 * sr))
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.8, int(0.8 * sr)))
    contrasted_sine = np.concatenate([silence_pad, tone, silence_pad])

    # 3. White Noise (Low SNR)
    rng = np.random.default_rng(42)
    white_noise = rng.normal(0, 0.1, sr)

    cases = [
        {"name": "Empty Audio", "audio": np.array([]), "expected": 0.0},
        {"name": "Short Audio", "audio": short_audio, "expected": 0.0},
        {
            "name": "Contrasted Sine (High SNR)",
            "audio": contrasted_sine,
            "expected": lambda x: x > 20.0,
        },
        {
            "name": "White Noise (Low SNR)",
            "audio": white_noise,
            "expected": lambda x: -5.0 < x < 5.0,
        },
    ]

    for case in cases:
        out = extractor._compute_snr(case["audio"])
        expected = case["expected"]

        if callable(expected):
            assert expected(out), f"SNR Case {case['name']} failed: {out}"
        else:
            assert np.isclose(
                out, expected, atol=1e-5
            ), f"SNR Case {case['name']} mismatch: {out} != {expected}"


def test_extract_duration(extractor):
    """Test that extract returns duration correctly."""
    sr = 16000
    # 2 seconds of audio
    audio = np.random.randn(sr * 2)

    features = extractor.extract(audio, sr)

    assert "duration" in features, "Duration missing from features"
    assert np.isclose(
        features["duration"], 2.0, atol=0.01
    ), f"Duration mismatch: {features['duration']}"


def test_extract_all_keys(extractor):
    """Test that extract returns all expected keys."""
    sr = 16000
    audio = np.random.randn(sr)

    features = extractor.extract(audio, sr)
    expected_keys = {"rms_db", "snr_db", "duration"}

    assert expected_keys.issubset(
        features.keys()
    ), f"Missing keys: {expected_keys - features.keys()}"
