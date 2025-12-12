import pytest
import soundfile as sf
import numpy as np
from asr_qe.validation import validate_audio_file

def test_validate_valid_file(tmp_path):
    # Create a valid wav file (1 sec, 16kHz)
    path = tmp_path / "valid.wav"
    sr = 16000
    audio = np.random.uniform(-0.1, 0.1, sr)
    sf.write(str(path), audio, sr)
    
    assert validate_audio_file(str(path), min_size_kb=1, min_duration_sec=0.5)

def test_validate_too_short(tmp_path):
    # Create short file (0.1 sec)
    path = tmp_path / "short.wav"
    sr = 16000
    audio = np.zeros(int(0.1 * sr))
    sf.write(str(path), audio, sr)
    
    with pytest.raises(ValueError, match="Duration too short"):
        validate_audio_file(str(path), min_duration_sec=0.5)

def test_validate_not_found():
    with pytest.raises(FileNotFoundError):
        validate_audio_file("non_existent.wav")
