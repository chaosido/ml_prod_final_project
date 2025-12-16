import numpy as np
import pytest
import soundfile as sf

from asr_qe.validation import validate_audio_file


class TestAudioValidation:
    """Tests for validate_audio_file function."""

    def test_valid_file(self, tmp_path):
        """Verify valid audio file passes validation."""
        path = tmp_path / "valid.wav"
        audio = np.random.uniform(-0.1, 0.1, 16000)  # 1 sec @ 16kHz
        sf.write(str(path), audio, 16000)

        assert validate_audio_file(str(path), min_size_kb=1, min_duration_sec=0.5)

    @pytest.mark.parametrize(
        "setup_file,error_type,error_match,description",
        [
            # (setup function name, exception type, regex match, description)
            ("short_audio", ValueError, "Duration too short", "audio too short"),
            ("text_file", ValueError, "Invalid audio format", "not a valid audio file"),
            ("missing_file", FileNotFoundError, None, "file does not exist"),
        ],
    )
    def test_invalid_files(
        self, tmp_path, setup_file, error_type, error_match, description
    ):
        """Verify invalid files raise appropriate errors."""
        # Setup file based on test case
        if setup_file == "short_audio":
            path = tmp_path / "short.wav"
            audio = np.zeros(int(0.1 * 16000))  # 0.1 sec
            sf.write(str(path), audio, 16000)
        elif setup_file == "text_file":
            path = tmp_path / "not_wav.txt"
            path.write_text("This is not a wav file. " * 100)
        elif setup_file == "missing_file":
            path = tmp_path / "non_existent.wav"
        else:
            raise ValueError(f"Unknown setup: {setup_file}")

        # Assert correct exception
        if error_match:
            with pytest.raises(error_type, match=error_match):
                validate_audio_file(str(path), min_size_kb=1, min_duration_sec=0.5)
        else:
            with pytest.raises(error_type):
                validate_audio_file(str(path), min_size_kb=1, min_duration_sec=0.5)
