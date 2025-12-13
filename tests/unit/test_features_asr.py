from unittest.mock import Mock, patch

import pytest

from asr_qe.features.asr import ASRProcessor


@pytest.fixture
def mock_nemo_model():
    """Mocks the NeMo ASR Model."""
    model = Mock()
    return model


@pytest.fixture
def processor(mock_nemo_model):
    with patch("asr_qe.features.asr.nemo_asr") as mock_nemo_module:
        mock_nemo_module.models.ASRModel.from_pretrained.return_value = mock_nemo_model
        
        with patch("asr_qe.features.asr.TranscribeConfig") as mock_config:
            mock_config.return_value = Mock()
            
            with patch("asr_qe.features.asr.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                
                proc = ASRProcessor()
                proc.load()
                return proc


def test_process_valid_output(processor, mock_nemo_model):

    mock_hyp = Mock()
    mock_hyp.text = "this is a test"
    mock_hyp.word_confidence = [0.9, 0.9, 0.9, 0.9]  # Mean = 0.9

    mock_nemo_model.transcribe.return_value = [mock_hyp]

    result = processor.process("dummy.wav")

    assert result["transcription"] == "this is a test"
    assert result["asr_confidence"] == 0.9

    mock_nemo_model.transcribe.assert_called_once()
    args, kwargs = mock_nemo_model.transcribe.call_args
    assert args[0] == ["dummy.wav"]
    assert "override_config" in kwargs


def test_process_empty_output(processor, mock_nemo_model):
    """Verify fallback behavior when ASR returns nothing."""
    mock_nemo_model.transcribe.return_value = []

    result = processor.process("empty.wav")

    # default is 0.5 if empty so we expect this
    assert result["asr_confidence"] == 0.5
    assert "transcription" not in result


def test_process_missing_confidence(processor, mock_nemo_model):
    """Verify behavior when hypothesis exists but has no confidence scores."""
    mock_hyp = Mock()
    mock_hyp.text = "text without confidence"
    mock_hyp.word_confidence = None  # Simulate missing confidence

    mock_nemo_model.transcribe.return_value = [mock_hyp]

    result = processor.process("no_conf.wav")

    # Should fall back to 0.5
    assert result["asr_confidence"] == 0.5
    assert result["transcription"] == "text without confidence"
