
import logging
from typing import Dict, Optional, Tuple
from functools import lru_cache
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig

logger = logging.getLogger(__name__)

class ASRProcessor:
    """
    Singleton Wrapper for the heavy ASR model (Parakeet).
    Handles loading and inference-time feature extraction (confidence).
    """

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
        self.model_name = model_name
        self._model = None
        self._transcribe_config = None
        
    def load(self):
        """Load the model into memory (GPU if available)."""
        if self._model is not None:
            return

        logger.info(f"Loading ASR Processor model: {self.model_name}")
        try:
            self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
            
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("ASR Processor using GPU")
            else:
                logger.info("ASR Processor using CPU")

            # we want the word level confidence from here 
            self._model.change_decoding_strategy(
                decoding_cfg={
                    "preserve_alignments": True,
                    "compute_confidence": True,
                    "confidence_cfg": {
                        "preserve_token_confidence": True,
                        "preserve_word_confidence": True,
                        "exclude_blank": True,
                        "aggregation": "mean",
                        "method_cfg": {"name": "max_prob"},
                    },
                }
            )
            
            # Helper config for transcription
            self._transcribe_config = TranscribeConfig(return_hypotheses=True)
            logger.info("ASR Processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise RuntimeError(f"Could not load ASR model: {e}")

    def process(self, audio_path: str) -> Dict[str, float]:
        """
        Run ASR on a file path and return features.
        NOTE: NeMo transcribe expects PATHS, not raw bytes usually.
        """
        if self._model is None:
            raise RuntimeError("ASR model not loaded. Call load() first.")

        # NeMo transcribe usually takes a list of paths
        hypotheses = self._model.transcribe([audio_path], override_config=self._transcribe_config)
        
        if not hypotheses:
            return {"asr_confidence": 0.5}

        hypothesis = hypotheses[0]
        
        # Logic extracted from generate_ground_truth.py
        word_confidence = 0.5
        
        if hasattr(hypothesis, 'word_confidence') and hypothesis.word_confidence is not None:
            try:
                # Handle potentially different return types (list, tensor)
                confs = hypothesis.word_confidence
                # Convert to flat list of floats
                if hasattr(confs, 'tolist'):
                    confs = confs.tolist()
                
                valid_confs = [float(c) for c in confs if c is not None and not np.isnan(c)]
                
                if valid_confs:
                    word_confidence = float(np.mean(valid_confs))
            except Exception as e:
                logger.warning(f"Error extracting confidence: {e}")

        return {"asr_confidence": word_confidence, "transcription": hypothesis.text}

@lru_cache(maxsize=1)
def get_asr_processor() -> ASRProcessor:
    """Singleton accessor."""
    return ASRProcessor()
