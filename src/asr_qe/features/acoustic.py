import numpy as np
import librosa
from typing import Dict
from asr_qe.features.interface import FeatureExtractor
from dataclasses import dataclass


@dataclass
class AcousticConfig:
    """Configuration for acoustic feature extraction."""
    frame_length: int = 2048
    hop_length: int = 512
    noise_ratio: float = 0.1
    epsilon: float = 1e-9
    silence_db: float = -100.0


class AcousticFeatureExtractor(FeatureExtractor):
    """
    Extracts basic acoustic features: RMS Energy, SNR estimation.
    """
    def __init__(self, config: AcousticConfig = None):
        self.cfg = config if config else AcousticConfig()

    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        features = {
            "rms_db": self.cfg.silence_db,
            "snr_db": 0.0,
            "duration": 0.0
        }
        
        if audio.size == 0:
            return features

        features["duration"] = float(len(audio) / sr)
        features["rms_db"] = self._compute_rms(audio)
        features["snr_db"] = self._compute_snr(audio)
        
        return features

    def _compute_rms(self, audio: np.ndarray) -> float:
        """Calculates RMS energy in dB."""
        rms = np.sqrt(np.mean(audio**2))
        return float(20 * np.log10(rms + self.cfg.epsilon))

    def _compute_snr(self, audio: np.ndarray) -> float:
        """Estimates SNR in dB using energy distribution."""
        if len(audio) < self.cfg.frame_length:
            return 0.0

        rmse_frames = librosa.feature.rms(
            y=audio, 
            frame_length=self.cfg.frame_length, 
            hop_length=self.cfg.hop_length
        )[0]
        
        rmse_frames = np.maximum(rmse_frames, self.cfg.epsilon)
        sorted_energy = np.sort(rmse_frames)
        
        noise_frames_count = max(1, int(len(sorted_energy) * self.cfg.noise_ratio))
        
        noise_profile = sorted_energy[:noise_frames_count]
        signal_profile = sorted_energy[noise_frames_count:]
        
        avg_noise_power = np.mean(noise_profile**2)
        
        if len(signal_profile) > 0:
            avg_signal_power = np.mean(signal_profile**2)
        else:
            avg_signal_power = avg_noise_power

        snr_linear_power = avg_signal_power / (avg_noise_power + self.cfg.epsilon)
        return float(10 * np.log10(snr_linear_power))
