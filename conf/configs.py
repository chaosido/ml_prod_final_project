from dataclasses import dataclass

@dataclass
class AcousticConfig:
    """
    Configuration schema for Acoustic Feature Extraction.
    """
    frame_length: int = 2048   # Samples per analysis window (~46ms @ 44.1kHz)
    hop_length: int = 512      # Samples between windows (75% overlap)
    noise_ratio: float = 0.1   # Bottom % of frames assumed to be noise
    epsilon: float = 1e-9      # Small value to prevent log(0)
    silence_db: float = -100.0 # Floor value for silent files.