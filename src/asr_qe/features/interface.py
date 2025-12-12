from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class FeatureExtractor(ABC):
    """
    Abstract base class for audio feature extraction strategies.
    """
    
    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract features from a loaded audio array.
        
        Args:
            audio: Audio time-series (numpy array).
            sr: Sampling rate.
            
        Returns:
            Dictionary mapping feature names to values.
        """
        pass
