import os
import soundfile as sf
from pathlib import Path
from typing import Union


def validate_audio_file(
    file_path: Union[str, Path], min_size_kb: int = 1, min_duration_sec: float = 0.5
) -> bool:
    """
    Validates an audio file based on size and duration.

    Args:
        file_path: Path to the audio file.
        min_size_kb: Minimum file size in KB.
        min_duration_sec: Minimum audio duration in seconds.

    Returns:
        True if valid, raises appropriate exceptions otherwise.
    """
    path = Path(file_path)

    # Check existence
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Check size
    size_kb = path.stat().st_size / 1024
    if size_kb < min_size_kb:
        raise ValueError(f"File too small: {size_kb:.2f}KB (Min: {min_size_kb}KB)")

    # Check format and duration (requires reading header)
    try:
        info = sf.info(str(path))
        if info.duration < min_duration_sec:
            raise ValueError(
                f"Duration too short: {info.duration:.2f}s (Min: {min_duration_sec}s)"
            )
    except Exception as e:
        raise ValueError(f"Invalid audio format or header: {e}")

    return True
