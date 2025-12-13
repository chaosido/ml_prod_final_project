import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

from asr_qe.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def download_voxpopuli_dutch(
    output_dir: str = "data/voxpopuli_nl",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Path:
    """
    Download VoxPopuli Dutch dataset.
    
    Args:
        output_dir: Directory to save audio files
        split: Dataset split (train/validation/test)
        max_samples: Maximum number of samples to download (None = all)
    """
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading VoxPopuli Dutch ({split} split)...")
    dataset = load_dataset("facebook/voxpopuli", "nl", split=split, streaming=True)
    
    # Check existing manifest to skip already downloaded files
    manifest_path = output_path / f"manifest_{split}.csv"
    existing_manifest_df = None
    existing_audio_ids = set()
    if manifest_path.exists():
        try:
            existing_manifest_df = pd.read_csv(manifest_path)
            existing_audio_ids = set(existing_manifest_df["audio_id"].tolist())
            logger.info(  # noqa: E501
                f"Found {len(existing_audio_ids)} existing samples"
            )
        except Exception as e:
            logger.warning(  # noqa: E501
                f"Could not read existing manifest: {e}, will download from scratch"
            )
    
    # Start with existing manifest entries or empty list
    manifest = []
    if existing_manifest_df is not None:
        manifest = existing_manifest_df.to_dict("records")
        # Limit to max_samples if specified and we already have enough
        if max_samples and len(manifest) > max_samples:
            logger.info(f"Truncating existing manifest from {len(manifest)} to {max_samples} samples")  # noqa: E501
            manifest = manifest[:max_samples]
    
    downloaded_count = 0
    skipped_count = 0
    
    logger.info(f"Downloading audio files to {audio_dir}...")
    logger.info(f"Target: {max_samples} samples (currently have {len(manifest)})")
    
    for idx, sample in enumerate(tqdm(dataset)):
        if max_samples and len(manifest) >= max_samples:
            break
        
        audio_id = sample["audio_id"]
        audio_path = audio_dir / f"{audio_id}.wav"
        
        # Skip if already downloaded
        if audio_id in existing_audio_ids or audio_path.exists():
            skipped_count += 1
            continue
        
        # Download new file
        audio_data = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        reference_text = sample["normalized_text"]
        
        sf.write(audio_path, audio_data, sampling_rate)
        downloaded_count += 1
        
        # Add to manifest
        manifest.append({
            "audio_id": audio_id,
            "audio_path": str(audio_path),
            "reference_text": reference_text,
            "speaker_id": sample["speaker_id"],
            "gender": sample["gender"],
        })
    
    # Save manifest (already contains existing + new entries)
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(manifest_path, index=False)
    
    logger.info(f"Downloaded {downloaded_count} new samples, skipped {skipped_count} existing")  # noqa: E501
    logger.info(f"Total samples in manifest: {len(manifest_df)}")
    if max_samples:
        logger.info(f"Target was {max_samples} samples")
    logger.info(f"Manifest saved to {manifest_path}")
    
    return manifest_path


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> int:
    """
    Main entry point for downloading VoxPopuli dataset.
    
    Uses Hydra config for paths, with command-line overrides available.
    # Example: python scripts/download_voxpopuli.py download.split=validation download.max_samples=100
    """
    # Get paths from config, with fallbacks
    download_cfg = cfg.get("download", {})
    output_dir = download_cfg.get("output_dir", "data/voxpopuli_nl")
    split = download_cfg.get("split", "train")
    max_samples = download_cfg.get("max_samples", None)
    
    logger.info("Starting VoxPopuli download")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split: {split}")
    logger.info(f"Max samples: {max_samples if max_samples else 'all'}")
    
    try:
        manifest_path = download_voxpopuli_dutch(
            output_dir=output_dir,
            split=split,
            max_samples=max_samples,
        )
        logger.info(f"Download complete! Manifest saved to: {manifest_path}")
        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
