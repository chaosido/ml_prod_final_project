import json
import logging
import shutil
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from asr_qe.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Replace colons with dashes to avoid filesystem issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    return filename.replace(":", "-")


def setup_verification_data(
    manifest_path: str,
    incoming_dir: str,
    lookup_path: str,
    sample_size: int = 50,
    random_state: int = 42,
) -> None:
    """Setup verification data for pipeline testing.
    
    Copies sample audio files from VoxPopuli dataset to incoming/ directory
    and creates a reference lookup file for WER calculation.
    
    Args:
        manifest_path: Path to VoxPopuli manifest CSV
        incoming_dir: Directory to copy files to (triggers pipeline)
        lookup_path: Path to save reference lookup JSON
        sample_size: Number of files to copy
        random_state: Random seed for reproducible sampling
        
    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If sample_size is invalid
        OSError: If file operations fail
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {sample_size}")
    
    logger.info(f"Reading manifest from {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    if len(df) == 0:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    
    # Sample data
    actual_sample_size = min(len(df), sample_size)
    sample_df = df.sample(n=actual_sample_size, random_state=random_state)
    
    logger.info(f"Sampled {actual_sample_size} files from {len(df)} total")
    
    # Ensure incoming directory exists and is clean
    incoming_path = Path(incoming_dir)
    if incoming_path.exists():
        logger.info(f"Cleaning {incoming_dir}")
        for item in incoming_path.glob("*"):
            if item.is_file():
                item.unlink()
    else:
        incoming_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {incoming_dir}")
    
    lookup: dict[str, str] = {}
    copied_count = 0
    skipped_count = 0
    
    logger.info(f"Copying {len(sample_df)} files to {incoming_dir}...")
    
    for _, row in sample_df.iterrows():
        original_path = row["audio_path"]
        
        # Resolve path (CSV paths are relative to project root)
        source_path = Path(original_path)
        if not source_path.is_absolute():
            source_path = source_path.resolve()
        
        if not source_path.exists():
            logger.warning(f"File not found: {source_path}. Skipping.")
            skipped_count += 1
            continue
        
        # Sanitize filename and copy
        original_filename = source_path.name
        sanitized_name = sanitize_filename(original_filename)
        dest_path = incoming_path / sanitized_name
        
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            
            # Add to lookup
            reference_text = row.get("reference_text", "")
            if reference_text:
                lookup[sanitized_name] = reference_text
        except OSError as e:
            logger.error(f"Failed to copy {source_path} to {dest_path}: {e}")
            skipped_count += 1
    
    # Save lookup
    lookup_path_obj = Path(lookup_path)
    lookup_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file if it exists (may have permission issues from Docker)
    if lookup_path_obj.exists():
        try:
            lookup_path_obj.unlink()
        except PermissionError:
            logger.warning(f"Cannot remove existing {lookup_path} (permission denied). File may be owned by Docker user.")
            logger.warning(f"Run: sudo rm {lookup_path} or sudo chown $USER:$USER {lookup_path}")
            raise OSError(f"Permission denied: Cannot write to {lookup_path}")
    
    logger.info(f"Saving reference lookup to {lookup_path}")
    with open(lookup_path, "w") as f:
        json.dump(lookup, f, indent=2)
    
    logger.info("Setup complete.")
    logger.info(f"  Files copied: {copied_count}")
    logger.info(f"  Files skipped: {skipped_count}")
    logger.info(f"  Reference entries: {len(lookup)}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> int:
    """Main entry point for setting up verification data.
    
    Uses Hydra config for paths, with command-line overrides available.
    
    Example:
        python scripts/setup_verification_data.py \
            setup_verification.sample_size=100 \
            setup_verification.manifest_path=data/voxpopuli_nl/manifest_train.csv
    """
    # Get config with fallbacks
    setup_cfg = cfg.get("setup_verification", {})
    
    # Get paths from config
    manifest_path = setup_cfg.get(
        "manifest_path",
        cfg.get("download", {}).get("output_dir", "data/voxpopuli_nl") + "/manifest_train.csv",
    )
    incoming_dir = setup_cfg.get(
        "incoming_dir",
        cfg.get("pipeline", {}).get("incoming_data", "data/incoming"),
    )
    lookup_path = setup_cfg.get(
        "lookup_path",
        cfg.get("data", {}).get("reference_lookup_path", "data/reference_lookup.json"),
    )
    sample_size = setup_cfg.get("sample_size", 50)
    random_state = setup_cfg.get("random_state", 42)
    
    logger.info("Starting verification data setup")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Incoming directory: {incoming_dir}")
    logger.info(f"Lookup path: {lookup_path}")
    logger.info(f"Sample size: {sample_size}")
    
    try:
        setup_verification_data(
            manifest_path=manifest_path,
            incoming_dir=incoming_dir,
            lookup_path=lookup_path,
            sample_size=sample_size,
            random_state=random_state,
        )
        logger.info("Verification data setup complete!")
        return 0
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

