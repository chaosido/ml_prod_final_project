# Import cuda-python early to avoid NeMo warnings
# NeMo checks for cuda-python at import time for CUDA graph optimization
try:
    import cuda  # noqa: F401  # cuda-python package imports as 'cuda'
except ImportError:
    pass  # Optional dependency, only affects performance

import logging
import os
import sys
from pathlib import Path

import hydra
import jiwer
import nemo.collections.asr as nemo_asr
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from tqdm import tqdm

from asr_qe.config import Config, register_configs
from asr_qe.features.acoustic import AcousticFeatureExtractor
from asr_qe.utils.logging import setup_logging

setup_logging()

# Register configs for Hydra
register_configs()
logger = logging.getLogger(__name__)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    try:
        return jiwer.wer(reference, hypothesis)
    except Exception:
        return 1.0  # Max error if computation fails


def generate_ground_truth(
    manifest_path: str,
    output_path: str,
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
    batch_size: int = 16,
    enable_confidence: bool = True,
) -> Path:
    """
    Generate WER labels and features using Parakeet with confidence scores.

    Args:
        manifest_path: Path to manifest CSV
        output_path: Output parquet path
        model_name: Parakeet model to use
        batch_size: Batch size for inference
        enable_confidence: Whether to compute ASR confidence scores

    Returns:
        Path to the output parquet file
    """

    logger.info(f"Loading manifest from {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    # Pre-load audio files to get durations and avoid double-loading
    logger.info("Pre-loading audio files to check durations...")
    audio_data_cache = {}
    durations = []
    for _, row in manifest.iterrows():
        audio_path = row["audio_path"]
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        audio_data_cache[audio_path] = (audio, sr)
        durations.append(duration)

    manifest["duration"] = durations
    avg_duration = np.mean(durations)
    max_duration = np.max(durations)
    total_duration = np.sum(durations)

    logger.info("Audio statistics:")
    logger.info(f"  Total samples: {len(manifest)}")
    logger.info(f"  Average duration: {avg_duration:.2f}s")
    logger.info(f"  Max duration: {max_duration:.2f}s ({max_duration/60:.2f} minutes)")
    logger.info(f"  Total audio time: {total_duration/60:.2f} minutes")

    logger.info(f"Loading Parakeet model: {model_name}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    # Enable GPU if available
    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
        logger.info("Using GPU for inference")

    # Configure model to compute confidence scores (optional, adds overhead)
    # This must be done via change_decoding_strategy, not TranscribeConfig
    confidence_enabled = False
    logger.info(f"enable_confidence parameter: {enable_confidence}")
    if enable_confidence:
        try:
            # NeMo requires preserve_alignments=True for word-level confidence
            asr_model.change_decoding_strategy(
                decoding_cfg={
                    "preserve_alignments": True,  # Required for word-level confidence
                    "compute_confidence": True,  # Enable confidence score computation
                    "confidence_cfg": {
                        "preserve_frame_confidence": False,  # Too detailed, not needed
                        "preserve_token_confidence": False,  # Use word-level instead
                        "preserve_word_confidence": True,  # Word-level confidence
                        "exclude_blank": True,  # Exclude blank tokens
                        "aggregation": "mean",  # Aggregate token confidences to words  # noqa: E501
                        "method_cfg": {
                            "name": "max_prob",  # Use maximum token probability  # noqa: E501
                        },
                    },
                }
            )
            confidence_enabled = True
            logger.info(
                "Confidence score computation enabled (may slow down inference)"  # noqa: E501
            )
        except Exception as e:
            logger.warning(
                f"Could not enable confidence scores: {e}. Continuing without confidence scores."  # noqa: E501
            )
    else:
        logger.info("Confidence score computation disabled for faster inference")

    # Configure transcription to return hypotheses (only needed if confidence is enabled)  # noqa: E501
    transcribe_config = TranscribeConfig(
        return_hypotheses=confidence_enabled,  # Only return hypotheses if we need confidence  # noqa: E501
    )

    feature_extractor = AcousticFeatureExtractor()
    results = []

    logger.info(f"Processing {len(manifest)} audio files in batches of {batch_size}...")

    # Process in batches
    for i in tqdm(range(0, len(manifest), batch_size), desc="Transcribing batches"):
        batch_manifest = manifest.iloc[i : i + batch_size]
        batch_paths = batch_manifest["audio_path"].tolist()

        # Transcribe batch with Parakeet
        if confidence_enabled:
            hypotheses = asr_model.transcribe(  # noqa: E501
                batch_paths, override_config=transcribe_config
            )
            # Debug: Log hypothesis structure for first sample of first batch
            if i == 0 and len(hypotheses) > 0:
                logger.info(f"First hypothesis type: {type(hypotheses[0])}")
                logger.info(
                    f"First hypothesis attributes: {[attr for attr in dir(hypotheses[0]) if not attr.startswith('_')]}"  # noqa: E501
                )
                if hasattr(hypotheses[0], "word_confidence"):
                    wc = hypotheses[0].word_confidence
                    logger.info(
                        f"word_confidence type: {type(wc)}, value: {wc}, length: {len(wc) if hasattr(wc, '__len__') else 'N/A'}"  # noqa: E501
                    )
                else:
                    logger.warning("Hypothesis does not have word_confidence attribute!")  # noqa: E501
                if hasattr(hypotheses[0], "token_confidence"):
                    tc = hypotheses[0].token_confidence
                    logger.info(
                        f"token_confidence type: {type(tc)}, value: {tc}, length: {len(tc) if hasattr(tc, '__len__') else 'N/A'}"  # noqa: E501
                    )
                if hasattr(hypotheses[0], "score"):
                    logger.info(f"score: {hypotheses[0].score}")
                if hasattr(hypotheses[0], "alignments"):
                    logger.info(
                        f"alignments type: {type(hypotheses[0].alignments)}, is None: {hypotheses[0].alignments is None}"  # noqa: E501
                    )
        else:
            # Fast path: just get text transcriptions
            transcriptions = asr_model.transcribe(batch_paths)

        # Process each sample in batch
        for idx, (_, row) in enumerate(batch_manifest.iterrows()):
            audio_path = row["audio_path"]
            reference_text = row["reference_text"]

            if confidence_enabled:
                hypothesis = hypotheses[idx]
                hypothesis_text = hypothesis.text
            else:
                # Simple text transcription - extract text from Hypothesis object or use string directly  # noqa: E501
                trans = transcriptions[idx]
                if isinstance(trans, str):
                    hypothesis_text = trans
                elif hasattr(trans, "text"):
                    hypothesis_text = trans.text
                else:
                    hypothesis_text = str(trans)
                hypothesis = None

            # Get audio from cache (already loaded)
            audio, sr = audio_data_cache[audio_path]

            # Extract confidence scores from hypothesis (only if enabled)
            word_confidence = None

            if confidence_enabled and hypothesis is not None:
                # Try to get word-level confidence
                if (
                    hasattr(hypothesis, "word_confidence")
                    and hypothesis.word_confidence is not None
                ):
                    try:
                        # Convert to list/array and check if non-empty
                        conf_list = (
                            list(hypothesis.word_confidence)
                            if hasattr(hypothesis.word_confidence, "__iter__")
                            else [hypothesis.word_confidence]
                        )
                        if len(conf_list) > 0:
                            # Extract values from PyTorch tensors or use directly
                            valid_confs = []
                            for c in conf_list:
                                if c is None:
                                    continue
                                # Handle PyTorch tensors
                                if hasattr(c, "item"):
                                    try:
                                        val = float(c.item())
                                        if not np.isnan(val):
                                            valid_confs.append(val)
                                    except Exception:
                                        pass
                                # Handle numpy arrays/scalars
                                elif isinstance(c, (int, float, np.number)):
                                    val = float(c)
                                    if not np.isnan(val):
                                        valid_confs.append(val)
                                # Try direct conversion
                                else:
                                    try:
                                        val = float(c)
                                        if not np.isnan(val):
                                            valid_confs.append(val)
                                    except Exception:
                                        pass

                            if valid_confs:
                                word_confidence = float(np.mean(valid_confs))
                            else:
                                logger.debug(
                                    f"All confidence values were invalid/empty for sample {row['audio_id']}"  # noqa: E501
                                )
                    except Exception as e:
                        logger.warning(f"Error extracting word_confidence: {e}")
                        logger.warning(f"  word_confidence value: {hypothesis.word_confidence}")  # noqa: E501

                # Fallback to token confidence
                if (  # noqa: E501
                    word_confidence is None
                    and hasattr(hypothesis, "token_confidence")
                    and hypothesis.token_confidence is not None
                ):
                    try:
                        if (  # noqa: E501
                            isinstance(hypothesis.token_confidence, (list, tuple))
                            and len(hypothesis.token_confidence) > 0
                        ):
                            token_confidences = []
                            for t in hypothesis.token_confidence:
                                # Handle PyTorch tensors
                                if hasattr(t, "item"):
                                    try:
                                        val = float(t.item())
                                        if not np.isnan(val):
                                            token_confidences.append(val)
                                    except Exception:
                                        pass
                                elif isinstance(t, (int, float, np.number)):
                                    val = float(t)
                                    if not np.isnan(val):
                                        token_confidences.append(val)

                            if token_confidences:
                                word_confidence = float(np.mean(token_confidences))
                    except Exception as e:
                        logger.warning(f"Error extracting token_confidence: {e}")

                # Fallback to score if available
                if word_confidence is None and hasattr(hypothesis, "score"):
                    try:
                        if hypothesis.score is not None:
                            word_confidence = float(hypothesis.score)
                    except Exception as e:
                        logger.warning(f"Error extracting score: {e}")

            # Final fallback: use 0.5 as default if no confidence available
            # This ensures the column exists even if confidence extraction fails or is disabled  # noqa: E501
            if word_confidence is None:
                word_confidence = 0.5  # Default neutral confidence
                if not confidence_enabled and idx == 0:  # Log only once
                    logger.info(
                        "Confidence scores disabled, using default 0.5 for all samples"
                    )

            # Compute WER
            wer = compute_wer(reference_text, hypothesis_text)

            # Extract acoustic features
            acoustic_features = feature_extractor.extract(audio, sr)

            # Combine
            result = {
                "audio_id": row["audio_id"],
                "audio_path": audio_path,
                "reference_text": reference_text,
                "hypothesis_text": hypothesis_text,
                "wer": wer,
                "asr_confidence": word_confidence,  # Model's confidence score
                "duration": row["duration"],
                **acoustic_features,
            }
            results.append(result)

    # Save to parquet
    df = pd.DataFrame(results)
    df.to_parquet(output_path)

    logger.info(f"Saved {len(df)} samples to {output_path}")
    logger.info("WER statistics:")
    logger.info(f"  Mean: {df['wer'].mean():.4f}")
    logger.info(f"  Median: {df['wer'].median():.4f}")
    logger.info(f"  Min: {df['wer'].min():.4f}")
    logger.info(f"  Max: {df['wer'].max():.4f}")

    if "asr_confidence" in df.columns:
        valid_confidence = df["asr_confidence"].dropna()
        if len(valid_confidence) > 0:
            logger.info("ASR Confidence statistics:")
            logger.info(f"  Mean: {valid_confidence.mean():.4f}")
            logger.info(f"  Median: {valid_confidence.median():.4f}")
            logger.info(f"  Min: {valid_confidence.min():.4f}")
            logger.info(f"  Max: {valid_confidence.max():.4f}")
            logger.info(f"  Samples with confidence: {len(valid_confidence)}/{len(df)}")

    return output_path


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> int:
    """
    Main entry point for generating ground truth WER labels.

    Uses Hydra config for paths, with command-line overrides available.
    # Example: python scripts/generate_ground_truth.py ground_truth.manifest_path=data/manifest.csv ground_truth.output_path=data/ground_truth.parquet
    """
    # Set NeMo cache directory from config
    nemo_cache_dir = Path(cfg.nemo.cache_dir)
    if nemo_cache_dir.exists() or nemo_cache_dir.parent.exists():
        os.environ["NEMO_CACHE_DIR"] = str(nemo_cache_dir)
        logger.info(f"Using NeMo cache directory: {nemo_cache_dir}")
    else:
        logger.warning(
            f"NeMo cache directory {nemo_cache_dir} does not exist. "
            "Using default NeMo cache location."
        )
    
    # Get paths from config
    ground_truth_cfg = cfg.ground_truth
    manifest_path = ground_truth_cfg.manifest_path
    output_path = ground_truth_cfg.output_path
    model_name = ground_truth_cfg.model_name
    batch_size = ground_truth_cfg.batch_size
    enable_confidence = ground_truth_cfg.enable_confidence

    if not manifest_path:
        logger.error("manifest_path is required. Set ground_truth.manifest_path in config or via CLI")  # noqa: E501
        return 1

    if not output_path:
        logger.error(
            "output_path is required. Set ground_truth.output_path in config or via CLI"
        )
        return 1

    logger.info("Starting ground truth generation")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Confidence enabled: {enable_confidence}")

    try:
        result_path = generate_ground_truth(
            manifest_path=manifest_path,
            output_path=output_path,
            model_name=model_name,
            batch_size=batch_size,
            enable_confidence=enable_confidence,
        )
        logger.info(f"Ground truth generation complete! Saved to: {result_path}")
        return 0
    except Exception as e:
        logger.error(f"Ground truth generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
