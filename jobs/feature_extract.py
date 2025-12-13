import io
import json
import logging
import os
import sys
from functools import partial

import hydra
import jiwer
import soundfile as sf
from omegaconf import DictConfig
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import FloatType, StringType, StructField, StructType

# I needed to point the workers to the right python path for some reason
if os.path.exists("/app/src"):
    sys.path.insert(0, "/app/src")
if os.path.exists("/opt/airflow/src"):
    sys.path.insert(0, "/opt/airflow/src")

from asr_qe.features.acoustic import AcousticFeatureExtractor
from asr_qe.features.asr import ASRProcessor


def process_partition(partition, lookup_path):
    """
    Worker function to extract features from audio files.

    Args:
        partition: A partition of the RDD containing audio files.
        lookup_path: Path to the reference lookup JSON file.

    Yields:
        A generator of Row objects containing the extracted features.
    """
    worker_logger = logging.getLogger("WorkerLogger")

    try:
        extractor = AcousticFeatureExtractor()
        asr_processor = ASRProcessor()
        # Explicit load inside worker (singleton-like behavior inside class but we init here)  # noqa: E501
        asr_processor.load()

        # Load reference lookup
        reference_lookup = {}
        if os.path.exists(lookup_path):
            with open(lookup_path, "r") as f:
                reference_lookup = json.load(f)
        else:
            worker_logger.warning(f"Reference lookup not found at {lookup_path}")

    except Exception as e:
        worker_logger.critical(f"Failed to initialize extractors: {str(e)}")
        raise e

    for row in partition:
        try:
            # Clean path for NeMo (needs absolute file path)
            clean_path = row.path
            filename = os.path.basename(clean_path)
            if clean_path.startswith("file:"):
                clean_path = clean_path.replace("file:", "")
                filename = os.path.basename(clean_path)

            # Extract acoustic features
            # Load audio from bytes
            audio_bytes = io.BytesIO(row.content)
            audio, sr = sf.read(audio_bytes)
            acoustic_features = extractor.extract(audio, sr)

            # ASRProcessor.process returns dict with 'asr_confidence' and 'transcription'  # noqa: E501
            asr_result = asr_processor.process(clean_path)

            # Calculate WER if reference exists
            wer = -1.0
            reference_text = reference_lookup.get(filename)
            hypothesis_text = asr_result.get("transcription", "")

            if reference_text:
                try:
                    wer = jiwer.wer(reference_text, hypothesis_text)
                except Exception as w_e:
                    worker_logger.warning(
                        f"WER calculation failed for {filename}: {w_e}"
                    )
            else:
                worker_logger.warning(f"No reference text found for {filename}")

            yield Row(
                path=row.path,
                rms_db=float(acoustic_features.get("rms_db", 0.0)),
                snr_db=float(acoustic_features.get("snr_db", 0.0)),
                duration=float(acoustic_features.get("duration", 0.0)),
                asr_confidence=float(asr_result.get("asr_confidence", 0.0)),
                wer=float(wer),
            )
        except Exception as e:
            worker_logger.warning(f"Processing failed for {row.path}: {str(e)}")

            yield Row(
                path=row.path,
                rms_db=None,
                snr_db=None,
                duration=None,
                asr_confidence=None,
                wer=None,
            )


@hydra.main(version_base=None, config_path="/opt/airflow/conf", config_name="config")
def main(cfg: DictConfig):
    # Explicitly set master URL to ensure correct format (spark:// protocol required)
    # This overrides any connection-based master URL that might be missing the protocol
    spark = (
        SparkSession.builder
        .appName(cfg.spark.app_name)
        .config("spark.master", cfg.spark.master)
        .getOrCreate()
    )

    sc = spark.sparkContext

    log4j_logger = sc._jvm.org.apache.log4j
    logger = log4j_logger.LogManager.getLogger(cfg.spark.app_name)

    # check how many cores to parallelise effectively
    total_cores = int(sc.defaultParallelism)
    num_partitions = total_cores * cfg.spark.partitions_multiplier

    logger.info(f"Job Started. Input: {cfg.data.input_path}")
    logger.info(f"Architecture: {total_cores} Cores, {num_partitions} Partitions")

    raw_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.wav")
        .option("recursiveFileLookup", "true")
        .load(cfg.data.input_path)
    )

    # Repartition to ensure reasonable parallelism
    num_partitions = total_cores * cfg.spark.partitions_multiplier
    raw_df = raw_df.repartition(num_partitions)

    # Schema with all 4 features: rms_db, snr_db, duration, asr_confidence
    feature_schema = StructType(
        [
            StructField("path", StringType(), True),
            StructField("rms_db", FloatType(), True),
            StructField("snr_db", FloatType(), True),
            StructField("duration", FloatType(), True),
            StructField("asr_confidence", FloatType(), True),
            StructField("wer", FloatType(), True),
        ]
    )

    # Process partitions (feature extraction) using Parakeet
    # Pass lookup path from config
    lookup_path = cfg.data.reference_lookup_path
    # Use partial to pass lookup_path to worker function (better serialization than lambda)  # noqa: E501
    process_func = partial(process_partition, lookup_path=lookup_path)
    feature_rdd = raw_df.rdd.mapPartitions(process_func)

    # Apply schema to the RDD
    feature_df = spark.createDataFrame(feature_rdd, schema=feature_schema)

    logger.info(f"Writing dataset to {cfg.data.output_path}")

    # Write
    feature_df.write.mode("append").parquet(cfg.data.output_path)

    spark.stop()


if __name__ == "__main__":
    main()
