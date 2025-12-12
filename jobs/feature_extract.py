import io
import logging
import soundfile as sf
import hydra
from omegaconf import DictConfig

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, FloatType, StringType

from asr_qe.features.acoustic import AcousticFeatureExtractor

def process_partition(partition):
    '''
    Worker function to extract features from audio files.
    
    Args:
        partition: A partition of the RDD containing audio files.
    
    Yields:
        A generator of Row objects containing the extracted features.
    '''
    worker_logger = logging.getLogger("WorkerLogger")
    

    try:
        extractor = AcousticFeatureExtractor()
    except Exception as e:
        worker_logger.critical(f"Failed to initialize feature extractor: {str(e)}")
        raise e
    
    for row in partition:
        try:
            audio_io = io.BytesIO(row.content)
            audio, sr = sf.read(audio_io)
            
            features = extractor.extract(audio, sr)
            
            yield Row(
                path=row.path, 
                rms_db=float(features.get("rms_db", 0.0)), 
                snr_db=float(features.get("snr_db", 0.0)), 
                duration=float(features.get("duration", 0.0)),
            )
        except Exception as e:
            worker_logger.warning(f"Processing failed for {row.path}: {str(e)}")
            
            yield Row(
                path=row.path, 
                rms_db=None, snr_db=None, duration=None,
            )

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    
    spark = SparkSession.builder \
        .appName(cfg.spark.app_name) \
        .getOrCreate()

    sc = spark.sparkContext
    
    log4j_logger = sc._jvm.org.apache.log4j
    logger = log4j_logger.LogManager.getLogger(cfg.spark.app_name)
    
    # check how many cores to parallelise effectively
    total_cores = int(sc.defaultParallelism)
    num_partitions = total_cores * cfg.spark.partitions_multiplier
    
    logger.info(f"Job Started. Input: {cfg.data.input_path}")
    logger.info(f"Architecture: {total_cores} Cores, {num_partitions} Partitions")
    
    raw_df = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.wav") \
        .option("recursiveFileLookup", "true") \
        .load(cfg.data.input_path)
        
    raw_df = raw_df.repartition(num_partitions)
    
    feature_schema = StructType([
        StructField("path", StringType(), True),
        StructField("rms_db", FloatType(), True),
        StructField("snr_db", FloatType(), True),
        StructField("duration", FloatType(), True),
    ])

    # Transform to rdd which allows for efficient parallelization in spark 
    feature_rdd = raw_df.rdd.mapPartitions(process_partition)
    feature_df = spark.createDataFrame(feature_rdd, schema=feature_schema)

    logger.info(f"Writing dataset to {cfg.data.output_path}")
    
    # Write
    feature_df.write.mode("append").parquet(cfg.data.output_path)

    spark.stop()

if __name__ == "__main__":
    main()