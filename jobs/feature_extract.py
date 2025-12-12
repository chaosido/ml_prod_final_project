import io
import os
import argparse
import soundfile as sf
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType

# Import the core logic from our verified package
from asr_qe.features.acoustic import AcousticFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Feature Extraction Job")
    parser.add_argument("--input", default="/opt/data/incoming", help="Path to raw audio files")
    parser.add_argument("--output", default="/opt/data/features.parquet", help="Path to output parquet")
    return parser.parse_args()

def main():
    args = parse_args()
    
    spark = SparkSession.builder \
        .appName("ASR-QE-FeatureExtraction") \
        .getOrCreate()
        
    print(f"Reading from: {args.input}")
    
    # Read binary files (content is byte[])
    raw_df = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.wav") \
        .option("recursiveFileLookup", "true") \
        .load(args.input)
        
    # Repartition to distribute work
    raw_df = raw_df.repartition(100)

    # Define the output schema
    feature_schema = StructType([
        StructField("rms_db", FloatType(), True),
        StructField("snr_db", FloatType(), True),
        StructField("duration", FloatType(), True),
    ])

    # Instantiate extractor once (per executor ideally, but inside UDF is easier for starters)
    # Note: Instantiating class inside UDF is slightly inefficient but safe. 
    # For high performance, we'd use mapPartitions or broadcast the instance.
    @F.udf(returnType=feature_schema)
    def extract_features_udf(audio_content):
        try:
            # Convert bytes to file-like object
            audio_io = io.BytesIO(audio_content)
            
            # Read audio using soundfile (safe for in-memory)
            audio, sr = sf.read(audio_io)
            
            # Extract
            extractor = AcousticFeatureExtractor()
            features = extractor.extract(audio, sr)
            
            return (
                float(features.get("rms_db", 0.0)),
                float(features.get("snr_db", 0.0)),
                float(features.get("duration", 0.0))
            )
        except Exception as e:
            # Log error (in a real system, print goes to executor stderr)
            # Returning None filters this row out later
            return None

    # Apply UDF
    features_df = raw_df.withColumn("features", extract_features_udf(F.col("content")))
    
    # Filter failures
    valid_features_df = features_df.filter(F.col("features").isNotNull())
    
    # Flatten the struct
    final_df = valid_features_df.select(
        F.col("path"),
        F.col("features.rms_db"),
        F.col("features.snr_db"),
        F.col("features.duration")
    )
    
    print(f"Writing to: {args.output}")
    final_df.write.mode("append").parquet(args.output)
    
    spark.stop()

if __name__ == "__main__":
    main()
