import logging
import os
import tempfile
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Histogram, REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator

import asr_qe
from asr_qe.config import TrainingConfig
from asr_qe.features.acoustic import AcousticFeatureExtractor
from asr_qe.features.asr import get_asr_processor
from asr_qe.models.loader import get_model_loader
from asr_qe.utils.logging import setup_logging
from serving.schemas import PredictionResponse

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Environment-based configuration
MODEL_PATH = os.getenv("ASR_QE_MODEL_PATH", "models/model.joblib")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.4"))

# Define metrics helpers
def get_or_create_histogram(name, documentation, buckets=Histogram.DEFAULT_BUCKETS):
    try:
        return Histogram(name, documentation, buckets=buckets)
    except ValueError:
        return REGISTRY._names_to_collectors[name]

def get_or_create_counter(name, documentation):
    try:
        return Counter(name, documentation)
    except ValueError:
        return REGISTRY._names_to_collectors[name]

# to keep track of the SNR of the input audio
SNR_DISTRIBUTION = get_or_create_histogram(
    "input_audio_snr_db",
    "Drift: Distribution of input Audio SNR",
    buckets=(-10, 0, 10, 20, 30, 40, 50, float("inf")),
)

PREDICTED_WER_DISTRIBUTION = get_or_create_histogram(
    "predicted_wer",
    "Model Health: Distribution of predicted WER",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
)


LOW_QUALITY_TRANSCRIPT_COUNT = get_or_create_counter(
    "low_quality_transcripts_total",
    f"Business: Total number of transcripts flagged for review (WER > {REVIEW_THRESHOLD})",  # noqa: E501
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model loader
    logger.info(f"Loading QE model from {MODEL_PATH}")
    loader = get_model_loader()
    loader.load(MODEL_PATH)
    logger.info("QE model loaded successfully")

    asr = get_asr_processor()
    logger.info("Loading ASR model... (this may take a minute)")
    asr.load()
    logger.info("ASR model loaded successfully")

    yield


app = FastAPI(title="ASR QE API", version=asr_qe.__version__, lifespan=lifespan)

Instrumentator().instrument(app).expose(app)


@app.get("/")
async def health_check():
    return {"status": "ok", "version": asr_qe.__version__}


@app.post("/predict", response_model=PredictionResponse)
async def predict(audio_file: UploadFile = File(...)):
    # our asr model requires a path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        content = await audio_file.read()
        tmp.write(content)
        tmp.flush()

        try:
            audio, sr = sf.read(tmp_path)

            acoustic_extractor = AcousticFeatureExtractor()
            acoustic_features = acoustic_extractor.extract(audio, sr)

            # for prometheus drift detection
            if "snr_db" in acoustic_features:
                SNR_DISTRIBUTION.observe(acoustic_features["snr_db"])

            asr_processor = get_asr_processor()
            asr_features = asr_processor.process(tmp_path)
            hypothesis = asr_features.get("transcription", "")
            confidence = asr_features.get("asr_confidence")
            all_features = {**acoustic_features, "asr_confidence": confidence}

            try:
                columns = TrainingConfig().feature_columns
                # ("rms_db", "snr_db", "duration", "asr_confidence")
                feature_vector = [all_features[col] for col in columns]
                features_array = np.array([feature_vector])
            except KeyError as e:
                raise HTTPException(
                    status_code=500, detail=f"Feature extraction failed: missing {e}"
                )

            loader = get_model_loader()
            prediction = loader.predict(features_array)

            predicted_wer = float(prediction[0])

            # for if predictions change over time (sign of bad audio quality)
            PREDICTED_WER_DISTRIBUTION.observe(predicted_wer)
            if predicted_wer > REVIEW_THRESHOLD:
                LOW_QUALITY_TRANSCRIPT_COUNT.inc()

            return {
                "predicted_wer": predicted_wer,
                "review_recommended": predicted_wer > REVIEW_THRESHOLD,
                "transcript": hypothesis,
            }

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
