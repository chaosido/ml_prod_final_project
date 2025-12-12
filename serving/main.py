import io
import os
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter


from contextlib import asynccontextmanager
import asr_qe
from asr_qe.models.loader import get_model_loader
from asr_qe.config import Settings, TrainingConfig
from asr_qe.features.acoustic import AcousticFeatureExtractor
from asr_qe.features.asr import get_asr_processor

settings = Settings()

# to keep track of the SNR of the input audio
SNR_DISTRIBUTION = Histogram(
    "input_audio_snr_db", 
    "Drift: Distribution of input Audio SNR", 
    buckets=(-10, 0, 10, 20, 30, 40, 50, float("inf"))
)

PREDICTED_WER_DISTRIBUTION = Histogram(
    "predicted_wer", 
    "Model Health: Distribution of predicted WER",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
)


LOW_QUALITY_TRANSCRIPT_COUNT = Counter(
    "low_quality_transcripts_total",
    f"Business: Total number of transcripts flagged for review (WER > {settings.review_threshold})"
)

@asynccontextmanager
async def lifespan(app: FastAPI):    
    # Initialize model loader
    loader = get_model_loader()
    try:
        loader.load(settings.model_path)
        print(f"Loaded QE model from {settings.model_path}")
    except Exception as e:
        print(f"Error loading QE model: {e}")

    asr = get_asr_processor()
    try:
        print("Loading ASR model... (this may take a minute)")
        asr.load()
        print("ASR model loaded.")
    except Exception as e:
        print(f"Error loading ASR model: {e}")


    yield

app = FastAPI(title="ASR QE API", version=asr_qe.__version__, lifespan=lifespan)

Instrumentator().instrument(app).expose(app)

@app.get("/")
async def health_check():
    return {"status": "ok", "version": asr_qe.__version__}

@app.post("/predict")
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

            #for prometheus drift detection
            if "snr_db" in acoustic_features:
                SNR_DISTRIBUTION.observe(acoustic_features["snr_db"])

            asr_processor = get_asr_processor()
            asr_features = asr_processor.process(tmp_path)
            hypothesis = asr_features.get("transcription", "")
            confidence = asr_features.get("asr_confidence")
            all_features = {**acoustic_features, "asr_confidence": confidence}
            
            try:
                columns = TrainingConfig.feature_columns 
                # ("rms_db", "snr_db", "duration", "asr_confidence")
                feature_vector = [all_features[col] for col in columns]
                features_array = np.array([feature_vector]) 
            except KeyError as e:
                    raise HTTPException(status_code=500, detail=f"Feature extraction failed: missing {e}")

            loader = get_model_loader()
            prediction = loader.predict(features_array)
            
            predicted_wer = float(prediction[0])

            # for if predictions change over time (sign of bad audio quality)
            PREDICTED_WER_DISTRIBUTION.observe(predicted_wer)
            if predicted_wer > settings.review_threshold:
                LOW_QUALITY_TRANSCRIPT_COUNT.inc()
            
            return {
                "predicted_wer": predicted_wer,
                "review_recommended": predicted_wer > settings.review_threshold,
                "transcript": hypothesis,
            }

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)