from fastapi import FastAPI
import asr_qe

app = FastAPI(title="ASR QE API", version=asr_qe.__version__)

@app.get("/")
async def health_check():
    return {"status": "ok", "version": asr_qe.__version__}

@app.get("/predict")
async def predict_placeholder():
    return {"message": "Prediction endpoint placeholder"}
