from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Restimator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "sleep_model.pkl"

# Load model once
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    model = None

class SleepInput(BaseModel):
    bedtime: str
    screen_time: int
    stress: int
    weather: str
    alarms: int
    desired_wake: str

class SleepPrediction(BaseModel):
    sleep_duration: float
    predicted_wake_time: str
    oversleep_warning: bool
    message: str

def time_to_minutes(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m

def minutes_to_time(m: int) -> str:
    h = (m // 60) % 24
    mins = m % 60
    return f"{h:02d}:{mins:02d}"

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=SleepPrediction)
def predict(data: SleepInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    features = pd.DataFrame([{
        "bedtime": time_to_minutes(data.bedtime),
        "screen_time": data.screen_time,
        "stress": data.stress,
        "weather": data.weather,
        "alarms": data.alarms
    }])

    try:
        pred_hours = float(model.predict(features)[0])
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Model prediction failed")

    # Calibrate (same logic as local script)
    pred_hours = max(4.0, pred_hours * 0.80 - 1.0)

    pred_wake = time_to_minutes(data.bedtime) + int(pred_hours * 60)
    if pred_wake >= 1440:
        pred_wake -= 1440

    pred_wake_time = minutes_to_time(pred_wake)
    desired_wake_minutes = time_to_minutes(data.desired_wake)
    oversleep = pred_wake > desired_wake_minutes
    message = "⚠️ Oversleep likely!" if oversleep else "✅ You will wake on time!"

    return SleepPrediction(
        sleep_duration=round(pred_hours, 2),
        predicted_wake_time=pred_wake_time,
        oversleep_warning=oversleep,
        message=message
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)