import uvicorn
import pandas as pd
import json
import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib


class InputFeatures(BaseModel):
    avg_isc_test_a: float
    avg_geff_test_w_m2: float
    avg_moduletemp1_c: float
    avg_temp_refcell_c: float
    avg_wind_speed_m_s: float
    avg_humidity_pct: float


app = FastAPI(title="Solar Power Prediction API")
MODEL_PATH = "model.joblib" # (تأكد من أن هذا هو اسم الملف الذي حفظته)
FEATURES_PATH = "features.json" # (هذا يبقى كما هو)

try:
    model = joblib.load(MODEL_PATH)
    
    with open(FEATURES_PATH, 'r') as f:
        features_list = json.load(f)
    
    print("✅ Model (joblib) and features loaded successfully.")

except FileNotFoundError:
    print(f"❌ ERROR: '{MODEL_PATH}' or '{FEATURES_PATH}' not found.")
    model = None
    features_list = None


app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    
    """
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: index.html not found.</h1>", status_code=404)



@app.post("/api/predict")
async def predict_power(data: InputFeatures):
    """
    
    """
    if model is None or features_list is None:
        return {"error": "Model not loaded. Please check server logs."}, 500

    try:
        
        input_df = pd.DataFrame([data.dict()], columns=features_list)
        
        
        prediction = model.predict(input_df)
        
        
        predicted_value = float(prediction[0])
        
        
        if predicted_value < 0:
            predicted_value = 0.0

        
        return {
            "predicted_power": predicted_value,
            "features_used": features_list
        }
    
    except Exception as e:
        return {"error": str(e)}, 400


if __name__ == "__main__":
   
    uvicorn.run(app, host="127.0.0.1", port=8000)