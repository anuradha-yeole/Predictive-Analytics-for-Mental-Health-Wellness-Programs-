from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import BEST_MODEL_PATH

app = FastAPI(title="Mental Health Wellness Risk API")

model = joblib.load(BEST_MODEL_PATH)


class ClientFeatures(BaseModel):
    Year: Optional[int] = None
    Indian_States: Optional[str] = None
    Chronic_Diseases: Optional[str] = None
    Family_Living_Status: Optional[str] = None
    GENDER: Optional[str] = None
    Age_in_years: Optional[int] = None
    Education: Optional[str] = None
    Income: Optional[float] = None
    Psychosocial_Factors: Optional[str] = None
    Sleep_duration_Hrs: Optional[float] = None
    Frequency_of_healthcare_visits: Optional[str] = None
    Follows_a_Diet_Plan: Optional[int] = None
    Obesity_Weight_Status: Optional[int] = None
    Physical_Activity: Optional[int] = None


@app.post("/predict")
def predict(features: ClientFeatures):
    # Map Pydantic fields to actual column names used in training
    data = {
        "Year": features.Year,
        "Indian States": features.Indian_States,
        "Chronic Diseases": features.Chronic_Diseases,
        "Family Living Status": features.Family_Living_Status,
        "GENDER": features.GENDER,
        "Age in years": features.Age_in_years,
        "Education": features.Education,
        "Income": features.Income,
        "Psychosocial Factors": features.Psychosocial_Factors,
        "Sleep duration(Hrs)": features.Sleep_duration_Hrs,
        "Frequency of healthcare visits": features.Frequency_of_healthcare_visits,
        "Follows a Diet Plan": features.Follows_a_Diet_Plan,
        "Obesity / Weight Status": features.Obesity_Weight_Status,
        "Physical Activity": features.Physical_Activity,
    }

    df = pd.DataFrame([data])
    proba = float(model.predict_proba(df)[0, 1])
    label = int(model.predict(df)[0])

    return {"predicted_label": label, "risk_score": proba}
