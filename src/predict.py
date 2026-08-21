import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


#--Load trained artifacts------
encoder = joblib.load(MODEL_DIR / "final_encoder.pkl")
scaler = joblib.load(MODEL_DIR / "final_scaler.pkl")
medians = joblib.load(MODEL_DIR / "final_medians.pkl")
model = joblib.load(MODEL_DIR / "final_xgb_model.pkl")
#joblib.load('../models/final_logreg_model.pkl')

THRESHOLD = 0.23


#--Pydantic input schema------
class LoanApplication(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True
    )
    
    loan_limit: Literal["cf", "ncf"]
    gender: Literal["male", "female", "joint", "sex_not_available"]
    approv_in_adv: Literal["pre", "nopre"]
    loan_type: Literal["type1", "type2", "type3"]
    loan_purpose: Literal["p1", "p2", "p3", "p4"]
    credit_worthiness: Literal["l1", "l2"]
    open_credit: Literal["nopc", "opc"]
    business_or_commercial: Literal["nob/c", "b/c"]
    loan_amount: float = Field(ge=16500, le=3576500)
    term: float = Field(ge=12, le=360)
    neg_ammortization: Literal["not_neg", "neg_amm"]
    interest_only: Literal["not_int", "int_only"]
    lump_sum_payment: Literal["not_lpsm", "lpsm"]
    property_value: Optional[float] = Field(default=None, ge=8000, le=16508000)
    construction_type: Literal["sb", "mh"]
    occupancy_type: Literal["pr", "ir", "sr"]
    secured_by: Literal["home", "land"]
    total_units: Literal["1u", "2u", "3u", "4u"]
    income: Optional[float] = Field(default=None, ge=0, le=578580)
    credit_score: int = Field(ge=500, le=900)
    co_applicant_credit_type: Literal["cib", "exp"] = Field(alias="co-applicant_credit_type")
    age: Literal["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
    submission_of_application: Literal["to_inst", "not_inst"]
    ltv: Optional[float] = Field(default=None, ge=0.96, le=7831.25)
    region: Literal["north", "south", "central", "north-east"]
    security_type: Literal["direct", "indriect"]
    dtir1: Optional[float] = Field(default=None, ge=5, le=61)


#--Preprocessing------

cat_columns = [
    'loan_limit', 'gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
    'credit_worthiness', 'open_credit', 'business_or_commercial',
    'neg_ammortization', 'interest_only', 'lump_sum_payment',
    'construction_type', 'occupancy_type', 'secured_by', 'total_units',
    'co-applicant_credit_type', 'age',
    'submission_of_application', 'region', 'security_type',
    'dtir1_was_missing'
       ]
    
num_columns = [
    'loan_amount', 'term', 'property_value', 'income', 'credit_score',
    'ltv', 'dtir1'
       ]

def preprocess(loan_application: LoanApplication) -> pd.DataFrame:
    df = pd.DataFrame([loan_application.model_dump(by_alias=True)])
    
    df['income'] = df['income'].fillna(medians['income'])
    df['dtir1_was_missing'] = df['dtir1'].isnull().astype('int')
    df['dtir1'] = df['dtir1'].fillna(medians['dtir1'])
    df['property_value'] = df['property_value'].fillna(medians['property_value'])
    df['ltv'] = df['ltv'].fillna(medians['ltv'])
    
    cat_encoded = encoder.transform(df[cat_columns])
    num_values = scaler.transform(df[num_columns])
    
    X = np.hstack([num_values, cat_encoded])
    return X
    

#--Prediction------

def predict(loan_application: LoanApplication):

    X = preprocess(loan_application)

    probability = model.predict_proba(X)[0, 1]

    prediction = int(
        probability >= THRESHOLD
    )

    return {
        "default_probability": float(probability),
        "prediction": prediction
    }