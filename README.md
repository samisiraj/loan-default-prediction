# Loan Default Prediction

> XGBoost model predicting loan default probability, served via FastAPI + Gradio, containerized and deployed on Render.

![Demo Screenshot](/reports/figures/app.png)

**Live demo:** https://loan-default-prediction-d6n8.onrender.com/

Note: Render free tier spins down on inactivity — first request after idle may take ~30-60s to wake up.

 **Full technical writeup:** [report.md](./reports/report.md)

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running Locally](#running-locally)
- [API Usage](#api-usage)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)
- [Limitations & Assumptions](#limitations--assumptions)
- [Acknowledgments](#acknowledgments)

---

## Overview
Banks frequently use ML systems to automatically reject loan applications before a human reviews them. Based on this use case of machine learning in banking, my ML project receives a loan application and predict's whether the applicant will default or not.

## Dataset

- Source: https://www.kaggle.com/datasets/yasserh/loan-default-dataset (Kaggle, "Loan Default Classification Problem")
- Size: [148670 rows] x [34 columns]
- Target variable: `status` (binary: default / no default)
- The dataset contains a collection of numerical and categorical features related to applicant's information like `gender`, `age`, `income`; loan information like `loan_limit`, `loan_type`,`loan_purpose`; and more.
- IMPORTANT — The dataset does not document units/currency for monetary fields; we have inferred USD from context (US region categories, HMDA-style fields) and treated this explicitly rather than assuming it. The dataset also has ambiguity about some fields like `loan_types` is a categorical feature with categories `type1`, `type2` and `type3` but without explicit mention of what these categories mean. We have still included these categories to be filled for getting the prediction outcome.
- More in-depth details about the leakage features identified and dropped during exploration and more are available in [report.md](./reports/report.md).
## Results

| Metric | Value |
|---|---|
| Model | XGBoost |
| Test ROC-AUC | 0.890 |
| Test PR-AUC | 0.834 |
| Decision threshold | 0.23 |

![Threshold range](reports/figures/threshold.png)

We selected 0.23 because it achieved our target recall while maintaining acceptable precision.

- Why XGBoost? For our business model the most important metric was recall (Out of every entry who actually defaulted how many did our model actually classify as a defaulting applicant). We can compare evaluation metrics for every models we tried [here](./reports/metrics.json). XGBoost had the best ROC-AUC and PR-AUC scores for the recall value we targeted.

<br>

![ROC curves for all models compared](reports/figures/roc_all_models.png)


## Project Structure

```
loan-default-prediction/
├── data/raw/                 
├── models/
│   ├── final_encoder.pkl            
│   ├── final_medians.pkl
│   ├── final_scaler.pkl           
│   └── final_xgb_model.pkl
├── notebooks/                 
├── reports/
│   ├── figures
│   │   └──roc_all_models.png
│   ├── metrics.json
│   └── report.md
├── src/
│   ├── api.py                
│   ├── app.py             
│   ├── data.py             
│   ├── features.py       
│   ├── predict.py               
│   └── train.py  
├── .dockerignore
├── .gitattributes
├── .gitignore            
├── .python-version
├── Dockerfile
├── README.md
├── pyproject.toml
├── start.sh
└── uv.lock
```

## Setup & Installation

<!--
Suggested content (adjust to whatever you actually use — pip, uv, poetry):
1. Clone the repo
2. Create virtual environment
3. Install dependencies
-->

```bash
git clone https://github.com/samisiraj/loan-default-prediction
cd loan-default-prediction
uv sync
```

## Running Locally

<!--
Suggested content:
- How to run without Docker (two terminals: uvicorn + python app.py)
- How to run with Docker (docker build, docker run)
-->

**Without Docker:**
```bash
./start.sh
```

**With Docker:**
```bash
docker build -t loan-default-prediction .
docker run -p 7860:7860 loan-default-prediction
```

## API Usage

<!--
Suggested content:
- Example curl request to /predict
- Note on /ping health check endpoint
- Link to or embed the Pydantic schema field list if useful
-->

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_limit": "cf",
    "gender": "male",
    "approv_in_adv": "pre",
    "loan_type": "type1",
    "loan_purpose": "p1",
    "credit_worthiness": "l1",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "loan_amount": 150000,
    "term": 180,
    "neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "property_value": 250000,
    "construction_type": "sb",
    "occupancy_type": "pr",
    "secured_by": "home",
    "total_units": "1u",
    "income": 60000,
    "credit_score": 650,
    "co-applicant_credit_type": "cib",
    "age": "25-34",
    "submission_of_application": "to_inst",
    "ltv": 80.5,
    "region": "north",
    "security_type": "direct",
    "dtir1": 35
  }'
```

### Response:
```bash
{
  "default_probability":0.056,
  "prediction":0
  }
```
## Deployment

- Hosted on Render (free tier), Docker-based deployment
- Single container runs both FastAPI and Gradio via start.sh


## Tech Stack

- Python
- XGBoost, scikit-learn
- FastAPI
- Gradio
- Docker
- Render (hosting)

## Limitations & Assumptions

- **Currency assumption:** The dataset does not document a currency or unit for monetary fields (`loan_amount`, `property_value`, `income`, etc.). USD was assumed based on contextual clues — US-style region categories and a categorical field structure resembling HMDA-style mortgage datasets — but this is an inference, not a confirmed fact from the source.
- **Undocumented categorical codes:** Fields like `loan_purpose` (p1–p4), `credit_worthiness` (l1/l2), and `loan_type` (type1–3) are dataset-defined codes with no published meaning. These were kept rather than dropped, since categorical values only need to be internally consistent between training and inference to be useful — the model doesn't need to know what "p1" means, only that it behaves consistently.
- **Decision threshold:** The classification threshold (0.23, below the default 0.5) was chosen to prioritize recall on defaulting applicant over precision — in a lending context, missing an actual defaulting applicant is typically costlier than flagging a safe applicant for review, so the threshold was tuned to catch more true defaults at the cost of more false positives.
- **Hosting:** Deployed on Render's free tier, which spins down after inactivity — the first request after idling may take 30–60 seconds to respond.
- **Dataset scope:** This is a portfolio/learning project built on a single public Kaggle dataset. It has not been validated against real-world lending data, and the class balance, feature distributions, or default patterns may not generalize beyond this dataset.

## Acknowledgments

- Course: [ML Zoomcamp: Free Machine Learning Engineering Course](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club
- Dataset: [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset) by yasserh on Kaggle
