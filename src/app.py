from fastapi import FastAPI
import uvicorn

from predict import LoanApplication, predict

app = FastAPI(title='Loan Default Prediction API')

@app.get('/ping')
def ping():
    return 'PONG'

@app.post('/predict')
def predict_endpoint(application: LoanApplication):
    return predict(application)

if __name__== '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)