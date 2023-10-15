from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Furniture(BaseModel):
    a: int
    b: int
    c: int
    d: float
    h: float
    w: float

from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import pickle

# Load the model using pickle
with open('C:/Users/Salma/Desktop/S3_master/cloud_native_ai_kelloubi/seance_fastAPI/Lab folder - FastAPI Pydantic/Lab folder - FastAPI Pydantic/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("item.html", {"request": request, "prediction": "ML model for Furniture prediction"})

@app.post('/make_predictions', response_class=HTMLResponse)
async def make_predictions(request: Request, features: Furniture):
    prediction = model.predict([[features.a, features.b, features.c, features.d, features.h, features.w]])[0]
    return templates.TemplateResponse("prediction_template.html", {"request": request, "prediction": prediction})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
