# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from starter.ml.model import load_model
from predict import predict

# load the model
model, encoder, lb = load_model('starter/starter/model')

# Declare the data object with its components and their type.
def alias_function(x : str) -> str:
    return x.replace('_','-')

class Input_Data(BaseModel):
    age: int = 38
    workclass: str = "Private"
    fnlgt: int = 28887
    education: str = "11th"
    education_num: int = 7
    marital_status: str = "Married-civ-spouse"
    occupation: str = "Sales"
    relationship: str = "Husband"
    race: str = "White"
    sex: str = 'Male'
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 50
    native_country: str = "United-States"
    class Config:
        alias_generator = alias_function

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def greetings():
    return {"greeting": "Welcome to Incomer Predictor API!!"}

@app.post("/predict/")
async def predict_item(item: Input_Data):

    data = pd.DataFrame(item.dict(by_alias=True),index=[0])
    preds = predict(data,model,encoder,lb)
    preds = ['<=50K' if p ==0 else '>50K' for p in preds]
    return {'preds': preds}
