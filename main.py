"""
ML application API Logic

Author: Gustavo Grinsteins
Date Created: 08/18/2023
"""

import pickle
import pandas as pd
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference

CAT_FEATURES = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

app = FastAPI()

class InferenceInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "workclass": "State-gov",
                "fnlgt": 141297,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "Asian-Pac-Islander",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "India"
            }
        }
        

@app.get("/")
async def get_greetings():
    return {"Greetings": "Welcome to the Income Prediction Tool (***Used for education purposes only***)"}

@app.post("/inference/")
async def post_inference(inference_input: InferenceInput) -> Dict:
    """
    Do model inference on census tuple to predict salary.

    Inputs
    ------
    inference_input (InferenceInput) : census data for one profile

    Returns
    -------
    Inference_prediction (Dict) : Predicted salary <=50K or >50K
    """
    model = pickle.load(open('model/LogisticRegressionModel.pkl', 'rb'))
    encoder = pickle.load(open('model/Encoder.pkl', 'rb'))
    lb = pickle.load(open('model/LinearBinarizer.pkl', 'rb'))
    # Replace python naming convention to match census. Dictionary comprehension
    # code adapted from https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
    inference_input = {feature.replace('_', '-'): value for feature, value in inference_input.dict().items()}
    inference_input_df = pd.DataFrame(inference_input, index=[0])
    print(inference_input_df.head())
    model_input, _, _, _ = process_data(
        inference_input_df, categorical_features=CAT_FEATURES, training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, model_input)
    salary = lb.inverse_transform(preds)[0]
    return {"Salary": salary}
