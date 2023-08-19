"""
API Code Test

Author: Gustavo Grinsteins
Date Created: 08/18/2023
"""

import sys
# Resolve issue with pytest not finding user defined modules https://knowledge.udacity.com/questions/981139
sys.path.append('./')
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Greetings": "Welcome to the Income Prediction Tool (***Used for education purposes only***)"}

def test_api_post_success_greater_than_50K():
    payload = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 149624,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 10000,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    response = client.post("/inference/", json = payload)
    assert response.status_code == 200
    assert response.json() == {"Salary": ">50K"}

def test_api_post_success_less_than_equal_to_50K():
    payload = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 99928,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"}

    response = client.post("/inference/", json = payload)
    assert response.status_code == 200
    assert response.json() == {"Salary": "<=50K"}
    