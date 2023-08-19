"""
Test POST method with the live API URL

Author: Gustavo Grinsteins
Date Created: 08/19/2023
"""

import requests
import json

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

response = requests.post("https://udacity-deploy-ml-cloud-app.onrender.com/inference/", data=json.dumps(payload))

print(f"Status: {str(response.status_code)}")
print("RESPONSE")
print(response.json())