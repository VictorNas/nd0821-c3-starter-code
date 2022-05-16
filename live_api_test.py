import requests
import json

data = {
      "age": 57,
      "workclass": "Federal-gov,",
      "fnlgt": 337895,
      "education": "Bachelors",
      "education-num": 13,
      "marital-status": "Married-civ-spouse",
      "occupation": "Prof-specialty",
      "relationship": "Husband",
      "race": "Black",
      "sex": "Male",
      "capital-gain": 0,
      "capital-loss": 0,
      "hours-per-week": 40,
      "native-country": "United-States"
    }
response = requests.post('https://income-predictor-0456.herokuapp.com/predict', data= json.dumps(data))
print(response.status_code)
print(response.json())