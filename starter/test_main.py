from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_greetings():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to Incomer Predictor API!!"}
def test_predict_item_pos():
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
    response = client.post("/predict/", headers=data)
    assert response.status_code == 200
    assert response.json() == { "preds": [">50K"]}


def test_predict_item_neg():
        data = {
        "age": 38,
        "workclass": "Private",
        "fnlgt": 28887,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 0,
        "native-country": "United-States"
         }
        response = client.post("/predict/", headers=data)
        assert response.status_code == 200
        assert response.json() == {"preds": ["<=50K"]}
