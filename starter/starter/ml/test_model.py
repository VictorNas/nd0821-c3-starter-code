import sys
sys.path.append('../starter')
from ml.model import load_model, save_model, inference, train_model
from ml.data import process_data
import pandas as pd
import os

path = './'
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
data = pd.read_csv('../data/census_clean.csv')

X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
def test_train_model():
    model = train_model(X,y)
    assert 'sklearn.ensemble._forest.RandomForestClassifier' in str(type(model)) , \
        f'model is of the type {type(model)} expected sklearn.ensemble._forest.RandomForestClassifier'

model = train_model(X,y)

def test_inference():
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0], 'expected one pred for each row in X'

def test_save_model():
    save_model(model ,encoder, lb, path)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    assert 'rfc_model.pkl' in files, 'rfc_model.pkl are not in the specified path'
    assert 'encoder.pkl' in files, 'encoder.pkl are not in the specified path'
    assert 'lb.pkl' in files, 'lb.pkl are not in the specified path'

def test_load_model():

    model, encoder, lb = load_model(path)
    assert 'sklearn.ensemble._forest.RandomForestClassifier' in str(type(model)) , \
        f'model is of the type {type(model)} expected sklearn.ensemble._forest.RandomForestClassifier'
    assert 'sklearn.preprocessing._encoders.OneHotEncoder' in str(type(encoder)) , \
        f'encoder is of the type {type(encoder)} expected sklearn.preprocessing._encoders.OneHotEncoder'
    assert 'sklearn.preprocessing._label.LabelBinarizer' in str(type(lb)), \
        f'lb is of the type {type(lb)} expected sklearn.preprocessing._label.LabelBinarizer'

