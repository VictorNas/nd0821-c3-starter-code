from .data import process_data
import pandas as pd

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

def test_process_data():
    data = pd.read_csv('./starter/data/census_clean.csv').loc[:50]
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert X.shape[0] == y.shape[0], 'X and y do not have the same number of columns'
    assert 'sklearn.preprocessing._encoders.OneHotEncoder' in str(type(encoder)), \
        f'encoder is of the type {type(encoder)} expected sklearn.preprocessing._encoders.OneHotEncoder'
    assert 'sklearn.preprocessing._label.LabelBinarizer' in str(type(lb)) , \
        f'lb is of the type {type(lb)} expected sklearn.preprocessing._label.LabelBinarizer'
