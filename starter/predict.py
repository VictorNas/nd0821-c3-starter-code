import sys
sys.path.append('./starter')

from ml.data import process_data
from ml.model import inference

def predict(item ,model, encoder, lb ):
    cat_features = [ "workclass",
                     "education",
                     "marital-status",
                     "occupation",
                     "relationship",
                     "race",
                     "sex",
                     "native-country"]

    X, _ , _, _ = process_data(item, categorical_features= cat_features,
                               label=None, training=False, encoder=encoder, lb=lb
    )

    preds = inference(model, X)
    print('Preds', preds)
    return preds