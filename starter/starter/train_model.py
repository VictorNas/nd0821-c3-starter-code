# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import compute_model_metrics, inference, compute_model_performance, train_model
from ml.data import process_data
import pandas as pd
import logging
import joblib

# Add code to load in the data.
logging.info("Reading the Data...")
data = pd.read_csv('../data/census_clean.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
logging.info("Splitting the Data into train a test sets...")
train, test = train_test_split(data, test_size=0.20)

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
logging.info("processing the data...")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
logging.info("Training a Random Forest Classifier")
model = train_model(X_train, y_train)

logging.info("Saving the Model")
joblib.dump(model, '../model/rfc_model.pkl')

logging.info("Computing the model performance")
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

performance_slices = compute_model_performance(model, X_test, categorical_features=cat_features,
                          label="salary", encoder=encoder, lb=lb)

performance_slices = performance_slices.append({'f1': fbeta,
                                                'precision': precision,
                                                'recall': recall,
                                                'slice': 'all_data'},
                                                 ignore_index=True)
logging.info("Saving Model Performance...")

performance_slices.to_csv('../model/model_performance.csv')