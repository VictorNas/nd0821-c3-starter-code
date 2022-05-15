from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
from data import process_data
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimatorsint=500, criterion='gini',
                                           max_depth=100, max_features='auto', random_state=42)
    model.fit(X_train, y_train)

    return model



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def compute_model_performance(model, X, categorical_features=[], label=None, encoder=None, lb=None):
    """ Compute the model performance on slices of the data.

       Inputs
       ------
       model : sklearn.ensemble._forest.RandomForestClassifier
               Trained machine learning model.
       X : pd.DataFrame
           Dataframe containing the features and label. Columns in `categorical_features`
       categorical_features: list[str]
           List containing the names of the categorical features (default=[])
       label : str
           Name of the label column in `X`. If None, then an empty array will be returned
           for y (default=None)
       encoder : sklearn.preprocessing._encoders.OneHotEncoder
           Trained sklearn OneHotEncoder.
       lb : sklearn.preprocessing._label.LabelBinarizer
           Trained sklearn LabelBinarizer.

       Returns
       -------
       performance_slices : pd.DataFrame
           Dataframe with precisiom, recall and f1 in every possible slice of the X dataframe
           based on the categorical features.
       """

    model_peformance = {}
    # loop for all categorical_features
    for feature in categorical_features:
        feature_slices = X[feature].unique().tolist()
        # create a slice for every unique value under the feature
        for f_slice in feature_slices:
            # get the slice
            df_slice = X[X[feature] == f_slice]
            # process data from the slice
            X_test_slice, y_test_slice, _, _ = process_data(
                df_slice, categorical_features=categorical_features,
                label=label, training=False, encoder=encoder, lb=lb)
            # get inference from the slice
            preds = inference(model, X_test_slice)
            # save the results in a dict
            model_peformance[f'{feature}_{f_slice}'] = compute_model_metrics(y_test_slice, preds)

    # build a dataframe with the results
    performance_slices = pd.DataFrame(model_peformance).T
    performance_slices.columns = ['precision', 'recall', 'f1']
    performance_slices['slice'] = performance_slices.index
    performance_slices.reset_index(inplace=True, drop=True)

    return performance_slices