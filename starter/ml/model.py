"""
Logistic Regression Model Functions

Author: Udacity
Modified by: Gustavo Grinsteins
Modified Date: 08/13/2023
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from .data import process_data

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
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and Fbeta.

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
    model : Logistic Regression
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

def output_metrics_by_slice(df,
                            model,
                            encoder,
                            lb,
                            cat_features):
    """
    performance of the model on data slices

    Inputs
    ------
        - df (pd.DataFrame): Input dataframe
        - model (sklearn.linear_model.LogisticRegression): Trained model binary file
        - encoder (sklearn.preprocessing.OneHotEncoder): fitted One Hot Encoder
        - lb (sklearn.preprocessing.LabelBinarizer): label binarizer
        - cat_features (list): list of categorical columns

    Returns
    -------
        - None
        
    File Output
    -------
        - slice_output.txt
    """

    with open("../slice_output.txt", 'w') as output_file:
        for column in cat_features:
            output_file.write(f"{str(column).upper()}, PRECISION, RECALL, FBETA\n")
            for category in df[column].unique():

                features, labels, _, _ = process_data(
                    X=df[df[column]==category],
                    categorical_features=cat_features,
                    label='salary',
                    training=False,
                    encoder=encoder,
                    lb=lb
                )

                preds = inference(model, features)

                precision, recall, fbeta = compute_model_metrics(labels, preds)
                output_file.write(f"{str(category)}, {precision}, {recall}, {fbeta}\n")