"""
Train Logistic Regression Model for salary classification

Author: Udacity
Modified by: Gustavo Grinsteins
Modified Date: 08/13/2023
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, output_metrics_by_slice


if __name__ == "__main__":
    # Add code to load in the data.
    data = pd.read_csv('../data/census.csv')
    
    # Data Pre processing: Remove missing values represente by '?'
    data = data.replace('?', np.nan)
    data = data.dropna(axis=0)

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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print("Precision:", precision)
    print("Recall", recall)
    print("fbeta", fbeta)

    pickle.dump(model, open('../model/LogisticRegressionModel.pkl', 'wb'))
    pickle.dump(encoder, open('../model/Encoder.pkl', 'wb'))
    pickle.dump(lb, open('../model/LinearBinarizer.pkl', 'wb'))

    output_metrics_by_slice(
        df=test,
        model=model,
        encoder=encoder,
        lb=lb,
        cat_features=cat_features
    )
