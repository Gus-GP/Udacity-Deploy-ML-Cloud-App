"""
Code to test model script

Author: Gustavo Grinsteins
Date Created: 08/13/2023
"""

import sys
# Resolve issue with pytest not finding user defined modules https://knowledge.udacity.com/questions/981139
sys.path.append('./')
import pytest
import pickle
import pandas as pd
import numpy as np
from os.path import exists
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference, output_metrics_by_slice


@pytest.fixture()
def data():
    """ Create test access to data """
    # Add code to load in the data.
    data = pd.read_csv('data/census.csv')
    data = data.replace('?', np.nan)
    data = data.dropna(axis=0)
    train, _ = train_test_split(data, test_size=0.20)
    return train

@pytest.fixture()
def cat_features():
    """ Create test access to cat_features"""
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
    return cat_features

@pytest.fixture()
def model():
    """ Create test access to model"""
    model = pickle.load(open('model/LogisticRegressionModel.pkl', 'rb'))
    return model

@pytest.fixture()
def encoder():
    """ Create test access to Encoder"""
    model = pickle.load(open('model/Encoder.pkl', 'rb'))
    return model

@pytest.fixture()
def lb():
    """ Create test access to Linear Binarizer"""
    model = pickle.load(open('model/LinearBinarizer.pkl', 'rb'))
    return model

def test_train_model(data,cat_features):
    """ Test Train Model """
    train, _ = train_test_split(data, test_size=0.20)
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)

def test_compute_model_metrics(data,cat_features):
    """ Test Compute Model Metrics """

    # Proces the train data with the process_data function
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    precision, recall, fbeta = compute_model_metrics(y_train, preds)

    assert precision <= 1.0
    assert recall <=1.0
    assert fbeta <=1.0

def test_inference(data,cat_features,model):
    """ Test Logistic Regression Inference """

    X_train, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    preds = inference(model, X_train)

    assert len(preds) == len(X_train)

def test_output_metrics_by_slice(data,cat_features,model,encoder,lb):
    """ Test That The Output txt file for sliced metrics exists """
    _, test = train_test_split(data, test_size=0.20)
    output_metrics_by_slice(
        df=test,
        model=model,
        encoder=encoder,
        lb=lb,
        cat_features=cat_features
    )
    assert exists('slice_output.txt')
