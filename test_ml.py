import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference
from ml.model import compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Verify that perfect predictions return
    precision, recall and F1 equal to 1.0.
    """
    # Your code here
    y = [1, 1, 0, 0]
    preds = [1, 1, 0, 0]

    precision, recall, fbeta = compute_model_metrics(
        y, preds
    )

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0

# TODO: implement the second test. Change the function name and input as needed
def test_train_model():
    """
    Verify that train_model returns a
    RandomForestClassifier instance.
    """
    # Your code here
    X = np.array([
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0]
    ])

    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)

    assert isinstance(
        model,
        RandomForestClassifier
    )


# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    """
    Verify that inference returns the
    expected number of predictions.
    """
    # Your code here
    X = np.array([
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0]
    ])

    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)

    preds = inference(model, X)

    assert len(preds) == len(y)