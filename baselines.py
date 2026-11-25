"""Baseline models for the bike rental prediction task."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def calculate_mse(actual, predicted):
    """Calculate mean squared error."""
    return float(mean_squared_error(actual, predicted))


def evaluate_baselines(data):
    """Evaluate training-mean and linear regression baselines."""
    x_train = data["train"].features.numpy()
    y_train = data["train"].targets.numpy()

    x_validation = data["validation"].features.numpy()
    y_validation = data["validation"].targets.numpy()

    x_test = data["test"].features.numpy()
    y_test = data["test"].targets.numpy()

    # Baseline 1: Predict the training-set mean.
    training_mean = float(np.mean(y_train))

    mean_validation_predictions = np.full_like(y_validation, training_mean)
    mean_test_predictions = np.full_like(y_test, training_mean)

    # Baseline 2: Linear regression.
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)

    linear_validation_predictions = linear_model.predict(x_validation)
    linear_test_predictions = linear_model.predict(x_test)

    return {
        "mean_baseline": {
            "validation_mse": calculate_mse(y_validation, mean_validation_predictions),
            "test_mse": calculate_mse(y_test, mean_test_predictions),
        },
        "linear_regression": {
            "validation_mse": calculate_mse(
                y_validation, linear_validation_predictions
            ),
            "test_mse": calculate_mse(y_test, linear_test_predictions),
        },
    }
