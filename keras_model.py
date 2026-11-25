
"""Keras equivalent of the PyTorch bike rental network."""

import keras
import numpy as np
import torch

from model import BikeRentalNetwork


def build_keras_model(input_size=8, hidden_sizes=(64, 32)):
    """Build the same architecture as BikeRentalNetwork."""
    layers = [keras.layers.Input(shape=(input_size,))]

    for hidden_size in hidden_sizes:
        layers.append(
            keras.layers.Dense(hidden_size, activation="relu")
        )

    layers.append(keras.layers.Dense(1))

    return keras.Sequential(layers)


def copy_torch_weights(torch_model, keras_model):
    """Copy PyTorch Linear weights and biases into Keras Dense layers."""
    torch_layers = [
        layer
        for layer in torch_model.modules()
        if isinstance(layer, torch.nn.Linear)
    ]

    keras_layers = [
        layer
        for layer in keras_model.layers
        if isinstance(layer, keras.layers.Dense)
    ]

    if len(torch_layers) != len(keras_layers):
        raise ValueError("The models have different numbers of Dense layers.")

    for torch_layer, keras_layer in zip(
        torch_layers, keras_layers, strict=True
    ):
        weights = torch_layer.weight.detach().cpu().numpy().T
        biases = torch_layer.bias.detach().cpu().numpy()

        keras_layer.set_weights([weights, biases])


def check_equivalence(torch_model, keras_model, features, tolerance=1e-5):
    """Check whether both models produce equivalent predictions."""
    features = np.asarray(features, dtype=np.float32)

    torch_model.eval()

    with torch.no_grad():
        torch_predictions = (
            torch_model(torch.from_numpy(features))
            .cpu()
            .numpy()
            .reshape(-1)
        )

    keras_predictions = (
        keras_model.predict(features, verbose=0).reshape(-1)
    )

    maximum_difference = float(
        np.max(np.abs(torch_predictions - keras_predictions))
    )

    return maximum_difference, maximum_difference <= tolerance


if __name__ == "__main__":
    torch_model = BikeRentalNetwork()
    keras_model = build_keras_model()

    torch_parameters = sum(
        parameter.numel() for parameter in torch_model.parameters()
    )

    keras_parameters = keras_model.count_params()

    print("PyTorch parameters:", torch_parameters)
    print("Keras parameters:", keras_parameters)

    copy_torch_weights(torch_model, keras_model)

    sample_features = np.random.default_rng(2026).normal(
        size=(64, 8)
    ).astype(np.float32)

    difference, equivalent = check_equivalence(
        torch_model, keras_model, sample_features
    )

    print("Maximum prediction difference:", difference)
    print("Predictions equivalent:", equivalent)

    assert torch_parameters == keras_parameters
    assert equivalent, "PyTorch and Keras predictions do not match."
