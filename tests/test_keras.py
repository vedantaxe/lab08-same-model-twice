"""Tests for the Keras bike rental implementation."""

import keras
import numpy as np
import pytest
import torch

from keras_model import (
    build_keras_model,
    check_equivalence,
    copy_torch_weights,
)
from model import BikeRentalNetwork
from prepare import prepare_data


@pytest.mark.parametrize(
    ("hidden_sizes", "expected_parameters"),
    [
        ((64, 32), 2689),
        ((32, 16), 833),
        ((16, 8), 289),
    ],
)
def test_parameter_count(hidden_sizes, expected_parameters):
    model = build_keras_model(input_size=8, hidden_sizes=hidden_sizes)
    assert model.count_params() == expected_parameters


def test_output_shape():
    model = build_keras_model()
    features = np.zeros((12, 8), dtype=np.float32)
    predictions = model(features, training=False).numpy()
    assert predictions.shape == (12, 1)


def test_dense_layer_count():
    model = build_keras_model(hidden_sizes=(64, 32))
    dense_layers = [
        layer for layer in model.layers if isinstance(layer, keras.layers.Dense)
    ]
    assert len(dense_layers) == 3


def test_weight_equivalence():
    torch.manual_seed(2026)
    torch_model = BikeRentalNetwork(input_size=8)
    keras_model = build_keras_model(input_size=8)

    copy_torch_weights(torch_model, keras_model)

    features = np.random.default_rng(2026).normal(size=(32, 8)).astype(np.float32)

    difference, equivalent = check_equivalence(
        torch_model, keras_model, features, tolerance=1e-5
    )

    assert equivalent
    assert difference < 1e-5


def test_prepared_data_is_repeatable():
    first = prepare_data(seed=2026)
    second = prepare_data(seed=2026)

    np.testing.assert_array_equal(
        first["train"].features.numpy(),
        second["train"].features.numpy(),
    )
    np.testing.assert_array_equal(
        first["test"].targets.numpy(),
        second["test"].targets.numpy(),
    )


def test_keras_save_and_reload(tmp_path):
    keras.utils.set_random_seed(2026)
    model = build_keras_model()

    features = np.random.default_rng(2026).normal(size=(8, 8)).astype(np.float32)

    original = model(features, training=False).numpy()

    path = tmp_path / "test_model.keras"
    model.save(path)

    restored = keras.models.load_model(path)
    reloaded = restored(features, training=False).numpy()

    np.testing.assert_allclose(original, reloaded, atol=1e-6)


def test_keras_initialization_is_repeatable():
    keras.backend.clear_session()
    keras.utils.set_random_seed(2026)
    first = build_keras_model()
    first_weights = [weight.copy() for weight in first.get_weights()]

    keras.backend.clear_session()
    keras.utils.set_random_seed(2026)
    second = build_keras_model()
    second_weights = second.get_weights()

    for first_weight, second_weight in zip(first_weights, second_weights, strict=True):
        np.testing.assert_array_equal(first_weight, second_weight)


def test_keras_can_memorize_tiny_batch():
    """Verify that Keras can memorize a tiny dataset."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(2026)

    model = build_keras_model(input_size=8, hidden_sizes=(32, 16))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
    )

    rng = np.random.default_rng(2026)
    features = rng.normal(size=(4, 8)).astype(np.float32)

    targets = (
        2.0 * features[:, 0] - 1.5 * features[:, 1] + 0.5 * features[:, 2]
    ).astype(np.float32)

    initial_predictions = model(features, training=False).numpy().reshape(-1)
    initial_mse = float(np.mean((initial_predictions - targets) ** 2))

    model.fit(
        features,
        targets,
        epochs=300,
        batch_size=4,
        shuffle=False,
        verbose=0,
    )

    predictions = model(features, training=False).numpy().reshape(-1)
    final_mse = float(np.mean((predictions - targets) ** 2))

    assert final_mse < initial_mse
    assert final_mse < 0.1


def test_best_weights_are_restored():
    keras.backend.clear_session()
    keras.utils.set_random_seed(2026)

    model = build_keras_model(input_size=8)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )

    features = np.random.default_rng(2026).normal(size=(32, 8)).astype(np.float32)
    targets = np.ones(32, dtype=np.float32)

    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    history = model.fit(
        features,
        targets,
        validation_data=(features, targets),
        epochs=10,
        batch_size=8,
        callbacks=[callback],
        verbose=0,
    )

    final_loss = float(model.evaluate(features, targets, verbose=0))
    best_loss = min(history.history["val_loss"])

    assert final_loss == pytest.approx(best_loss, abs=1e-3)
