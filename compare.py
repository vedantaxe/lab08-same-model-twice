"""Compare equivalent PyTorch and Keras models across three seeds."""

import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pandas as pd
import torch

from keras_model import build_keras_model
from model import BikeRentalNetwork
from prepare import create_loaders, prepare_data
from training import fit_model

REPORTS = Path("reports")
SEEDS = (2026, 2027, 2028)
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def predict_torch(model, features):
    """Return PyTorch predictions as a NumPy array."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(features)).numpy().reshape(-1)


def mae(targets, predictions):
    """Calculate mean absolute error."""
    return float(np.mean(np.abs(targets.reshape(-1) - predictions.reshape(-1))))


def run_torch(seed, data):
    """Train PyTorch, restore best weights, and evaluate once."""
    torch.manual_seed(seed)

    loaders = create_loaders(data, batch_size=BATCH_SIZE, seed=seed)
    input_size = data["train"].features.shape[1]
    model = BikeRentalNetwork(input_size=input_size, hidden_sizes=(64, 32))

    start = time.perf_counter()
    result = fit_model(
        model,
        loaders["train"],
        loaders["validation"],
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )
    seconds = time.perf_counter() - start

    x_val = data["validation"].features.numpy()
    y_val = data["validation"].targets.numpy()
    x_test = data["test"].features.numpy()
    y_test = data["test"].targets.numpy()

    validation_mae = mae(y_val, predict_torch(model, x_val))
    test_mae = mae(y_test, predict_torch(model, x_test))

    return {
        "framework": "PyTorch",
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "validation_mae": validation_mae,
        "test_mae": test_mae,
        "seconds": seconds,
    }


def run_keras(seed, data):
    """Train Keras, restore best weights, and evaluate once."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    x_train = data["train"].features.numpy()
    y_train = data["train"].targets.numpy()
    x_val = data["validation"].features.numpy()
    y_val = data["validation"].targets.numpy()
    x_test = data["test"].features.numpy()
    y_test = data["test"].targets.numpy()

    model = build_keras_model(input_size=x_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EPOCHS,
        restore_best_weights=True,
    )

    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[callback],
        verbose=0,
    )
    seconds = time.perf_counter() - start

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    validation_predictions = model.predict(x_val, verbose=0).reshape(-1)
    test_predictions = model.predict(x_test, verbose=0).reshape(-1)

    return {
        "framework": "Keras",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": mae(y_val, validation_predictions),
        "test_mae": mae(y_test, test_predictions),
        "seconds": seconds,
    }


def main():
    """Run six experiments and save the comparison."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in SEEDS:
        print(f"\nPreparing data for seed {seed}...", flush=True)
        data = prepare_data(seed=seed)

        print(f"Training PyTorch: seed {seed}", flush=True)
        torch_result = run_torch(seed, data)
        rows.append(torch_result)
        print(torch_result, flush=True)

        print(f"Training Keras: seed {seed}", flush=True)
        keras_result = run_keras(seed, data)
        rows.append(keras_result)
        print(keras_result, flush=True)

    frame = pd.DataFrame(rows)
    output = REPORTS / "comparison.csv"
    frame.to_csv(output, index=False)

    print("\nFull comparison:")
    print(frame.to_string(index=False))

    print("\nFramework averages:")
    print(
        frame.groupby("framework")[["validation_mae", "test_mae", "seconds"]]
        .mean()
        .round(4)
    )

    print("\nSaved:", output.resolve())


if __name__ == "__main__":
    main()
