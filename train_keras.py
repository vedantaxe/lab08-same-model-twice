
"""Train the Keras bike rental model and save reports."""

from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras_model import build_keras_model
from prepare import prepare_data

REPORTS = Path("reports")


def train_keras(seed=2026, epochs=100, learning_rate=0.001, batch_size=64):
    """Train Keras on the same prepared data as PyTorch."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(seed)

    data = prepare_data(seed=seed)

    x_train = data["train"].features.numpy()
    y_train = data["train"].targets.numpy()
    x_val = data["validation"].features.numpy()
    y_val = data["validation"].targets.numpy()

    model = build_keras_model(input_size=x_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=epochs,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(REPORTS / "best_keras_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    history_frame = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.history["loss"]) + 1),
            "train_loss": history.history["loss"],
            "validation_loss": history.history["val_loss"],
            "validation_mae": history.history["val_mae"],
        }
    )

    history_frame.to_csv(REPORTS / "keras_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_frame["epoch"],
        history_frame["train_loss"],
        label="Training MSE",
    )
    plt.plot(
        history_frame["epoch"],
        history_frame["validation_loss"],
        label="Validation MSE",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Keras Bike Rental Learning Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "keras_learning_curves.png", dpi=150)
    plt.close()

    best_epoch = int(history_frame["validation_loss"].idxmin()) + 1

    print("\nKeras training complete!")
    print("Best epoch:", best_epoch)
    print(
        "Best validation MSE:",
        round(float(history_frame["validation_loss"].min()), 4),
    )
    print("Reports saved in:", REPORTS.resolve())

    return {
        "model": model,
        "history": history_frame,
        "best_epoch": best_epoch,
    }


if __name__ == "__main__":
    train_keras()
