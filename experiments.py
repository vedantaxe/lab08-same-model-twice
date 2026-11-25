"""Run controlled neural network experiments."""

import torch

from model import BikeRentalNetwork
from prepare import create_loaders
from training import fit_model

EXPERIMENTS = [
    {
        "name": "main",
        "hidden_sizes": (64, 32),
        "learning_rate": 0.001,
    },
    {
        "name": "higher_learning_rate",
        "hidden_sizes": (64, 32),
        "learning_rate": 0.005,
    },
    {
        "name": "wider_first_layer",
        "hidden_sizes": (128, 32),
        "learning_rate": 0.001,
    },
]


def run_experiments(data, epochs=100, seed=2026):
    """Train and compare models using validation loss only."""
    results = []
    best_model = None
    best_result = None
    best_validation_loss = float("inf")

    input_size = data["train"].features.shape[1]

    for config in EXPERIMENTS:
        torch.manual_seed(seed)

        loaders = create_loaders(
            data,
            batch_size=64,
            seed=seed,
        )

        model = BikeRentalNetwork(
            input_size=input_size,
            hidden_sizes=config["hidden_sizes"],
        )

        print(f"Running experiment: {config['name']}")

        training_result = fit_model(
            model,
            loaders["train"],
            loaders["validation"],
            epochs=epochs,
            learning_rate=config["learning_rate"],
        )

        validation_loss = training_result["best_validation_loss"]

        results.append(
            {
                "name": config["name"],
                "hidden_sizes": str(config["hidden_sizes"]),
                "learning_rate": config["learning_rate"],
                "best_epoch": training_result["best_epoch"],
                "validation_mse": validation_loss,
                "history": training_result["history"],
            }
        )

        print(f"Best validation MSE: {validation_loss:.4f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = model
            best_result = config["name"]

    return {
        "experiments": results,
        "best_model": best_model,
        "best_experiment": best_result,
    }
