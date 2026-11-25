"""Run reproducible bike rental experiments and save reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from baselines import evaluate_baselines
from experiments import run_experiments
from prepare import prepare_data
from training import evaluate_one_epoch

REPORTS = Path("reports")


def save_learning_curve(history, filename, title):
    """Save training and validation MSE curves."""
    frame = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(frame["epoch"], frame["train_loss"], label="Training")
    plt.plot(
        frame["epoch"],
        frame["validation_loss"],
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / filename, dpi=150)
    plt.close()


def main():
    """Run experiments and generate all required outputs."""
    REPORTS.mkdir(parents=True, exist_ok=True)

    seed = 2026
    torch.manual_seed(seed)

    print("Preparing bike rental data...")
    data = prepare_data(seed=seed)

    print("Running neural network experiments...")
    results = run_experiments(data, epochs=100, seed=seed)

    experiments = results["experiments"]
    best_model = results["best_model"]
    best_name = results["best_experiment"]

    # Save all experiment learning curves.
    for experiment in experiments:
        save_learning_curve(
            experiment["history"],
            f"{experiment['name']}_learning_curves.png",
            f"Learning Curves: {experiment['name']}",
        )

    # Save the main experiment's learning curves and history.
    main_experiment = experiments[0]

    save_learning_curve(
        main_experiment["history"],
        "learning_curves.png",
        "Main Model Learning Curves",
    )

    pd.DataFrame(main_experiment["history"]).to_csv(
        REPORTS / "history.csv",
        index=False,
    )

    # Save experiment comparison without nested histories.
    experiment_rows = [
        {key: value for key, value in experiment.items() if key != "history"}
        for experiment in experiments
    ]

    pd.DataFrame(experiment_rows).to_csv(
        REPORTS / "experiments.csv",
        index=False,
    )

    # Final evaluation: the selected model is tested once.
    from prepare import create_loaders

    loaders = create_loaders(data, seed=seed)
    loss_fn = nn.MSELoss()

    network_test_mse = evaluate_one_epoch(
        best_model,
        loaders["test"],
        loss_fn,
    )

    baseline_results = evaluate_baselines(data)

    metrics = {
        "seed": seed,
        "selected_experiment": best_name,
        "selected_validation_mse": min(
            experiment["validation_mse"] for experiment in experiments
        ),
        "neural_network_test_mse": network_test_mse,
        "baselines": baseline_results,
    }

    with (REPORTS / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nBest experiment:", best_name)
    print("Neural network test MSE:", round(network_test_mse, 4))
    print("Reports saved in:", REPORTS.resolve())


if __name__ == "__main__":
    main()
