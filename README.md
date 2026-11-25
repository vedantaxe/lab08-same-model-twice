
# Lab 07 — A Network from First Principles

## What it does

This project builds a PyTorch neural network from first principles to predict hourly bike rentals using simulated bike-share data.

The project implements custom data preparation, PyTorch Dataset and DataLoader objects, a feed-forward neural network, a manual training loop, baseline comparisons, and controlled experiments.

## How to run it

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate lab07-first-principles
```

Run the full experiment:

```bash
python train.py
```

Run the required quality checks:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Data and split

The provided `bikes_data.py` generates 6,000 deterministic bike rental records.

Input features:

- hour
- temperature_c
- rain_mm
- wind_kph
- is_weekend
- station_type

Target: rentals.

Data is split before preprocessing:

- Training: 3,600 rows (60%)
- Validation: 1,200 rows (20%)
- Test: 1,200 rows (20%)

Numerical features are standardized and station types are one-hot encoded.

The preprocessing transformations are fitted only on training data to prevent leakage.

## Model and training

The main neural network architecture is:

8 inputs → 64 neurons → ReLU → 32 neurons → ReLU → 1 output.

The model uses mean squared error loss and the Adam optimizer.

The training loop explicitly implements:

- optimizer.zero_grad()
- loss.backward()
- optimizer.step()

The best model checkpoint is selected using validation loss and restored after training.

## Results

The best experiment used a learning rate of 0.005.

| Model | Validation MSE | Test MSE |
|---|---:|---:|
| Training-mean baseline | 201.0260 | 190.2915 |
| Linear regression | 172.8229 | 159.7902 |
| Selected neural network | 13.6733 | 12.9449 |

The selected neural network substantially outperformed both baselines on the generated dataset.

## Experiments

Three neural network configurations were compared:

| Experiment | Hidden layers | Learning rate | Best epoch | Validation MSE |
|---|---|---|---:|---:|
| Main | (64, 32) | 0.001 | 100 | 18.2234 |
| Higher learning rate | (64, 32) | 0.005 | 65 | 13.6733 |
| Wider first layer | (128, 32) | 0.001 | 100 | 21.0614 |

Each experiment changed only one hyperparameter relative to the main configuration.

The best configuration was selected using validation performance, not test performance.

Generated reports are stored in the `reports/` directory.

## Limitations

- The dataset is simulated rather than collected from real bike-share stations.
- The model has not been evaluated on real-world rental demand.
- Only three neural network configurations were tested.
- Results may differ on other datasets.
- The network does not explicitly enforce nonnegative predictions.
- Additional experiments and longer training could improve performance.

## Project files

- `bikes_data.py` — Provided data generator
- `prepare.py` — Data splitting and preprocessing
- `model.py` — PyTorch neural network
- `training.py` — Training and evaluation loops
- `baselines.py` — Baseline comparisons
- `experiments.py` — Controlled experiments
- `train.py` — Full experiment runner
- `tests/` — Automated tests
- `reports/` — Generated metrics, CSV files, and plots
