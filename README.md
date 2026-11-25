# Lab 08 — The Same Model, Twice

## Overview

This project compares equivalent PyTorch and Keras neural networks for predicting bike rental demand. It extends the Lab 07 PyTorch implementation with a Keras model, framework equivalence checks, reproducible training experiments, automated tests, and a comparison report.

## Model Architecture

Both frameworks use the same feedforward neural network:

- Input: 8 features
- First hidden layer: 64 neurons, ReLU activation
- Second hidden layer: 32 neurons, ReLU activation
- Output: 1 neuron with linear activation
- Total trainable parameters: 2,689

Training uses Adam optimization, a learning rate of 0.001, a batch size of 64, and up to 100 epochs.

## Setup

Create and activate the Conda environment:

```powershell
conda env create -f environment.yml
conda activate lab08-same-model-twice
```

## Running the Project

Train the original PyTorch implementation:

```powershell
python train.py
```

Train the Keras implementation:

```powershell
python train_keras.py
```

Compare both frameworks across three random seeds:

```powershell
python compare.py
```

The comparison uses seeds 2026, 2027, and 2028.

## Testing and Code Quality

Run all automated tests:

```powershell
python -m pytest -q
```

Check linting:

```powershell
python -m ruff check .
```

Check formatting:

```powershell
python -m ruff format --check .
```

The test suite covers PyTorch training behavior, Keras architecture, parameter counts, equivalent predictions after weight copying, reproducibility, model serialization, and memorization of a small dataset.

## Results

| Metric | PyTorch | Keras |
|---|---:|---:|
| Average validation MAE | 3.5231 | 3.5605 |
| Average test MAE | 3.5504 | 3.5364 |
| Average training time | 18.92 s | 30.38 s |

Both frameworks demonstrated comparable predictive accuracy. PyTorch trained faster in these experiments, while Keras provided a convenient high-level training interface.

For the complete interpretation and recommendation, see `COMPARISON.md`.

## Key Files

| File | Purpose |
|---|---|
| `model.py` | PyTorch network architecture |
| `keras_model.py` | Equivalent Keras network and weight-copying utilities |
| `prepare.py` | Data generation and preprocessing |
| `training.py` | PyTorch training functions |
| `train.py` | Original PyTorch training workflow |
| `train_keras.py` | Keras training workflow |
| `compare.py` | Three-seed framework comparison |
| `tests/` | Automated tests |
| `COMPARISON.md` | Experimental findings and recommendation |
| `reports/comparison.csv` | Per-seed comparison results |
| `reports/keras_history.csv` | Keras training history |
| `reports/keras_learning_curves.png` | Keras learning curves |
| `reports/best_keras_model.keras` | Saved Keras model |

## Conclusion

The experiment demonstrates that equivalent neural networks can be implemented in different frameworks while producing similar predictive performance. Framework selection should consider training efficiency, maintainability, team familiarity, and deployment requirements.