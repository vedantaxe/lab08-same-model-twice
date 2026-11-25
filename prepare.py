"""Prepare bike rental data for PyTorch training."""

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC_FEATURES = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["station_type"]


class BikeRentalDataset(Dataset):
    """Return individual feature and target tensors."""

    def __init__(self, features, targets):
        self.features = torch.tensor(
            np.asarray(features, dtype=np.float32),
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            np.asarray(targets, dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def split_data(frame, seed=2026):
    """Split data into 60% training, 20% validation and 20% test."""
    train_frame, remaining = train_test_split(
        frame,
        test_size=0.4,
        random_state=seed,
        shuffle=True,
    )

    validation_frame, test_frame = train_test_split(
        remaining,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )

    return train_frame, validation_frame, test_frame


def prepare_data(seed=2026):
    """Fit preprocessing on training rows only."""
    frame = make_rentals(seed=seed)

    train_frame, validation_frame, test_frame = split_data(frame, seed=seed)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    x_train = preprocessor.fit_transform(train_frame.drop(columns=[TARGET]))
    x_validation = preprocessor.transform(validation_frame.drop(columns=[TARGET]))
    x_test = preprocessor.transform(test_frame.drop(columns=[TARGET]))

    y_train = train_frame[TARGET].to_numpy(dtype=np.float32)
    y_validation = validation_frame[TARGET].to_numpy(dtype=np.float32)
    y_test = test_frame[TARGET].to_numpy(dtype=np.float32)

    return {
        "train": BikeRentalDataset(x_train, y_train),
        "validation": BikeRentalDataset(x_validation, y_validation),
        "test": BikeRentalDataset(x_test, y_test),
        "preprocessor": preprocessor,
        "frames": (train_frame, validation_frame, test_frame),
    }


def create_loaders(data, batch_size=64, seed=2026):
    """Create repeatable DataLoaders; shuffle training only."""
    generator = torch.Generator().manual_seed(seed)

    return {
        "train": DataLoader(
            data["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        "validation": DataLoader(
            data["validation"],
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            data["test"],
            batch_size=batch_size,
            shuffle=False,
        ),
    }


if __name__ == "__main__":
    prepared = prepare_data()
    loaders = create_loaders(prepared)

    for name in ("train", "validation", "test"):
        dataset = prepared[name]
        print(f"{name}: {len(dataset)} rows")

    features, targets = next(iter(loaders["train"]))

    print("Batch features:", features.shape, features.dtype)
    print("Batch targets:", targets.shape, targets.dtype)
