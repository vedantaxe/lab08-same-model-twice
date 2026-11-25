"""Tests for the bike rental neural network pipeline."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from bikes_data import make_rentals
from model import BikeRentalNetwork
from prepare import (
    BikeRentalDataset,
    create_loaders,
    prepare_data,
    split_data,
)
from training import evaluate_one_epoch, fit_model, train_one_epoch


def test_split_sizes():
    frame = make_rentals()
    train, validation, test = split_data(frame)

    assert len(train) == 3600
    assert len(validation) == 1200
    assert len(test) == 1200


def test_split_has_no_overlap():
    frame = make_rentals()
    train, validation, test = split_data(frame)

    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)


def test_split_is_reproducible():
    frame = make_rentals()

    first = split_data(frame, seed=2026)
    second = split_data(frame, seed=2026)

    for first_frame, second_frame in zip(first, second, strict=True):
        assert first_frame.index.tolist() == second_frame.index.tolist()


def test_scaler_is_fitted_on_training_only():
    data = prepare_data()
    train_frame = data["frames"][0]

    scaler = data["preprocessor"].named_transformers_["numeric"]
    expected = train_frame[
        [
            "hour",
            "temperature_c",
            "rain_mm",
            "wind_kph",
            "is_weekend",
        ]
    ].mean()

    np.testing.assert_allclose(
        scaler.mean_,
        expected.to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_dataset_uses_float32():
    data = prepare_data()

    features, target = data["train"][0]

    assert features.dtype == torch.float32
    assert target.dtype == torch.float32


@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_dataloader_batch_size(batch_size):
    data = prepare_data()
    loaders = create_loaders(data, batch_size=batch_size)

    features, targets = next(iter(loaders["train"]))

    assert features.shape[0] == batch_size
    assert targets.shape[0] == batch_size


def test_network_output_shape():
    model = BikeRentalNetwork(input_size=8)
    features = torch.randn(12, 8)

    predictions = model(features)

    assert predictions.shape == (12,)


def test_optimizer_updates_weights():
    torch.manual_seed(2026)

    model = BikeRentalNetwork(input_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    features = torch.randn(16, 8)
    targets = torch.randn(16)
    loader = DataLoader(
        BikeRentalDataset(features, targets),
        batch_size=8,
    )

    before = deepcopy(model.state_dict())

    train_one_epoch(model, loader, optimizer, loss_fn)

    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.state_dict().items()
    )


def test_evaluation_does_not_update_weights():
    model = BikeRentalNetwork(input_size=8)
    loss_fn = nn.MSELoss()

    features = torch.randn(16, 8)
    targets = torch.randn(16)
    loader = DataLoader(
        BikeRentalDataset(features, targets),
        batch_size=8,
    )

    before = deepcopy(model.state_dict())

    evaluate_one_epoch(model, loader, loss_fn)

    for name, parameter in model.state_dict().items():
        assert torch.equal(before[name], parameter)


def test_model_can_memorize_small_dataset():
    torch.manual_seed(2026)

    features = torch.randn(8, 8)
    targets = torch.ones(8) * 5

    loader = DataLoader(
        BikeRentalDataset(features, targets),
        batch_size=8,
    )

    model = BikeRentalNetwork(input_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    initial_loss = evaluate_one_epoch(model, loader, loss_fn)

    for _ in range(200):
        train_one_epoch(model, loader, optimizer, loss_fn)

    final_loss = evaluate_one_epoch(model, loader, loss_fn)

    assert final_loss < initial_loss * 0.1


def test_best_checkpoint_is_restored():
    torch.manual_seed(2026)

    features = torch.randn(32, 8)
    targets = torch.randn(32)

    loader = DataLoader(
        BikeRentalDataset(features, targets),
        batch_size=16,
    )

    model = BikeRentalNetwork(input_size=8)

    result = fit_model(
        model,
        loader,
        loader,
        epochs=5,
        learning_rate=0.001,
    )

    restored_loss = evaluate_one_epoch(
        model,
        loader,
        nn.MSELoss(),
    )

    assert restored_loss == pytest.approx(
        result["best_validation_loss"],
        rel=1e-5,
    )
