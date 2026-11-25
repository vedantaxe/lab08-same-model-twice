"""Training and evaluation loops for the bike rental network."""

from copy import deepcopy

import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, loss_fn):
    """Train the model for one complete epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        optimizer.zero_grad()

        predictions = model(features)
        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate_one_epoch(model, loader, loss_fn):
    """Evaluate without updating model parameters."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features)
            loss = loss_fn(predictions, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def fit_model(
    model,
    train_loader,
    validation_loader,
    epochs=100,
    learning_rate=0.001,
):
    """Train and restore weights from the best validation epoch."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    loss_fn = nn.MSELoss()

    history = []
    best_validation_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        training_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)

        validation_loss = evaluate_one_epoch(model, validation_loader, loss_fn)

        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_weights)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
