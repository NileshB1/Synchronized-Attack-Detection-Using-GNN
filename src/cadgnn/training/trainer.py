"""Training loop for the coordination model, with early stopping on validation loss."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from cadgnn.config import TrainingConfig
from cadgnn.fusion import CoordinationModel, coordination_loss, positive_weight
from cadgnn.training.balance import balance_indices

log = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    best_epoch: int
    best_val_loss: float
    pos_weight: float
    history: list[dict[str, float]] = field(default_factory=list)


def train_model(
    model: CoordinationModel,
    train: Sequence[HeteroData],
    val: Sequence[HeteroData],
    config: TrainingConfig,
    seed: int,
    device: torch.device | str = "cpu",
) -> TrainingResult:
    """Fit the scaler and class weight on ``train``, then train with Adam.

    ``train`` is balanced first as ``config.balancing`` says; ``val`` never is.
    The weights from the epoch with the lowest validation loss are restored at
    the end. ``val`` is only used for that choice, never for gradient updates.
    Accuracy in the history counts a score of 0.5 or more as coordinated.
    """
    if not train or not val:
        raise ValueError("training and validation sets must not be empty")
    model.fit_scaler(train)
    if config.balancing == "class_weight":
        pos_weight = positive_weight(train)
    else:
        positive_weight(train)  # still checks that both classes are there
        pos_weight = torch.tensor(1.0)
        y = [int(g.y) for g in train]
        train = [train[i] for i in balance_indices(y, config.balancing, seed)]
    neg_weight = config.fp_cost
    model.to(device)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        list(train), batch_size=config.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(list(val), batch_size=config.batch_size)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_loss, best_epoch, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses, correct, seen = [], 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            logit = model(batch).logit
            loss = coordination_loss(logit, batch.y, pos_weight, neg_weight)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())
            correct += int(((logit.detach() >= 0) == (batch.y.view(-1) == 1)).sum())
            seen += batch.num_graphs

        val_loss, val_acc = evaluate(model, val_loader, pos_weight, device, neg_weight)
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss,
            "train_acc": correct / seen, "val_acc": val_acc,
        })  # fmt: skip
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            log.info("stopping early at epoch %d, best was %d", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    log.info("best validation loss %.4f at epoch %d", best_loss, best_epoch)
    return TrainingResult(best_epoch, best_loss, float(pos_weight), history)


@torch.no_grad()
def evaluate(
    model: CoordinationModel,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    device,
    neg_weight: float = 1.0,
) -> tuple[float, float]:
    """Mean loss and accuracy over the loader."""
    model.eval()
    total, correct, count = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch).logit
        loss = coordination_loss(logit, batch.y, pos_weight, neg_weight)
        total += loss.item() * batch.num_graphs
        correct += int(((logit >= 0) == (batch.y.view(-1) == 1)).sum())
        count += batch.num_graphs
    return total / count, correct / count


def evaluate_loss(
    model: CoordinationModel, loader: DataLoader, pos_weight: torch.Tensor, device
) -> float:
    return evaluate(model, loader, pos_weight, device)[0]
