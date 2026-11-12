from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData


def positive_weight(train_graphs: Sequence[HeteroData]) -> Tensor:
    """Ratio of coincidental to coordinated episodes in the training set.

    Used as pos_weight so the rarer coordinated class counts as much as the
    coincidental one in the loss. Fit it on the training graphs only.
    """
    labels = torch.cat([g.y.view(-1) for g in train_graphs]) if train_graphs else torch.empty(0)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"training set needs both classes, got {positives} coordinated "
            f"and {negatives} coincidental episodes"
        )
    return torch.tensor(negatives / positives)


def coordination_loss(
    logit: Tensor, y: Tensor, pos_weight: Tensor | None = None, neg_weight: float = 1.0
) -> Tensor:
    """Binary cross-entropy on the coordination logit; y is 1 for coordinated.

    ``neg_weight`` scales the loss of coincidental episodes. Above 1 it makes a
    false alarm cost more than a missed campaign (cost-sensitive learning).
    """
    y = y.view(-1)
    if logit.shape != y.shape:
        raise ValueError(f"logit shape {tuple(logit.shape)} does not match labels {tuple(y.shape)}")
    if not torch.isin(y, torch.tensor([0, 1], device=y.device)).all():
        raise ValueError("labels must be 0 (coincidental) or 1 (coordinated)")
    if pos_weight is not None:
        pos_weight = pos_weight.to(logit.device)
    target = y.to(logit.dtype)
    if neg_weight == 1.0:
        return F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight)
    loss = F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=pos_weight, reduction="none"
    )
    return (loss * torch.where(y == 0, neg_weight, 1.0)).mean()
