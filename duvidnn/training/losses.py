import torch
from torch import Tensor, nn


class CensoredMSELoss(nn.Module):
    """Squared error respecting left- and right-censored targets.

    Censor values are:

    - ``-1``: true value is <= target
    - ``0``: exact observation
    - ``1``: true value is >= target
    """

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        censor: Tensor | None = None
    ) -> Tensor:

        error = prediction - target
        if censor is None:
            censor = torch.zeros_like(error)

        exact = censor == 0
        less_than = censor < 0
        greater_than = censor > 0

        loss = torch.where(
            exact,
            error.square(),
            torch.zeros_like(error),
        )

        loss = loss + torch.where(
            less_than,
            torch.relu(error).square(),
            torch.zeros_like(error),
        )

        loss = loss + torch.where(
            greater_than,
            torch.relu(-error).square(),
            torch.zeros_like(error),
        )
        return loss.mean()
