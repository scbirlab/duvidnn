

from torch import Tensor, nn


class L1Regularizer:
    """L1 penalty over model parameters."""

    def __init__(
        self,
        weight: float = 1e-4,
        bias: bool = False,
    ) -> None:
        if weight < 0.:
            raise ValueError(
                f"`weight` must be non-negative, but was {weight}."
            )

        self.weight = float(weight)
        self.bias = bias

    def __call__(
        self,
        model: nn.Module,
    ) -> Tensor:
        penalty = None

        for name, parameter in model.named_parameters():
            if (
                not self.bias
                and name.endswith("bias")
            ):
                continue

            value = parameter.abs().sum()

            penalty = (
                value
                if penalty is None
                else penalty + value
            )

        if penalty is None:
            parameter = next(
                model.parameters(),
                None,
            )

            if parameter is None:
                raise ValueError(
                    "Cannot regularize a model with no parameters."
                )

            penalty = parameter.new_zeros(())

        return self.weight * penalty
