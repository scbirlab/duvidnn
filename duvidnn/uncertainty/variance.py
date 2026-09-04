from dataclasses import dataclass


@dataclass
class Variance:
    dim: int = -1

    name = "variance"

    def prepare(
        self,
        box,
        **kwargs,
    ):
        return None

    def __call__(
        self,
        *,
        prediction,
        box=None,
        batch=None,
        state=None,
    ):
        import torch

        if self.dim < 0:
            self.dim += prediction.ndim

        if (
            self.dim < 0
            or self.dim >= prediction.ndim
        ):
            raise ValueError(
                f"Variance dimension {self.dim} "
                f"is invalid for prediction shape "
                f"{tuple(prediction.shape)}."
            )

        if prediction.shape[self.dim] <= 1:
            raise ValueError(
                "Variance requires the selected prediction "
                "dimension to contain more than one value. "
                f"Received shape {tuple(prediction.shape)} "
                f"and dim={self.dim}."
            )

        return torch.var(
            prediction,
            dim=self.dim,
            correction=0,
        )