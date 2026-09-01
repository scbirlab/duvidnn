

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base import PhysicalModel


def hill_curve(
    conc: Tensor,
    log_ic50: Tensor,
    slope: Tensor = 1.,
    efficacy: Tensor = 1.,
    bottom: Tensor | None = None,
    top: Tensor = 1.,
    is_growth: bool = False
) -> Tensor:
    if torch.any(conc <= 0):
        raise ValueError(f"Concentration must be positive, but was {conc}.")
    if bottom is None:
        bottom = top * (1. - efficacy)
    inhibition = bottom + (top - bottom) * torch.sigmoid(
        slope * (torch.log(conc) - log_ic50)
    )
    if is_growth:
        return top - inhibition
    else:
        return inhibition


# class HillCurve(nn.Module):
#     """Map latent potency parameters and concentration to fractional response.

#     Parameters
#     ==========
#     slope
#         Shared Hill slope. If ``None``, the latent representation must contain
#         both log-IC50 and an unconstrained slope parameter.
#     trainable
#         Whether a shared slope should be trainable.

#     Notes
#     =====
#     The first latent value is interpreted as ln(IC50).

#     With ``slope=None`` the second latent value is interpreted as an
#     unconstrained compound-specific slope and transformed with softplus.

#     ``context`` is concentration in the same units as IC50.
#     """

#     def __init__(
#         self,
#         slope: float | None = None,
#         trainable_slope: bool = False
#     ) -> None:
#         super().__init__()

#         if slope is None:
#             if trainable_slope:
#                 raise ValueError(
#                     "`trainable_slope=True` requires an initial `slope`."
#                 )

#             self.raw_slope = None
#             self.register_buffer(
#                 "fixed_slope",
#                 None,
#             )

#         elif slope <= 0:
#             raise ValueError(
#                 f"`slope` must be positive, but was {slope}."
#             )

#         elif trainable_slope:
#             raw_slope = torch.tensor(float(slope))
#             self.raw_slope = nn.Parameter(  # inv softplus
#                 torch.log(torch.expm1(raw_slope))
#             )
#             self.register_buffer(
#                 "fixed_slope",
#                 None,
#             )

#         else:
#             self.raw_slope = None
#             self.register_buffer(
#                 "fixed_slope",
#                 torch.tensor(float(slope)),
#             )

#     def forward(
#         self,
#         latent: Tensor,
#         context: Tensor
#     ) -> Tensor:
#         log_ic50 = latent[..., 0]

#         if self.raw_slope is not None:
#             slope = F.softplus(self.raw_slope)

#         elif self.fixed_slope is not None:
#             slope = self.fixed_slope

#         else:
#             if latent.shape[-1] != 2:
#                 raise ValueError(
#                     "HillReadout with compound-specific slope expects "
#                     "latent[..., 2] = [log_ic50, raw_slope]."
#                 )

#             slope = F.softplus(
#                 latent[..., 1]
#             )

#         concentration = context.squeeze(-1)
#         response = hill_curve(
#             conc=concentration,
#             log_ic50=log_ic50,
#             slope=slope,
#         )
#         return response.unsqueeze(-1)


class HillCurve(PhysicalModel):
    """Map latent potency parameters and concentration to fractional response.

    Parameters
    ==========
    slope
        Shared Hill slope. If ``None``, the latent representation must contain
        both log-IC50 and an unconstrained slope parameter.
    trainable
        Whether a shared slope should be trainable.

    Notes
    =====
    The first latent value is interpreted as ln(IC50).

    With ``slope=None`` the second latent value is interpreted as an
    unconstrained compound-specific slope and transformed with softplus.

    ``context`` is concentration in the same units as IC50.

    """

    def __init__(
        self,
        slope: float = 1.,
        trainable_slope: bool = False
    ):
        if slope <= 0.:
            raise ValueError(
                f"`slope` must be positive, but was {slope}."
            )
        slope = torch.tensor(float(slope))
        if trainable_slope:
            fixed_params = None
            trainable_params = {"slope": torch.log(torch.expm1(slope))}
            transforms = {"slope": F.softplus}
        else:
            fixed_params = {"slope": slope}
            trainable_params = None
            transforms = None

        super().__init__(
            fn=hill_curve,
            fixed_params=fixed_params,
            trainable_params=trainable_params,
            latent_params="log_ic50",
            context_params="conc",
            transforms=transforms,
        )
