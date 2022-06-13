"""functorch implementations of torch loss functions which are not implemented in functorch yet.
Copied verbatim from: https://github.com/orobix/fwdgrad"""

import torch
from typing import Callable, Tuple
import torch.nn.functional as F


def functional_xent(
    params: Tuple[torch.nn.Parameter, ...],
    fmodel: Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    y = fmodel(params, x)
    return _xent(y, t, num_classes)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax function.
    Args:
        x (torch.Tensor): tensor over which to apply softmax.
        dim (int, optional): dimension over which to apply softmax. Defaults to 1.
    Returns:
        torch.Tensor: softmax of x.
    """
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return torch.div(x_exp, x_exp_sum)

def clamp_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)

def _xent(x: torch.Tensor, t: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Compute cross-entropy loss.
    Args:
        x (torch.Tensor): Output of the model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.
    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = clamp_probs(softmax(x))
    logy = -torch.log(y)
    loss = torch.mean(torch.sum(logy * F.one_hot(t.to(torch.int64), num_classes), dim=1))
    return loss
  