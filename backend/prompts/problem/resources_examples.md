# [Example 1: DTLZ1 (Decomposition-based Test Problem)]
import torch
from evox.core import Problem
from evox.operators.sampling import uniform_sampling

class DTLZ1(Problem):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        self.sample, _ = uniform_sampling(self.ref_num * self.m, self.m)
        self.device = self.sample.device

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part
        return f

    def pf(self):
        f = self.sample / 2
        return f


# [Example 2: ZDT1 (Bimodal Test Problem)]
import torch
from evox.core import Problem

class ZDT1(Problem):
    def __init__(self, d: int = 30, m: int = 2):
        super().__init__()
        self.d = d
        self.m = m

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        f1 = X[:, 0]
        g = 1 + 9 * torch.mean(X[:, 1:], dim=1)
        f2 = g * (1 - torch.sqrt(f1 / g))
        return torch.stack([f1, f2], dim=1)

    def pf(self):
        f1 = torch.linspace(0, 1, 100, device=torch.get_default_device())
        f2 = 1 - torch.sqrt(f1)
        return torch.stack([f1, f2], dim=1)