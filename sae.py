import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import einsum, reduce, rearrange
from math import sqrt


def rectangle(x: Tensor) -> Tensor:
    return ((x > -0.5) & (x < 0.5)).type_as(x)


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x * (x > threshold)).type_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = grad_output.clone()
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class SAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_features: int,
        bandwidth: float,
        threshold: float,
        lambda_p: float,
        c: float,
    ):
        super().__init__()
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.lambda_p = lambda_p
        self.c = c

        self.w_d = nn.Parameter(torch.empty(d_model, n_features))
        nn.init.uniform_(self.w_d.data, -1 / sqrt(d_model), 1 / sqrt(d_model))
        self.w_e = nn.Parameter((d_model / n_features) * self.w_d.data.T)

        self.b_d = nn.Parameter(torch.zeros(d_model))
        self.b_e = nn.Parameter(torch.empty(n_features))

        self.t = nn.Parameter(torch.full((n_features,), threshold))

    def init_be(self, batch: Tensor):
        with torch.no_grad():
            acts = einsum(self.w_e, batch, "f d, t d -> t f")
            q = torch.quantile(acts.float(), 1 - 10_000 / self.n_features, dim=0)
            self.b_e.data = (torch.exp(self.t) - q).type_as(batch)

    def encode(self, x: Tensor) -> Tensor:
        acts = einsum(self.w_e, x, "f d, t d -> t f") + self.b_e
        acts: Tensor = JumpReLU.apply(acts, torch.exp(self.t), self.bandwidth)  # type: ignore
        return acts

    def decode(self, acts: Tensor) -> Tensor:
        x_hat = einsum(self.w_d, acts, "d f, t f -> t d") + self.b_d
        return x_hat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        acts = self.encode(x)
        x_hat = self.decode(acts)
        return acts, x_hat

    def get_loss(
        self, x: Tensor, x_hat: Tensor, acts: Tensor, lambda_s: float
    ) -> Tensor:
        wd_norm = torch.norm(self.w_d, dim=0)
        l_mse = F.mse_loss(x_hat, x)
        l_s = (lambda_s * torch.tanh(self.c * acts.abs() * wd_norm)).sum(-1).mean()
        l_p = (
            (self.lambda_p * F.relu(torch.exp(self.t) - acts) * wd_norm).sum(-1).mean()
        )
        return l_mse + l_s + l_p
