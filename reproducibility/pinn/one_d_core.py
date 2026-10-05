"""Neural representation and diagnostic operators for checkpoint evaluation."""

from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trapz_compat(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        fn = np.trapz
    return fn(y, x, axis=axis)


@dataclass
class TrainingSets:
    x_data: np.ndarray
    t_data: np.ndarray
    y_data: np.ndarray
    x_r: np.ndarray
    t_r: np.ndarray
    x_ic: np.ndarray
    t_ic: np.ndarray
    y_ic: np.ndarray
    x_bc0: np.ndarray
    t_bc0: np.ndarray
    y_bc0: np.ndarray
    x_bc1: np.ndarray
    t_bc1: np.ndarray
    y_bc1: np.ndarray


def sample_training_sets(
    x: np.ndarray, t: np.ndarray, rho_num: np.ndarray, n_data: int, n_res: int, seed: int
) -> TrainingSets:
    """
    Sample interior observations, the stored initial snapshot and zero boundary targets.
    """
    rng = np.random.default_rng(seed)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    interior_mask = np.ones_like(xx, dtype=bool)
    interior_mask[0, :] = False
    interior_mask[:, 0] = False
    interior_mask[:, -1] = False
    x_data_all = xx[interior_mask].reshape(-1, 1)
    t_data_all = tt[interior_mask].reshape(-1, 1)
    y_data_all = rho_num[interior_mask].reshape(-1, 1)
    n_data_eff = min(n_data, x_data_all.shape[0])
    data_ids = rng.choice(x_data_all.shape[0], size=n_data_eff, replace=False)
    x_data = x_data_all[data_ids]
    t_data = t_data_all[data_ids]
    y_data = y_data_all[data_ids]
    x_r = rng.uniform(float(x[1]), float(x[-2]), size=(n_res, 1)).astype(np.float32)
    t_r = rng.uniform(float(t[1]), float(t[-1]), size=(n_res, 1)).astype(np.float32)
    x_ic = x.reshape(-1, 1).astype(np.float32)
    t_ic = np.zeros_like(x_ic, dtype=np.float32)
    y_ic = rho_num[0, :].reshape(-1, 1).astype(np.float32)
    t_bc = t.reshape(-1, 1).astype(np.float32)
    x_bc0 = np.zeros_like(t_bc, dtype=np.float32)
    y_bc0 = np.zeros_like(t_bc, dtype=np.float32)
    x_bc1 = np.ones_like(t_bc, dtype=np.float32)
    y_bc1 = np.zeros_like(t_bc, dtype=np.float32)
    return TrainingSets(
        x_data=x_data.astype(np.float32),
        t_data=t_data.astype(np.float32),
        y_data=y_data.astype(np.float32),
        x_r=x_r,
        t_r=t_r,
        x_ic=x_ic,
        t_ic=t_ic,
        y_ic=y_ic,
        x_bc0=x_bc0,
        t_bc0=t_bc,
        y_bc0=y_bc0,
        x_bc1=x_bc1,
        t_bc1=t_bc,
        y_bc1=y_bc1,
    )


class MLP(nn.Module):

    def __init__(
        self, layers: List[int], x_bounds: Tuple[float, float], t_bounds: Tuple[float, float]
    ) -> None:
        super().__init__()
        if len(layers) < 3:
            raise ValueError("Need at least input, one hidden, and output layer.")
        self.x_min, self.x_max = x_bounds
        self.t_min, self.t_max = t_bounds
        mods: List[nn.Module] = []
        for i in range(len(layers) - 2):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            mods.append(nn.Tanh())
        mods.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*mods)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)

    def _normalize(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_hat = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        t_hat = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return torch.cat([x_hat, t_hat], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = self._normalize(x, t)
        return self.net(z)


def to_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)
    tensor.requires_grad_(requires_grad)
    return tensor


def model_and_derivatives(
    model: nn.Module, x: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = model(x, t)
    ones = torch.ones_like(rho)
    rho_x = torch.autograd.grad(rho, x, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    rho_t = torch.autograd.grad(rho, t, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    rho_xx = torch.autograd.grad(
        rho_x, x, grad_outputs=torch.ones_like(rho_x), retain_graph=True, create_graph=True
    )[0]
    return (rho, rho_x, rho_t, rho_xx)


def pde_residual(
    model: nn.Module, x: torch.Tensor, t: torch.Tensor, a: float, nu: float
) -> torch.Tensor:
    rho, rho_x, rho_t, rho_xx = model_and_derivatives(model, x, t)
    return rho_t + a * rho_x - nu * rho_xx


def compute_loss_components(
    model: nn.Module, train: Dict[str, torch.Tensor], a: float, nu: float
) -> Dict[str, torch.Tensor]:
    pred_data = model(train["x_data"], train["t_data"])
    data_loss = torch.mean((pred_data - train["y_data"]) ** 2)
    r = pde_residual(model, train["x_r"], train["t_r"], a=a, nu=nu)
    pde_loss = torch.mean(r**2)
    pred_ic = model(train["x_ic"], train["t_ic"])
    ic_loss = torch.mean((pred_ic - train["y_ic"]) ** 2)
    pred_bc0 = model(train["x_bc0"], train["t_bc0"])
    pred_bc1 = model(train["x_bc1"], train["t_bc1"])
    bc_loss = 0.5 * (
        torch.mean((pred_bc0 - train["y_bc0"]) ** 2) + torch.mean((pred_bc1 - train["y_bc1"]) ** 2)
    )
    return {"data": data_loss, "pde": pde_loss, "ic": ic_loss, "bc": bc_loss}


def eval_on_points(
    model: nn.Module,
    x_np: np.ndarray,
    t_np: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    need_derivatives: bool = False,
) -> Dict[str, np.ndarray]:
    if x_np.shape != t_np.shape:
        raise ValueError("x_np and t_np must have the same shape.")
    n = x_np.shape[0]
    rho_list: List[np.ndarray] = []
    rho_x_list: List[np.ndarray] = []
    rho_t_list: List[np.ndarray] = []
    rho_xx_list: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xb = to_tensor(x_np[start:stop].reshape(-1, 1), device, requires_grad=need_derivatives)
        tb = to_tensor(t_np[start:stop].reshape(-1, 1), device, requires_grad=need_derivatives)
        if need_derivatives:
            rho, rho_x, rho_t, rho_xx = model_and_derivatives(model, xb, tb)
            rho_list.append(rho.detach().cpu().numpy().reshape(-1))
            rho_x_list.append(rho_x.detach().cpu().numpy().reshape(-1))
            rho_t_list.append(rho_t.detach().cpu().numpy().reshape(-1))
            rho_xx_list.append(rho_xx.detach().cpu().numpy().reshape(-1))
        else:
            with torch.no_grad():
                rho = model(xb, tb)
            rho_list.append(rho.detach().cpu().numpy().reshape(-1))
    out = {"rho": np.concatenate(rho_list, axis=0)}
    if need_derivatives:
        out["rho_x"] = np.concatenate(rho_x_list, axis=0)
        out["rho_t"] = np.concatenate(rho_t_list, axis=0)
        out["rho_xx"] = np.concatenate(rho_xx_list, axis=0)
    return out


@dataclass
class EvaluationResults:
    rho_pred: np.ndarray
    rho_x: np.ndarray
    rho_t: np.ndarray
    rho_xx: np.ndarray
    residual: np.ndarray
    data_error: np.ndarray
    actual_error: np.ndarray
    mass_balance: np.ndarray
    metrics: Dict[str, float]


def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    t: np.ndarray,
    rho_num: np.ndarray,
    rho_ana: np.ndarray,
    a: float,
    nu: float,
    train_time_s: float,
    device: torch.device,
) -> EvaluationResults:
    xx, tt = np.meshgrid(x, t, indexing="xy")
    eval_out = eval_on_points(
        model, x_np=xx.reshape(-1), t_np=tt.reshape(-1), device=device, need_derivatives=True
    )
    rho_pred = eval_out["rho"].reshape(t.size, x.size)
    rho_x = eval_out["rho_x"].reshape(t.size, x.size)
    rho_t = eval_out["rho_t"].reshape(t.size, x.size)
    rho_xx = eval_out["rho_xx"].reshape(t.size, x.size)
    residual = rho_t + a * rho_x - nu * rho_xx
    data_error = rho_pred - rho_num
    actual_error = rho_pred - rho_ana
    int_rho_t = trapz_compat(rho_t, x, axis=1)
    b0 = eval_on_points(model, x_np=np.zeros_like(t), t_np=t, device=device, need_derivatives=True)
    b1 = eval_on_points(model, x_np=np.ones_like(t), t_np=t, device=device, need_derivatives=True)
    rho_left = b0["rho"]
    rho_right = b1["rho"]
    rho_x_left = b0["rho_x"]
    rho_x_right = b1["rho_x"]
    mass_balance = int_rho_t + a * (rho_right - rho_left) - nu * (rho_x_right - rho_x_left)
    metrics = {
        "MSE_data": float(np.mean(data_error**2)),
        "MSE_ref": float(np.mean(actual_error**2)),
        "MSE_PDE": float(np.mean(residual**2)),
        "MSE_IC": float(np.mean((rho_pred[0, :] - rho_num[0, :]) ** 2)),
        "MSE_BC": float(0.5 * (np.mean(rho_pred[:, 0] ** 2) + np.mean(rho_pred[:, -1] ** 2))),
        "Ebal_RMS": float(np.sqrt(np.mean(mass_balance**2))),
        "final_time_L2_ref": float(
            np.sqrt(trapz_compat((rho_pred[-1, :] - rho_ana[-1, :]) ** 2, x))
        ),
        "final_time_Linf_ref": float(np.max(np.abs(rho_pred[-1, :] - rho_ana[-1, :]))),
        "train_time_seconds": float(train_time_s),
    }
    return EvaluationResults(
        rho_pred=rho_pred,
        rho_x=rho_x,
        rho_t=rho_t,
        rho_xx=rho_xx,
        residual=residual,
        data_error=data_error,
        actual_error=actual_error,
        mass_balance=mass_balance,
        metrics=metrics,
    )
