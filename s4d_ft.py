
import math
import copy
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from einops import repeat



class DropoutNd(nn.Module):

    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask_shape = x.shape[:2] + (1,) * (x.ndim - 2) if self.tie else x.shape
        mask = (torch.rand(*mask_shape, device=x.device) >= self.p).float()
        return x * mask / (1.0 - self.p)



class S4DKernel(nn.Module):

    def __init__(self, d_model: int, cfr: float, cfi: float, N: int = 64,
                 dt_min: float = 0.0001, dt_max: float = 0.1,
                 lr: float = None, lr_dt: float = None, wd: float = None):
        super().__init__()
        H = d_model

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        self.register_ssm_param("log_dt", log_dt, wd=0, lr=lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2)) * cfr

        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H) * cfi

        self.register_ssm_param("log_A_real", log_A_real, lr=lr)
        self.register_ssm_param("A_imag", A_imag, lr=lr)

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        dtA = A * dt.unsqueeze(-1)

        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)

        C = C * (torch.exp(dtA) - 1.0) / A

        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register_ssm_param(self, name: str, tensor: torch.Tensor,
                           wd: float = 0, lr: float = None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim_cfg = {"weight_decay": wd}
            if lr is not None:
                optim_cfg["lr"] = lr
            setattr(getattr(self, name), "_optim", optim_cfg)



class S4DLayer(nn.Module):

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.0,
                 cfr: float = 1.0, cfi: float = 1.0, transposed: bool = True,
                 **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = d_model
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(d_model))

        self.kernel = S4DKernel(d_model, cfr, cfi, N=d_state, **kernel_args)

        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None]:
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        k = self.kernel(L=L)

        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]

        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)

        return y, None



class S4DFT(nn.Module):

    def __init__(self, d_input: int, d_output: int = 1, d_model: int = 128,
                 n_layers: int = 6, dropout: float = 0.12, cfg: dict = None,
                 prenorm: bool = False):
        super().__init__()
        self.prenorm = prenorm

        if cfg is None:
            cfg = {
                "lr_min": 0.0004, "lr": 0.0004, "d_state": 128,
                "min_dt": 0.001, "max_dt": 1.0, "lr_dt": 0.0,
                "cfr": 10.0, "cfi": 10.0, "wd": 0.0,
            }

        self.encoder = nn.Linear(d_input, d_model)

        pt_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if pt_version >= (1, 12):
            dropout_fn = nn.Dropout1d
        else:
            dropout_fn = nn.Dropout2d

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                S4DLayer(
                    d_model,
                    dropout=dropout,
                    transposed=True,
                    lr=min(cfg["lr_min"], cfg["lr"]),
                    d_state=cfg["d_state"],
                    dt_min=cfg["min_dt"],
                    dt_max=cfg["max_dt"],
                    lr_dt=cfg["lr_dt"],
                    cfr=cfg["cfr"],
                    cfi=cfg["cfi"],
                    wd=cfg["wd"],
                )
            )
            self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(dropout_fn(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)

        x = x.transpose(-1, -2)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z)

            z, _ = layer(z)
            z = dropout(z)
            x = z + x

            if not self.prenorm:
                x = norm(x)

        x = x.transpose(-1, -2)

        x = x.mean(dim=1)

        x = self.decoder(x)

        return x



class NSELoss(nn.Module):

    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                q_stds: torch.Tensor) -> torch.Tensor:
        squared_error = (y_pred - y_true) ** 2
        weights = 1.0 / (q_stds + self.eps) ** 2
        return torch.mean(weights * squared_error)



def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = obs.flatten(), sim.flatten()
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return float('nan')
    return 1.0 - np.sum((sim - obs) ** 2) / denom



def warmup_cosine_annealing_lr(epoch: int, warmup_epochs: int,
                                total_epochs: int, lr_start: float = 1.0,
                                lr_end: float = 0.0) -> float:
    if epoch < warmup_epochs:
        return lr_start * (epoch / max(warmup_epochs, 1))
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return lr_end + (lr_start - lr_end) * 0.5 * (1.0 + np.cos(np.pi * progress))


def setup_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    epochs: int, warmup_epochs: int):
    all_params = list(model.parameters())

    general_params = [p for p in all_params if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(general_params, lr=lr, weight_decay=weight_decay)

    hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
    unique_hps = [dict(s) for s in sorted(set(frozenset(hp.items()) for hp in hps))]

    for hp in unique_hps:
        params = [p for p in all_params if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    lr_lambda = lambda epoch: warmup_cosine_annealing_lr(
        epoch, warmup_epochs, epochs, lr_start=1.0, lr_end=0.0
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("=" * 60)
    print("Optimizer parameter groups:")
    for i, g in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in g['params'])
        print(f"  Group {i}: {len(g['params'])} tensors, "
              f"{n_params:,} params, lr={g.get('lr', 'default')}, "
              f"wd={g.get('weight_decay', 'default')}")
    print("=" * 60)

    return optimizer, scheduler



class SyntheticRainfallRunoffDataset(Dataset):

    def __init__(self, n_basins: int = 50, n_days: int = 3650,
                 seq_length: int = 365, n_dynamic: int = 5, n_static: int = 27):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_dynamic + n_static

        samples_x, samples_y, samples_qstd = [], [], []

        for _ in range(n_basins):
            dyn = np.random.randn(n_days, n_dynamic).astype(np.float32)
            static = np.random.randn(1, n_static).astype(np.float32)
            static_rep = np.repeat(static, n_days, axis=0)
            x_full = np.concatenate([dyn, static_rep], axis=1)

            precip = dyn[:, 0]
            cumsum = np.cumsum(precip)
            discharge = np.abs(np.sin(cumsum * 0.01)) * 2.0 + np.random.randn(n_days) * 0.1
            discharge = np.maximum(discharge, 0).astype(np.float32)

            q_std = np.std(discharge)

            for i in range(n_days - seq_length + 1):
                samples_x.append(x_full[i:i + seq_length])
                samples_y.append(discharge[i + seq_length - 1:i + seq_length])
                samples_qstd.append(q_std)

        self.x = torch.from_numpy(np.array(samples_x))
        self.y = torch.from_numpy(np.array(samples_y))
        self.q_stds = torch.tensor(samples_qstd, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.q_stds[idx]



def train_epoch(model, loader, optimizer, loss_fn, device, clip_value=1.0):
    model.train()
    total_loss = 0.0
    for x, y, q_std in loader:
        x, y, q_std = x.to(device), y.to(device), q_std.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y, q_std)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for x, y, q_std in loader:
        x, y, q_std = x.to(device), y.to(device), q_std.to(device)
        out = model(x)
        loss = loss_fn(out, y, q_std)
        total_loss += loss.item()
    return total_loss / len(loader)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="S4D-FT Reproduction")
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=128, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of S4D layers')
    parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.03, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.12, help='Dropout')
    parser.add_argument('--cfr', type=float, default=10.0, help='Freq tuning real')
    parser.add_argument('--cfi', type=float, default=10.0, help='Freq tuning imag')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=365, help='Sequence length')
    parser.add_argument('--seed', type=int, default=200, help='Random seed')
    parser.add_argument('--n_basins', type=int, default=20, help='Synthetic basins')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id (-1=cpu)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = {
        "lr_min": args.lr,
        "lr": args.lr,
        "d_state": args.d_state,
        "min_dt": 0.001,
        "max_dt": 1.0,
        "lr_dt": 0.0,
        "cfr": args.cfr,
        "cfi": args.cfi,
        "wd": 0.0,
    }

    d_input = 32
    print(f"\nCreating synthetic dataset ({args.n_basins} basins)...")
    dataset = SyntheticRainfallRunoffDataset(
        n_basins=args.n_basins,
        n_days=2000,
        seq_length=args.seq_length,
    )
    n_total = len(dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    model = S4DFT(
        d_input=d_input,
        d_output=1,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        cfg=cfg,
        prenorm=False,
    ).to(device)

    n_params = count_parameters(model)
    print(f"\nS4D-FT Model: {n_params:,} trainable parameters")
    print(f"  d_model={args.d_model}, d_state={args.d_state}, n_layers={args.n_layers}")
    print(f"  cfr={args.cfr}, cfi={args.cfi} (Frequency Tuning)")
    print(f"  dropout={args.dropout}, lr={args.lr}, wd={args.weight_decay}\n")

    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, warmup_epochs=0,
    )

    loss_fn = NSELoss(eps=0.1)

    print("\nTraining S4D-FT on synthetic data...")
    print("-" * 60)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"Train loss: {train_loss:.5f}  "
              f"Val loss: {val_loss:.5f}  "
              f"LR: {lr_current:.6f}")
        scheduler.step()

    print("\n" + "=" * 60)
    print("Shape verification:")
    x_test = torch.randn(4, 365, 32).to(device)
    with torch.no_grad():
        y_test = model(x_test)
    print(f"  Input:  {x_test.shape}  → Output: {y_test.shape}")
    assert y_test.shape == (4, 1), f"Expected (4, 1), got {y_test.shape}"
    print("  ✓ Shape check passed")
    print("=" * 60)

    print("\nS4D-FT reproduction verification complete.")
    print("To reproduce paper results, connect to the CAMELS dataset")
    print("following the instructions in the original repository.")


if __name__ == "__main__":
    main()
