
import torch
import torch.nn as nn
import torch.nn.functional as F

from s4d_ft import S4DFT
from data_pipeline import SCALER

_P_MEAN = float(SCALER["input_means"][0])
_P_STD = float(SCALER["input_stds"][0])
_Q_MEAN = float(SCALER["output_mean"][0])
_Q_STD = float(SCALER["output_std"][0])


class S4DFT_MC(S4DFT):

    def __init__(self, d_input: int = 32, d_output: int = 1,
                 d_model: int = 128, n_layers: int = 6,
                 dropout: float = 0.12, cfg: dict = None,
                 prenorm: bool = False, k_max: float = 0.1):
        super().__init__(
            d_input=d_input, d_output=d_output, d_model=d_model,
            n_layers=n_layers, dropout=dropout, cfg=cfg, prenorm=prenorm,
        )
        self.k_max = k_max

        self.et_head = nn.Linear(d_model, 1)
        self.runoff_head = nn.Linear(d_model, 1)
        self.k_head = nn.Linear(d_model, 1)

        for head in [self.et_head, self.runoff_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        nn.init.zeros_(self.k_head.weight)
        nn.init.constant_(self.k_head.bias, -2.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = self.encoder(u)
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
        h = x.transpose(-1, -2)

        p_phys = torch.relu(u[:, :, 0:1] * _P_STD + _P_MEAN)
        f_et = torch.sigmoid(self.et_head(h))
        f_runoff = torch.sigmoid(self.runoff_head(h))
        k = torch.sigmoid(self.k_head(h)) * self.k_max

        B, L, _ = h.shape
        S = torch.zeros(B, 1, device=u.device)

        for t in range(L):
            p_t = p_phys[:, t, :]
            et_t = f_et[:, t, :] * p_t
            p_eff = p_t - et_t
            q_fast = f_runoff[:, t, :] * p_eff
            k_t = k[:, t, :]
            q_base = k_t * S
            S = S * (1.0 - k_t) + (1.0 - f_runoff[:, t, :]) * p_eff
            S = S.relu()

        q_last = q_fast + q_base
        return (q_last - _Q_MEAN) / _Q_STD