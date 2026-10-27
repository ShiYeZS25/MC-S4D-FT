import torch
import torch.nn.functional as F

from data_pipeline import SCALER
from s4d_ft import S4DFT


class S4DFT_Softplus(S4DFT):
    def forward(self, x):
        out = super().forward(x)

        mean = float(SCALER["output_mean"][0])
        std = float(SCALER["output_std"][0])

        q = F.softplus(out * std + mean)
        out = (q - mean) / std

        zero_floor = torch.nextafter(
            out.new_tensor(-mean / std),
            out.new_tensor(float("inf")),
        )
        return out.clamp_min(zero_floor)