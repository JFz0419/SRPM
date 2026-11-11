
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .srpm_net import SRPMNet


def time_slot(value: torch.Tensor, size: int) -> torch.Tensor:
    """占比 → 整数档位：``(value * size).long()`` 后 clamp（防御浮点噪声）。"""
    return (value * size).long().clamp_(0, size - 1)


class SRPM(nn.Module):

    def __init__(self, num_nodes: int, input_len: int = 12, output_len: int = 12,
                 period: int = 288, init_path: str = None, hid: int = 192,
                 depth: int = 3, rank: int = 2, mlp_dropout: float = 0.15,
                 **kwargs):
        super().__init__()
        if init_path is None:
            raise ValueError("SRPM 需要 init_path：先跑 "
                             "python prob_baselines/SRPM/build_init.py --dataset <DS>")
        z = np.load(init_path, allow_pickle=True)
        if int(z['num_nodes']) != int(num_nodes):
            raise ValueError(f"初始化文件的节点数 {int(z['num_nodes'])} "
                             f"与配置的 {num_nodes} 不符（{init_path}）")
        if int(z['period']) != int(period):
            raise ValueError(f"初始化文件的一天档数 {int(z['period'])} "
                             f"与 desc.json 算出的 {period} 不符（{init_path}）")
        rank = int(min(rank, z['z_init'].shape[1]))
        self.init_path = init_path
        self.num_nodes = int(num_nodes)
        self.period = int(period)
        self.net = SRPMNet(
            num_nodes=int(num_nodes), period=int(period),
            input_len=int(input_len), output_len=int(output_len),
            hid=int(hid), depth=int(depth), rank=rank,
            dropout=float(mlp_dropout),
            mu_g=float(z['mu_g']), sd_g=float(z['sd_g']),
            s_init=z['s_init'], alpha_init=z['alpha_init'],
            d_init=z['d_init'] if rank > 0 else None,
            z_init=z['z_init'] if rank > 0 else None,
        )
        print(f'[SRPM] init={init_path}  N={num_nodes} P={period} rank={rank} '
              f'mu_g={float(z["mu_g"]):.4f} sd_g={float(z["sd_g"]):.4f}', flush=True)

    # ------------------------------------------------------------------ 时刻
    def _slots(self, data: torch.Tensor):

        tod = time_slot(data[:, :, 0, 1], self.period)
        dow = time_slot(data[:, :, 0, 2], 7)
        return tod, dow

    # ------------------------------------------------------------------ 前向
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen=None, epoch=None, train: bool = True, **kwargs):
        if history_data.shape[-1] < 3:
            raise ValueError(
                "SRPM 的 tod/dow 档位索引需要 3 个通道 [flow, tod, dow]，"
                f"收到 {history_data.shape[-1]} 个（检查 CFG.MODEL.FORWARD_FEATURES）")
        x_hist = history_data[..., 0]                        # (B,L,N) 已标准化
        tod_w, dow_w = self._slots(history_data)
        tod_f, dow_f = self._slots(future_data)

        mu, sigma, aux = self.net(x_hist, tod_w, dow_w, tod_f, dow_f)
        return {
            'prediction': mu.unsqueeze(-1),                   # (B,H,N,1) 标准化空间
            'prediction_sigma': sigma.unsqueeze(-1),          # (B,H,N,1) 标准化空间
            'srpm_reg': aux['reg'],
            'srpm_lvl_rel': aux['lvl_rel'].detach(),
        }

    # ------------------------------------------------------------ 便捷入口
    @torch.no_grad()
    def predict_distribution(self, history_data: torch.Tensor) -> dict:

        H = self.net.H
        future = history_data[:, -1:].repeat(1, H, 1, 1).clone()   # (B,H,N,C)
        k = torch.arange(1, H + 1, device=history_data.device, dtype=history_data.dtype)
        tod0 = history_data[:, -1, 0, 1:2]                          # (B,1)
        dow0 = history_data[:, -1, 0, 2:3]
        future[:, :, :, 1] = ((tod0 + k / self.period) % 1.0).unsqueeze(-1)
        future[:, :, :, 2] = ((dow0 + k / (7 * self.period)) % 1.0).unsqueeze(-1)
        return self.forward(history_data, future, train=False)
