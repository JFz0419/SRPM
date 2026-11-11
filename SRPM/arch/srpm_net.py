
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):


    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x)))) + x


class ResidualMLP(nn.Module):


    def __init__(self, in_dim: int, hid: int, depth: int, dropout: float = 0.15):
        super().__init__()
        self.emb = nn.Linear(in_dim, hid)
        self.encoder = nn.Sequential(
            *[ResidualBlock(hid, dropout) for _ in range(int(depth))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.emb(x))


class SRPMNet(nn.Module):
    

    def __init__(self, num_nodes: int, period: int, input_len: int = 12,
                 output_len: int = 12, hid: int = 192, depth: int = 3,
                 rank: int = 2, dropout: float = 0.15,
                 mu_g: float = 0.0, sd_g: float = 1.0,
                 s_init=None, alpha_init=None, d_init=None, z_init=None):
        super().__init__()
        self.N, self.P = int(num_nodes), int(period)
        self.L, self.H = int(input_len), int(output_len)
        self.rank = int(rank)
        self.hid = int(hid)
        self.mu_g, self.sd_g = float(mu_g), float(sd_g)

        if s_init is None:
            s_init = torch.ones(7, self.P)
        self.template = nn.Parameter(torch.as_tensor(s_init, dtype=torch.float32).clone())

        if self.rank > 0:
            if d_init is None or z_init is None:
                raise ValueError("rank>0 时必须同时提供 d_init 与 z_init")
            self.D = nn.Parameter(torch.as_tensor(d_init, dtype=torch.float32).clone())
            self.z = nn.Parameter(torch.as_tensor(z_init, dtype=torch.float32).clone())

        if alpha_init is None:
            alpha_init = torch.ones(self.N)
        b0 = -self.mu_g / self.sd_g
        self.log_alpha = nn.Parameter(
            torch.log(torch.as_tensor(alpha_init, dtype=torch.float32).clone()))
        self.beta = nn.Parameter(torch.full((self.N,), b0))

        self.kappa = nn.Parameter(torch.tensor(
            [0.90 + (0.72 - 0.90) * i / max(self.H - 1, 1) for i in range(self.H)],
            dtype=torch.float32))

        self.log_sig_c = nn.Parameter(torch.zeros((), dtype=torch.float32))

        self.fdim = self.L + 5 + 2 + self.H + 1        # 12+5+2+12+1 = 32
        self.mlp = ResidualMLP(self.fdim, self.hid, depth, dropout)
        self.head = nn.Linear(self.hid, self.H)        # 均值残差
        self.head_s = nn.Linear(self.hid, 1)           # 窗口尺度 s
        for lin in (self.head, self.head_s):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    # ---------------------------------------------------------------- 周期基
    def _base(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        s = self.template[dow, tod]                                    # (B,T)
        base = torch.exp(self.log_alpha).view(1, 1, -1) * s.unsqueeze(-1) \
            + self.beta.view(1, 1, -1)                                 # (B,T,N)
        if self.rank > 0:
            d = self.D[:, dow, tod]                                    # (r,B,T)
            base = base + torch.einsum('rbt,nr->btn', d, self.z)
        return base

    def reg(self) -> torch.Tensor:
        t = self.template
        d2 = torch.roll(t, -1, 1) - 2 * t + torch.roll(t, 1, 1)
        dd = torch.roll(t, -1, 0) - t
        return (d2 ** 2).mean() + 0.5 * (dd ** 2).mean()

    # ------------------------------------------------------------------ 前向
    def forward(self, x_hist: torch.Tensor, tod_w: torch.Tensor, dow_w: torch.Tensor,
                tod_f: torch.Tensor, dow_f: torch.Tensor):
        
        base_w = self._base(tod_w, dow_w)                              # (B,L,N)
        base_f = self._base(tod_f, dow_f)                              # (B,H,N)


        r = x_hist - base_w                                            # (B,L,N)
        lvl = r.mean(1)                                                # (B,N)
        mu = base_f + self.kappa.view(1, -1, 1) * lvl.unsqueeze(1)     # (B,H,N)


        a = torch.exp(self.log_alpha).clamp(min=0.05).view(1, -1)
        rp = r.permute(0, 2, 1)                                        # (B,N,L)
        rn = rp / a.unsqueeze(-1)
        slope = (rp[:, :, -1] - rp[:, :, 0]) / a
        rstd = rp.std(2) / a
        lvl_n = lvl / a
        xmean = x_hist.mean(1) / a
        two_pi = 2 * math.pi
        phase = torch.stack([torch.sin(tod_w[:, 0].float() * two_pi / self.P),
                             torch.cos(tod_w[:, 0].float() * two_pi / self.P)], -1)
        s_f = self.template[dow_f, tod_f]                              # (B,H)
        s_w = self.template[dow_w, tod_w]                              # (B,L)
        s_f_n = s_f / s_w.mean(1, keepdim=True).clamp(min=1e-3)
        lvl_rel = lvl / (s_w.mean(1, keepdim=True) * a).clamp(min=1e-3)

        feats = torch.cat([
            rn,
            lvl_n.unsqueeze(-1), slope.unsqueeze(-1), rstd.unsqueeze(-1),
            xmean.unsqueeze(-1),
            self.log_alpha.view(1, -1, 1).expand(r.shape[0], -1, 1),
            phase.unsqueeze(1).expand(-1, self.N, -1),
            s_f_n.unsqueeze(1).expand(-1, self.N, -1),
            lvl_rel.unsqueeze(-1)], -1)                                # (B,N,32)

        h = self.mlp(feats)                                            # (B,N,hid)
        mu = mu + self.head(h).permute(0, 2, 1)                        # 均值残差修正


        lsig = self.log_sig_c + self.head_s(h).view(mu.shape[0], 1, self.N)
        sigma = torch.exp(lsig.clamp(min=-3.0, max=6.0))

        return mu, sigma, {'lvl_rel': lvl_rel, 'reg': self.reg()}
