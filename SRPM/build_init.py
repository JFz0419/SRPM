"""为 SRPM 构造初始化文件 ``srpm_init_<dataset>.npz``。

为什么需要它
------------
SRPM 的周期模板、节点增益、低秩基**不是随机初始化**，而是在训练段的
「星期 × 时刻」张量上直接算出来的：

1. ``a_n``  = 训练段逐节点流量均值（增益，使秩-1 曲线可比）；
2. ``S[d, t]`` = 逐节点除以 ``a_n`` 后跨节点平均，再做 5 点平滑、归一化到均值 1
   （= 全网共享的周期曲线，秩-1 假设的直接产物）；
3. ``dev = Tn − a_n·S``，对它做 SVD，前 ``r`` 个奇异分量给 ``D_j`` 与 ``z_nj``
   （= ``a_n·S + b_n`` 之外的逐节点弥散偏差，主表 r=2）。

这些量必须从**训练段**算（绝不碰 valid/test），因此与 Sparkle 的 ``basis_*.npz``
一样落盘，由配置的 ``init_path`` 读取。同时落盘 ``mu_g`` / ``sd_g``：
σ 的换算与 ``beta`` 的初值都依赖 z-score 统计量。

用法（仓库根目录）::

    python prob_baselines/SRPM/build_init.py --dataset PEMS08
    python prob_baselines/SRPM/build_init.py --dataset Seattle --rank 2

口径与 ``basicts/scaler/ZScoreScaler`` 严格一致：``mu_g`` / ``sd_g`` 取
``data[:int(T*train_ratio), :, 0]`` 的全局均值/标准差（``NORM_EACH_CHANNEL=False``）。
"""
import argparse
import json
import os

import numpy as np


def build(dataset: str, rank: int = 2, train_ratio: float = 0.6) -> dict:
    desc = json.load(open(os.path.join('datasets', dataset, 'desc.json'),
                          encoding='utf-8'))
    T, N, C = desc['shape']
    P = 1440 // int(desc['frequency (minutes)'])          # 5 min → 288, 60 min → 24

    raw = np.memmap(os.path.join('datasets', dataset, 'data.dat'),
                    dtype=np.float32, mode='r', shape=(T, N, C))

    # ---- 与 ZScoreScaler 逐位一致：训练段全局标量 ----
    # 必须保持 float32 输入（scaler 就是 memmap float32 直接 np.mean/np.std），
    # 先转 float64 再统计会在末位差 1e-7，重训时与历史结果对不上。
    n_scale = int(T * train_ratio)
    train32 = np.array(raw[:n_scale, :, 0], dtype=np.float32)   # (n_scale,N) 副本
    mu_g = float(np.mean(train32))
    sd_g = float(np.std(train32))

    x = np.asarray(raw[:, :, 0], dtype=np.float64)         # (T,N) 只用 flow 通道
    train_len = T - int(T * 0.2) - int(T * 0.2)
    tr = x[:train_len]                                     # 绝不用 valid/test

    # ---- 1) 节点增益 ----
    a_raw = np.maximum(tr.mean(0), 1e-2)                   # (N,)

    # ---- 2) 全网共享周期曲线 ----
    nd = train_len // P
    tr2 = tr[:nd * P].reshape(nd, P, N)                    # (nday,P,N)
    dow_d = np.arange(nd) % 7
    Tn = np.zeros((7, P, N))
    for d in range(7):
        m = dow_d == d
        Tn[d] = tr2[m].mean(0) if m.any() else tr2.mean(0)
    S = (Tn / a_raw[None, None, :]).mean(-1)               # (7,P)
    ker = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    for d in range(7):
        S[d] = np.convolve(np.r_[S[d][-2:], S[d], S[d][:2]], ker, mode='same')[2:-2]
    S = S / S.mean()

    # ---- 3) 逐节点低秩修正 ----
    dev = Tn - a_raw[None, None, :] * S[:, :, None]        # (7,P,N)
    dev_m = dev.reshape(7 * P, N).T                        # (N,7P)
    r_eff = int(min(max(int(rank), 0), N))
    if r_eff > 0:
        U, sv, Vt = np.linalg.svd(dev_m - dev_m.mean(0, keepdims=True),
                                 full_matrices=False)
        z_init = U[:, :r_eff] * sv[:r_eff]                 # (N,r)
        D_init = Vt[:r_eff].reshape(r_eff, 7, P)           # (r,7,P)
        scale = np.abs(z_init).mean() + 1e-8               # 尺度归一到 O(1)
        z_init, D_init = z_init / scale, D_init * scale
        cum = float((sv[:r_eff] ** 2).sum() / (sv ** 2).sum())
    else:
        z_init, D_init, cum = np.zeros((N, 0)), np.zeros((0, 7, P)), 0.0

    out = dict(
        num_nodes=np.int64(N), period=np.int64(P), rank=np.int64(r_eff),
        mu_g=np.float64(mu_g), sd_g=np.float64(sd_g),
        # 转到标准化空间，与模型内部的计算域一致
        s_init=(S * (mu_g / sd_g)).astype(np.float32),
        alpha_init=(a_raw / mu_g).astype(np.float32),
        d_init=(D_init / sd_g).astype(np.float32),
        z_init=z_init.astype(np.float32),
        # 诊断量（不进模型）
        a_raw=a_raw.astype(np.float32),
        dev_energy_frac=np.float64(cum),
        train_len=np.int64(train_len),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--rank', type=int, default=2)
    ap.add_argument('--train-ratio', type=float, default=0.6)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    out = build(a.dataset, rank=a.rank, train_ratio=a.train_ratio)
    path = a.out or os.path.join('prob_baselines', 'SRPM',
                                 f'srpm_init_{a.dataset}.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print(f'[SRPM] 写出 {path}')
    print(f'       N={int(out["num_nodes"])} P={int(out["period"])} '
          f'rank={int(out["rank"])} train_len={int(out["train_len"])}')
    print(f'       mu_g={float(out["mu_g"]):.4f} sd_g={float(out["sd_g"]):.4f}  '
          f'低秩解释偏差能量 {float(out["dev_energy_frac"]) * 100:.2f}%  '
          f'(r={int(out["rank"])})')
    print(f'       a_raw 分位: '
          f'{np.round(np.percentile(out["a_raw"], [0, 50, 100]), 2)}')


if __name__ == '__main__':
    main()
