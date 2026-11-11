"""概率预测的闭式评估指标（边际口径）。

约定（与 BasicTS masked 系指标一致）：
- 张量最后一维是通道维（C=1 的 flow 通道）；prediction / target 形状 [B, L, N, C]；
- null_val 掩码：target == null_val 的位置不计入（GBA 的 NULL_VAL=0.0）；
- 指标在真实尺度上计算（runner 的 postprocessing 已反归一化）；
- 掩码加权沿用 BasicTS 惯例 mask / mean(mask)，即等价于"有效位置上的均值"。

闭式来源：
- 单高斯 CRPS（闭式）：
    CRPS = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y - mu) / sigma
- S 分量等权高斯混合 CRPS（erf 精确、无采样噪声）：
    CRPS = sum_i w_i * A(y - mu_i, sigma_i^2) - 0.5 * sum_i sum_j w_i w_j * A(mu_i - mu_j, sigma_i^2 + sigma_j^2)
    其中 A(mu, sigma^2) = mu * (2*Phi(mu/sigma) - 1) + 2*sigma*phi(mu/sigma)，w_i = 1/S。
    对应我们的两级结构：节点边际 = N(z_s + mu_n, sigma_n) 的 S 分量等权混合
    （各分量共享对角 sigma_n，分量均值来自 S 条 region 轨迹 + 同一节点修正）。

用法（配置文件内）：
    from basicts.metrics import crps_gaussian, crps_gmm, mis_gaussian, picp_gaussian
    CFG.METRICS.FUNCS = EasyDict({
        'MAE': masked_mae, 'RMSE': masked_rmse,
        'CRPS': crps_gaussian,            # 单高斯头（点预测改造版）
        # 'CRPS': crps_gmm,               # S 分量混合头（RegionDiT 两级不确定性）
    })
注意：metric_forward 会按函数签名自动从 runner forward 返回的 dict 里取
'prediction' / 'prediction_sigma' / 'prediction_means' / 'prediction_sigmas' / 'target' 等字段，
并自动注入 null_val——所以这些指标函数的参数名就是与 runner 的契约。
"""

import math

import numpy as np
import torch

_SQRT_2 = float(np.sqrt(2.0))
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_SQRT_PI = float(np.sqrt(np.pi))


def _get_mask(target: torch.Tensor, null_val) -> torch.Tensor:
    """BasicTS 惯例的 null 掩码：mask / mean(mask)，等价于有效位置上的均值加权。"""
    if null_val is not None and isinstance(null_val, float) and np.isnan(null_val):
        mask = ~torch.isnan(target)
    elif null_val is not None:
        mask = (target != null_val)
    else:
        mask = torch.ones_like(target, dtype=torch.bool)
    mask = (mask & torch.isfinite(target)).to(target.dtype)
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.nan_to_num(values * mask))


def _phi(z: torch.Tensor) -> torch.Tensor:
    """标准正态密度 phi(z)。"""
    return torch.exp(-0.5 * z * z) / _SQRT_2PI


def _Phi(z: torch.Tensor) -> torch.Tensor:
    """标准正态分布函数 Phi(z)（erf 精确）。"""
    return 0.5 * (1.0 + torch.erf(z / _SQRT_2))


def _A(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """CRPS 混合公式的核 A(mu, sigma^2) = E|X - 0|, X ~ N(mu, sigma^2)。"""
    z = mu / sigma
    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z))


def crps_gaussian(prediction: torch.Tensor, prediction_sigma: torch.Tensor,
                  target: torch.Tensor, null_val=np.nan) -> torch.Tensor:
    """单高斯边际 CRPS（闭式）。

    Args:
        prediction: 高斯均值 mu，[B, L, N, C]（真实尺度）。
        prediction_sigma: 高斯标准差 sigma，形状同 prediction。
        target: 真值，形状同 prediction。
        null_val: 数据缺失值（自动注入）。

    Returns:
        torch.Tensor: 标量 CRPS。
    """

    sigma = torch.clamp(prediction_sigma, min=1e-6)
    z = (target - prediction) / sigma
    crps = sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - 1.0 / _SQRT_PI)
    mask = _get_mask(target, null_val)
    return _masked_mean(crps, mask)


def crps_gmm(prediction_means: torch.Tensor, prediction_sigmas: torch.Tensor,
             target: torch.Tensor, null_val=np.nan, prediction_weights=None) -> torch.Tensor:
    """Gaussian-mixture marginal CRPS; optional learned weights, uniform by default.

    Args:
        prediction_means: 分量均值，[S, B, L, N, C]（真实尺度）。
        prediction_sigmas: 分量标准差，[S, B, L, N, C] 或 [B, L, N, C]
            （后者 = 各分量共享对角 sigma，即我们的节点残差头形态）。
        target: 真值，[B, L, N, C]。
        null_val: 数据缺失值（自动注入）。
        prediction_weights: Optional probabilities [S,B,L,N,C], summing to 1
            along the component axis. Omit for RegionDiT's equal-weight mixture.

    Returns:
        torch.Tensor: 标量 CRPS。
    """

    if prediction_sigmas.dim() == prediction_means.dim() - 1:
        # 共享 sigma：[B,L,N,C] -> [S,B,L,N,C]
        prediction_sigmas = prediction_sigmas.unsqueeze(0).expand_as(prediction_means)
    S = prediction_means.shape[0]
    sigma = torch.clamp(prediction_sigmas, min=1e-6)

    # 第一项：sum_i w_i * A(y - mu_i, sigma_i^2)
    a_i = _A(target.unsqueeze(0) - prediction_means, sigma)  # [S,B,L,N,C]
    weights = (torch.full_like(prediction_means, 1.0 / S) if prediction_weights is None
               else prediction_weights.expand_as(prediction_means))
    term1 = (weights * a_i).sum(dim=0)

    # 第二项：0.5 * sum_i sum_j w_i w_j * A(mu_i - mu_j, sigma_i^2 + sigma_j^2)
    # 逐分量循环累加以控制显存（S=50 时一次性建 [S,S,B,L,N,C] 会爆内存）
    cross_sum = None
    for i in range(S):
        d = prediction_means - prediction_means[i:i + 1]          # [S,B,L,N,C]
        s_ij = torch.sqrt(sigma * sigma + sigma[i:i + 1] ** 2)    # [S,B,L,N,C]
        a_ij = _A(d, s_ij)
        pair = weights[i] * (weights * a_ij).sum(dim=0)
        cross_sum = pair if cross_sum is None else cross_sum + pair
    term2 = 0.5 * cross_sum

    crps = term1 - term2
    mask = _get_mask(target, null_val)
    return _masked_mean(crps, mask)


def _z_score_from_alpha(alpha: float) -> float:
    """双侧区间水平 alpha 对应的 z 分位数（如 alpha=0.1 -> 90% 区间 -> z≈1.645）。"""
    return _SQRT_2 * float(torch.erfinv(torch.tensor(1.0 - alpha)))


def mis_gaussian(prediction: torch.Tensor, prediction_sigma: torch.Tensor,
                 target: torch.Tensor, null_val=np.nan, alpha: float = 0.1) -> torch.Tensor:
    """单高斯平均区间得分 MIS（越小越好；alpha=0.1 即 90% 区间）。

    MIS = (u - l) + (2/alpha) * (l - y) * 1[y < l] + (2/alpha) * (y - u) * 1[y > u]
    """

    z = _z_score_from_alpha(alpha)
    sigma = torch.clamp(prediction_sigma, min=1e-6)
    lower = prediction - z * sigma
    upper = prediction + z * sigma
    score = (upper - lower) \
        + (2.0 / alpha) * torch.relu(lower - target) \
        + (2.0 / alpha) * torch.relu(target - upper)
    mask = _get_mask(target, null_val)
    return _masked_mean(score, mask)


def picp_gaussian(prediction: torch.Tensor, prediction_sigma: torch.Tensor,
                  target: torch.Tensor, null_val=np.nan, alpha: float = 0.1) -> torch.Tensor:
    """单高斯预测区间覆盖率 PICP（alpha=0.1 即 90% 区间的经验覆盖率，理想≈0.9）。"""

    z = _z_score_from_alpha(alpha)
    sigma = torch.clamp(prediction_sigma, min=1e-6)
    covered = ((target >= prediction - z * sigma) & (target <= prediction + z * sigma)).to(target.dtype)
    mask = _get_mask(target, null_val)
    return _masked_mean(covered, mask)


def crps_quantile(quantile_preds: torch.Tensor, quantile_levels: torch.Tensor,
                  target: torch.Tensor, null_val=np.nan) -> torch.Tensor:
    """分位数预测的边际 CRPS（pinball 恒等式，离散近似）。

    精确恒等式：CRPS(F, y) = 2 * ∫₀¹ ρ_τ(y - F⁻¹(τ)) dτ，其中 ρ_τ 为 pinball
    损失。对 Q 个分位水平取均值近似该积分（水平近均匀时无偏）——与
    crps_gaussian / crps_gmm 同一真实尺度 + mean 池化口径，可直接同表对读。

    Args:
        quantile_preds: 分位数预测，[Q, B, L, N, C]（Q 轴与采样契约的 S 轴
            同位；真实尺度，已仿射反归一化）。
        quantile_levels: 分位水平，[Q]（模型 forward 透传）。
        target: 真值，[B, L, N, C]。
        null_val: 数据缺失值（自动注入）。

    Returns:
        torch.Tensor: 标量 CRPS。
    """

    diff = target.unsqueeze(0) - quantile_preds               # [Q,B,L,N,C]
    tau = quantile_levels.to(device=diff.device, dtype=diff.dtype)
    tau = tau.view([-1] + [1] * target.dim())                # [Q,1,1,1,1]
    pinball = tau * torch.relu(diff) + (1.0 - tau) * torch.relu(-diff)
    crps = 2.0 * pinball.mean(dim=0)                          # [B,L,N,C]
    mask = _get_mask(target, null_val)
    return _masked_mean(crps, mask)


@torch.no_grad()
def quantile_intervals(quantile_preds: torch.Tensor, quantile_levels: torch.Tensor,
                       alphas=(0.1,)):
    """从分位数预测提取中心区间（与 gmm_intervals 同形状协议）。

    对每个 alpha 取与 alpha/2、1 - alpha/2 最近的分位档（要求 quantile_preds
    已沿 Q 维非交叉——PointQuantileWrapper 评估期已排序）。
    返回 lower / upper，形状 [A, B, L, N, C]，供 picp_interval / mis_interval
    （interval_index 索引 alpha 档）消费。

    注意：mis_interval 的 alpha 惩罚参数需与这里选的区间档一致（默认均 0.1）。
    """

    levels = quantile_levels.reshape(-1)                       # [Q]
    lowers, uppers = [], []
    for a in alphas:
        lo_idx = int((levels - a / 2).abs().argmin())
        hi_idx = int((levels - (1.0 - a / 2)).abs().argmin())
        lowers.append(quantile_preds[lo_idx])                  # [B,L,N,C]
        uppers.append(quantile_preds[hi_idx])
    return torch.stack(lowers, dim=0), torch.stack(uppers, dim=0)


@torch.no_grad()
def gmm_intervals(means, sigmas, alphas=(0.2, 0.1), chunk_size=8192, weights=None):
    """Central intervals from the equal-weight mixture CDF, not Gaussian moments.

    Returns lower/upper [A,B,L,N,C]. Chunk positions to bound memory. Each
    quantile is bracketed by the min/max component quantile at that probability.
    """
    if not alphas or any(not 0 < a < 1 for a in alphas):
        raise ValueError('Interval alpha must be between zero and one')
    if sigmas.dim() == means.dim() - 1:
        sigmas = sigmas.unsqueeze(0)
    sigmas = sigmas.expand_as(means).clamp_min(1e-6)
    shape = means.shape[1:]
    mu, sd = means.flatten(1), sigmas.flatten(1)
    weight = None if weights is None else weights.expand_as(means).flatten(1)
    probs = means.new_tensor([p for a in alphas for p in (a / 2, 1 - a / 2)])
    z = _SQRT_2 * torch.erfinv(2 * probs - 1)
    result = means.new_empty(len(probs), mu.shape[1])
    for start in range(0, mu.shape[1], chunk_size):
        m = mu[:, start:start + chunk_size].unsqueeze(0)
        s = sd[:, start:start + chunk_size].unsqueeze(0)
        component_q = m + s * z[:, None, None]
        lo, hi = component_q.amin(1), component_q.amax(1)
        for _ in range(28):
            mid = (lo + hi) / 2
            component_cdf = _Phi((mid[:, None] - m) / s)
            cdf = (component_cdf.mean(1) if weight is None else
                   (component_cdf * weight[:, start:start + chunk_size][None]).sum(1))
            below = cdf < probs[:, None]
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        result[:, start:start + chunk_size] = (lo + hi) / 2
    result = result.reshape(len(alphas), 2, *shape)
    return result[:, 0], result[:, 1]


def picp_interval(prediction_lower, prediction_upper, target, null_val=np.nan,
                  interval_index=0):
    lower, upper = prediction_lower[interval_index], prediction_upper[interval_index]
    covered = ((target >= lower) & (target <= upper)).to(target.dtype)
    return _masked_mean(covered, _get_mask(target, null_val))


def width_interval(prediction_lower, prediction_upper, target, null_val=np.nan,
                   interval_index=0):
    width = prediction_upper[interval_index] - prediction_lower[interval_index]
    return _masked_mean(width, _get_mask(target, null_val))


def mis_interval(prediction_lower, prediction_upper, target, null_val=np.nan,
                 interval_index=0, alpha=0.1):
    lower, upper = prediction_lower[interval_index], prediction_upper[interval_index]
    score = upper - lower + 2 / alpha * (
        torch.relu(lower - target) + torch.relu(target - upper))
    return _masked_mean(score, _get_mask(target, null_val))




# ======================================================================
# 联合（多变量）指标 —— 采样口径
# ======================================================================
# 边际指标（CRPS/MIS/PICP/MPIW）只看单个 (node, horizon)，对依赖结构完全不敏感：
# 把跨节点、跨 horizon 的相关改成任意值，这四个数一分不动。要评价"联合"，
# 必须用多变量 proper scoring rule。这里实现两个：
#
#   ES（Energy Score, Gneiting et al. 2008）
#       ES(P, y) = E_P‖X − y‖₂ − ½ E_P‖X − X′‖₂ ,  X, X′ iid ~ P
#       严格 proper，越小越好。采样估计量（无偏）：
#       ÊS = (1/S)Σ_s‖x_s − y‖ − 1/(2S(S−1)) Σ_{s≠s'}‖x_s − x_s'‖
#
#   VS（Variogram Score, Scheuerer & Hamill 2015）
#       VS(P, y) = Σ_{i<j} ( |y_i − y_j|^p − E_P|X_i − X_j|^p )²
#       只依赖两两差，对维度不敏感；ES 在高维判别力会退化时的替代。
#
# axis 决定"沿哪一维做联合"，两者含义完全不同，报告时必须写清楚：
#   'horizon' → 每个节点一条 P 步轨迹（D=P），site=(window, node)
#   'node'    → 每个 horizon 一张全网切片（D=N），site=(window, horizon)
#
# 缺失处理：一个 site 只在**联合维全部有效**时计入（d 维欧氏范数缺一项就没定义）。
#   d=12 时约丢 4%；d=149 时约丢 40%——所以 'node' 口径在 PEMS 上样本损失很大，
#   只能当参考列。返回值里的 mask 可以用来核对实际计入比例。


def _joint_view(paths, target, null_val, axis):
    """把 (S,B,P,N) 摊成 (S, M, D) / (M, D) / (M,)，并给出 site 有效性。"""
    if paths.dim() == 5:                                   # 兼容 (S,B,P,N,C=1)
        paths = paths[..., 0]
    if target.dim() == 4:
        target = target[..., 0]
    if axis == 'horizon':
        x = paths.permute(0, 1, 3, 2)                      # (S,B,N,P)
        y = target.permute(0, 2, 1)                        # (B,N,P)
    elif axis == 'node':
        x = paths                                          # (S,B,P,N)
        y = target
    else:
        raise ValueError(f"axis 只能是 'horizon' 或 'node'，收到 {axis!r}")
    S, B = x.shape[0], x.shape[1]
    D = x.shape[-1]
    x = x.reshape(S, B * x.shape[2], D)
    y = y.reshape(B * y.shape[1], D)
    ok = ~torch.isnan(y)
    if null_val is not None and not (isinstance(null_val, float) and np.isnan(null_val)):
        ok = ok & (y != null_val)
    site_ok = ok.all(-1)                                   # (M,)
    return x, y, site_ok


def _energy_score_site(x, y, chunk=512):
    """(S,M,D),(M,D) -> (M,) 每个 site 的 ES。"""
    S, M, D = x.shape
    out = x.new_zeros(M)
    for i in range(0, M, chunk):
        xs = x[:, i:i + chunk]                             # (S,m,D)
        ys = y[i:i + chunk]                                # (m,D)
        term1 = (xs - ys.unsqueeze(0)).norm(dim=-1).mean(0)          # (m,)
        sq = (xs ** 2).sum(-1)                                       # (S,m)
        g = torch.einsum('smd,tmd->stm', xs, xs)                     # (S,S,m)
        d2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * g
        pair = d2.clamp_min(0).sqrt().sum(dim=(0, 1)) / (S * (S - 1))
        out[i:i + chunk] = term1 - 0.5 * pair
    return out


def energy_score(paths, target, null_val=np.nan, axis='horizon', chunk=512):
    """Energy Score（真实尺度、越小越好）。返回 (score, mask)，形状 (B,N) 或 (B,P)。"""
    x, y, site_ok = _joint_view(paths, target, null_val, axis)
    score = _energy_score_site(x, y, chunk=chunk)
    B = target.shape[0]
    K = score.numel() // B
    return score.reshape(B, K), site_ok.reshape(B, K)


def _variogram_score_site(x, y, p=1.0, chunk=512, pairs=None):
    """(S,M,D),(M,D) -> (M,) 每个 site 的 VS（未加权求和，i<j）。"""
    S, M, D = x.shape
    if pairs is None:
        pairs = torch.triu_indices(D, D, offset=1)
    i, j = pairs[0], pairs[1]
    out = x.new_zeros(M)
    for s0 in range(0, M, chunk):
        xs = x[:, s0:s0 + chunk]                           # (S,m,D)
        ys = y[s0:s0 + chunk]
        obs = (ys[:, i] - ys[:, j]).abs() ** p             # (m,E)
        sim = (xs[:, :, i] - xs[:, :, j]).abs() ** p       # (S,m,E)
        out[s0:s0 + chunk] = ((obs - sim.mean(0)) ** 2).mean(-1)
    return out


def variogram_score(paths, target, null_val=np.nan, axis='horizon', p=1.0,
                    chunk=512):
    """Variogram Score（真实尺度、越小越好）。返回 (score, mask)。"""
    x, y, site_ok = _joint_view(paths, target, null_val, axis)
    score = _variogram_score_site(x, y, p=p, chunk=chunk)
    B = target.shape[0]
    K = score.numel() // B
    return score.reshape(B, K), site_ok.reshape(B, K)


# ----------------------------------------------------------------------
# runner 契约包装：metric_forward 要求「函数返回标量张量」且形参名与 dict 键一致，
# 而上面两个函数的首个形参叫 paths、且返回 (score, mask) 二元组。
# 这两个包装把 samples → 标量，供 CFG.METRICS.FUNCS 直接注册；
# 训练/验证期若 forward 没返回 'samples'，runner 的 _metric_args_available
# 会判定缺参并自动跳过，不会报错、也不会悄悄算错口径。
# ----------------------------------------------------------------------

def _joint_scalar(samples, target, null_val, axis, kind, p=1.0):
    if samples.dim() == 4:                                # (B,P,N,C) 退化：无法联合
        return None
    if samples.dim() == 5 and samples.shape[0] <= 1:
        return None
    fn = energy_score if kind == 'es' else variogram_score
    kw = {} if kind == 'es' else {'p': p}
    score, mask = fn(samples, target, null_val=null_val, axis=axis, **kw)
    keep = mask.to(score.dtype)
    return (score * keep).sum() / keep.sum().clamp_min(1.0)


def energy_score_metric(samples, target, null_val=np.nan, axis='horizon'):
    """Energy Score（越小越好）。samples 形如 [S,B,P,N,C]，由 runner 反归一化后传入。"""
    return _joint_scalar(samples, target, null_val, axis, 'es')


def variogram_score_metric(samples, target, null_val=np.nan, axis='horizon', p=1.0):
    """Variogram Score（越小越好）。只依赖两两差，对高维/vs ES 更敏感。"""
    return _joint_scalar(samples, target, null_val, axis, 'vs', p=p)
