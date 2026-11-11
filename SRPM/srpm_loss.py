"""SRPM 的训练损失：单高斯 CRPS + 模板平滑正则。

为什么是 CRPS 而不是 NLL
-------------------------
两者对 σ 的梯度形状完全不同：
``∂/∂σ`` 在 CRPS 下于 ``|z|`` 大时饱和于 −0.564，而 NLL 下是 ``(1−z²)/σ``，
``|z|=10`` 时对 σ 的推力是 CRPS 的约 175 倍。NLL 会把「放大 σ」当成比
「改进 μ」更便宜的下降方向，实测 MAE 反而变差 6.7%–10.0%。正式训练一律用 CRPS。

签名即契约
----------
runner 按参数名从模型 forward 的返回 dict 里取参，``null_val`` 由 runner 注入。
模型返回的正则项名为 ``srpm_reg``（``SRPMNet.reg()``），带默认值，
故缺席时损失照常计算。

正则在模型里算的原因：它作用在 ``template`` 参数上，而 loss 只能看到前向结果。
"""
import numpy as np
import torch

from basicts.metrics import crps_gaussian


def srpm_crps(prediction: torch.Tensor, prediction_sigma: torch.Tensor,
              target: torch.Tensor, null_val=np.nan, srpm_reg=None,
              reg_w: float = 0.1) -> torch.Tensor:
    """单高斯 CRPS（masked）+ ``reg_w × 模板平滑正则``。

    Args:
        prediction: 高斯均值，``[B,H,N,1]``（原始单位）。
        prediction_sigma: 高斯标准差，形状同 prediction（原始单位）。
        target: 真值，形状同 prediction。
        null_val: 缺失标记（runner 注入）。
        srpm_reg: 模型返回的模板平滑正则标量（可选）。
        reg_w: 正则权重，主表口径 0.1。
    """
    loss = crps_gaussian(prediction, prediction_sigma, target, null_val)
    if srpm_reg is not None and reg_w:
        loss = loss + float(reg_w) * srpm_reg
    return loss
