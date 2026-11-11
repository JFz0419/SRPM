"""SRPM：共享周期基 + 节点标定 + 残差 MLP 的概率预测模型（定稿版）。

目录：
    arch/srpm_net.py   网络本体（周期基 + 节点标定 + 共享残差 MLP + 单高斯头）
    arch/srpm_arch.py  BasicTS 适配层（批次 ↔ 网络签名 ↔ 概率契约）
    srpm_loss.py       训练损失（单高斯 CRPS + 模板平滑正则）
    build_init.py      训练段统计 → srpm_init_<dataset>.npz（周期模板/节点增益/低秩基）
    SRPM_<dataset>.py  训练配置

前置：先建初始化文件（仓库根目录）
    python prob_baselines/SRPM/build_init.py --dataset PEMS08 --rank 2

训练：
    CODEBUDDY_SAFE_DELETE_ENABLED=0 python experiments/train.py \
        -c prob_baselines/SRPM/SRPM_PEMS08.py --gpus 0

一句话概括：**不用图、不用节点嵌入、不用深度编码器**——一张全网共享的
7×P 周期表 + 每节点 2 个标定标量 + 一个全网共享残差 MLP。

与 ``prob_baselines/SCM`` 的关系：SRPM 是 SCM 在消融实验确定最终结构后的**定稿
固化**，不含任何消融开关，参数名与 SCM 最终配置完全一致，可直接加载历史
checkpoint。要复现消融链请回到 SCM。
"""
from .arch import SRPM, SRPMNet
from .srpm_loss import srpm_crps

__all__ = ['SRPM', 'SRPMNet', 'srpm_crps']
