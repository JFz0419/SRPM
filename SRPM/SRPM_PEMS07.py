"""SRPM / PEMS07（12→12）。

差异只有 num_nodes=883 与 period=288。主表未使用该数据集，配置保留以便扩展。

训练（仓库根目录）：
    CODEBUDDY_SAFE_DELETE_ENABLED=0 python experiments/train.py \
        -c prob_baselines/SRPM/SRPM_PEMS07.py --gpus 0
"""
import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from prob_baselines.SRPM.SRPM_base import build_cfg

DATA_NAME = 'PEMS07'
CFG = build_cfg(DATA_NAME)
