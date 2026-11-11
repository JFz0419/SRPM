"""SRPM / PEMS03（12→12）。

差异只有 num_nodes=358 与 period=288（5 分钟采样），二者由 ``desc.json`` 自动读取。
主表用 hid=192 / depth=3 / rank=2。

训练（仓库根目录）：
    CODEBUDDY_SAFE_DELETE_ENABLED=0 python experiments/train.py \
        -c prob_baselines/SRPM/SRPM_PEMS03.py --gpus 0
"""
import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from prob_baselines.SRPM.SRPM_base import build_cfg

DATA_NAME = 'PEMS03'
CFG = build_cfg(DATA_NAME)
