"""SRPM / PEMS08（12→12）。

差异只有 num_nodes=170 与 period=288（5 分钟采样）。
主表用 hid=192 / depth=3 / rank=2，参数量 237,922。

训练（仓库根目录）：
    CODEBUDDY_SAFE_DELETE_ENABLED=0 python experiments/train.py \
        -c prob_baselines/SRPM/SRPM_PEMS08.py --gpus 0
"""
import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from prob_baselines.SRPM.SRPM_base import build_cfg

DATA_NAME = 'PEMS08'
CFG = build_cfg(DATA_NAME)
