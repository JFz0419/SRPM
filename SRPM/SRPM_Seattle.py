"""SRPM / Seattle（12→12）。

与其它数据集同构，差异只有 num_nodes=323 与 period=24（60 分钟采样），
二者都由 ``desc.json`` 自动读取。主表用 hid=192 / depth=3 / rank=2。

训练（仓库根目录）：
    CODEBUDDY_SAFE_DELETE_ENABLED=0 python experiments/train.py \
        -c prob_baselines/SRPM/SRPM_Seattle.py --gpus 0
"""
import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from prob_baselines.SRPM.SRPM_base import build_cfg

DATA_NAME = 'Seattle'
CFG = build_cfg(DATA_NAME)
