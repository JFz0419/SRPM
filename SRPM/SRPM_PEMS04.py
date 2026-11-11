"""SRPM / PEMS04（12→12）。

与其它数据集唯一的差别：主表用 hid=256（其余集为 192）。
该宽度来自当时的一组宽度扫描，改回 192 会让 MAE 变差约 0.3。

训练（仓库根目录）：
    CODEBUDDY_SAFE_DELETE_ENABLED=0 python experiments/train.py \
        -c prob_baselines/SRPM/SRPM_PEMS04.py --gpus 0
"""
import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from prob_baselines.SRPM.SRPM_base import build_cfg

DATA_NAME = 'PEMS04'
CFG = build_cfg(DATA_NAME, hid=256)
