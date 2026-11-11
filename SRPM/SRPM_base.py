"""SRPM（Shared-period Residual Prediction Model）的公共配置构造器。

四个数据集的差异只有三件事，其余完全同构：

1. ``num_nodes`` / ``period`` —— 由 ``datasets/<name>/desc.json`` 决定
   （``period = 1440 // frequency``，5 min → 288，60 min → 24）；
2. ``init_path`` 指向各自的 ``srpm_init_<name>.npz``；
3. 结构超参 ``hid / depth``（主表：PEMS04 用 256，其余 192；depth 一律 3）。

结构是**定稿的**，不留消融开关
------------------------------
模型侧只有一条路径：周期基（7×P 共享表 + 逐节点增益/偏置 + rank-2 低秩修正）
→ 窗口残差水平外推（κ_h）→ 全网共享残差 MLP → 单高斯头。
σ 侧是「全局标量尺度 × 逐节点窗口尺度」，不含 ``φ₁μ+φ₂μ²`` 方差律、
不含逐 horizon / 逐节点尺度乘子——这些是被消融否定的分支，不再保留代码路径。

想复现消融结论请用 ``prob_baselines/SCM``（保留全部开关），本目录只交付定稿。

参数量
------
::

    MLP   = (32h + h) + depth·(2h² + 2h) + (h·H + H) + (h + 1)
    其它  = 7P(template) + r·7P(D) + N·r(z) + 2N(log_alpha, beta) + H(κ) + 1(log c)

PEMS08 (N=170, P=288, r=2, h=192, depth=3) ⇒ 237,922，与训练日志一致。

σ 的单位
--------
net 交付的是**标准化空间**的 σ，runner 的 postprocessing 对 σ 只乘 ``std``
（平移无关），因此原始单位下 ``σ_real = std·exp(log_sig_c + s)``。
注意参数名 ``sigma_raw_unit`` 与实测相反，看 ``_rescale_sigma`` 而不是名字。
"""
import json
import os

import numpy as np
from easydict import EasyDict
from functools import partial

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import (crps_gaussian, masked_mae, masked_mape, masked_rmse,
                             mis_gaussian, picp_gaussian)
from basicts.runners import ProbTSFRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import GLOBAL_SEED, get_regular_settings

from prob_baselines.SRPM.arch import SRPM
from prob_baselines.SRPM.srpm_loss import srpm_crps

############################## Hot Parameters ##############################
OUTPUT_LEN = 12
INPUT_LEN = 12

HID = 192            # 主表默认宽度；PEMS04 在配置里覆盖为 256
DEPTH = 3            # 残差块个数
RANK = 2             # 低秩时序修正的秩
DROPOUT = 0.15
SMOOTH_W = 0.1       # 模板平滑正则权重

# 运行参数（不影响结构）
EPOCHS = int(os.environ.get('SRPM_EPOCHS', 400))
BATCH_SIZE = int(os.environ.get('SRPM_BATCH_SIZE', 64))
TAG = os.environ.get('SRPM_TAG') or None


def _lr_milestones(num_epochs: int):
    """milestones = {0.01, 0.3, 0.6, 0.8} × epochs。"""
    return sorted({max(1, int(num_epochs * f)) for f in (0.01, 0.3, 0.6, 0.8)})


def _init_path(data_name: str) -> str:
    path = os.path.join('prob_baselines', 'SRPM', f'srpm_init_{data_name}.npz')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"缺少初始化文件 {path}，先跑：\n"
            f"    python prob_baselines/SRPM/build_init.py --dataset {data_name} --rank {RANK}")
    return path


def _desc(data_name: str) -> dict:
    return json.load(open(os.path.join('datasets', data_name, 'desc.json'),
                          encoding='utf-8'))


def build_cfg(data_name: str, tag: str = None, description: str = None,
              num_epochs: int = None, hid: int = None, depth: int = None,
              rank: int = None, **overrides) -> EasyDict:
    """构造某个数据集的完整 CFG。"""
    regular_settings = get_regular_settings(data_name)
    TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
    NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
    RESCALE = regular_settings['RESCALE']
    NULL_VAL = regular_settings['NULL_VAL']
    if not RESCALE:
        raise ValueError(
            f'{data_name} 的 RESCALE=False，σ 的原始单位换算不成立；'
            '本实现只支持 RESCALE=True 的数据集')

    hid = int(HID if hid is None else hid)
    depth = int(DEPTH if depth is None else depth)
    rank = int(RANK if rank is None else rank)
    num_epochs = int(EPOCHS if num_epochs is None else num_epochs)
    tag = tag or TAG
    desc = _desc(data_name)
    num_nodes = int(desc['shape'][1])
    period = 1440 // int(desc['frequency (minutes)'])
    init_path = _init_path(data_name)

    z = np.load(init_path, allow_pickle=True)
    if int(z['num_nodes']) != num_nodes or int(z['period']) != period:
        raise ValueError(
            f"{init_path} 的 (N={int(z['num_nodes'])}, P={int(z['period'])}) 与 "
            f"{data_name} 的 (N={num_nodes}, P={period}) 不符，请重建初始化文件")
    rank = int(min(rank, z['z_init'].shape[1]))

    # 参数量核对（形态与 srpm_net 的说明一致）
    fdim = INPUT_LEN + 5 + 2 + OUTPUT_LEN + 1
    n_mlp = ((fdim * hid + hid) + depth * (2 * hid * hid + 2 * hid)
             + (hid * OUTPUT_LEN + OUTPUT_LEN) + (hid + 1))
    n_par = (n_mlp + 7 * period
             + (rank * 7 * period + num_nodes * rank if rank > 0 else 0)
             + 2 * num_nodes + OUTPUT_LEN + 1)
    print(f"[SRPM/{data_name}] N={num_nodes} P={period} rank={rank} "
          f"hid={hid} depth={depth} "
          f"mu_g={float(z['mu_g']):.4f} sd_g={float(z['sd_g']):.4f} "
          f"低秩解释偏差 {float(z['dev_energy_frac']) * 100:.2f}% "
          f"≈参数 {n_par}", flush=True)

    MODEL_PARAM = {
        'num_nodes': num_nodes,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
        'period': period,
        'init_path': init_path,
        'hid': hid,
        'depth': depth,
        'rank': rank,
        'mlp_dropout': DROPOUT,
    }
    MODEL_PARAM.update(overrides)

    CFG = EasyDict()
    CFG.DESCRIPTION = description or f'SRPM / {data_name} 12->12'
    CFG.GPU_NUM = 1
    CFG.RUNNER = ProbTSFRunner

    CFG.DATASET = EasyDict()
    CFG.DATASET.NAME = data_name
    CFG.DATASET.TYPE = TimeSeriesForecastingDataset
    CFG.DATASET.PARAM = EasyDict({
        'dataset_name': data_name,
        'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
    })

    CFG.SCALER = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': data_name,
        'train_ratio': TRAIN_VAL_TEST_RATIO[0],
        'norm_each_channel': NORM_EACH_CHANNEL,
        'rescale': RESCALE,
    })

    CFG.MODEL = EasyDict()
    CFG.MODEL.NAME = 'SRPM'
    CFG.MODEL.ARCH = SRPM
    CFG.MODEL.PARAM = MODEL_PARAM
    # 通道 0 = flow，1 = time-of-day（日内占比），2 = day-of-week（周内占比）
    CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
    CFG.MODEL.TARGET_FEATURES = [0]

    CFG.METRICS = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({
        'MAE': masked_mae,
        'MAPE': masked_mape,
        'RMSE': masked_rmse,
        'CRPS': crps_gaussian,                       # 概率主指标（闭式）
        'MIS': mis_gaussian,                         # 90% 平均区间得分
        'PICP': picp_gaussian,                       # 90% 覆盖率
        # ---- 95% 口径：与 RIPCN / 概率基线主表一致，论文汇报以这一档为准 ----
        'MIS95': partial(mis_gaussian, alpha=0.05),
        'PICP95': partial(picp_gaussian, alpha=0.05),
    })
    CFG.METRICS.TARGET = 'CRPS'
    CFG.METRICS.BEST = 'min'
    CFG.METRICS.NULL_VAL = NULL_VAL

    CFG.TRAIN = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = num_epochs
    # 目录名带种子：多 seed 重复时不能写同一目录（easytorch 会当成续训/覆盖）
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'prob_checkpoints', 'SRPM',
        '_'.join([data_name, f'{tag or "base"}_s{GLOBAL_SEED}',
                  str(num_epochs), str(INPUT_LEN), str(OUTPUT_LEN)]))
    CFG.TRAIN.LOSS = partial(srpm_crps, reg_w=SMOOTH_W)
    CFG.TRAIN.OPTIM = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = 'Adam'
    CFG.TRAIN.OPTIM.PARAM = {'lr': 0.002, 'weight_decay': 0.0001}
    CFG.TRAIN.LR_SCHEDULER = EasyDict()
    CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
    CFG.TRAIN.LR_SCHEDULER.PARAM = {'milestones': _lr_milestones(num_epochs),
                                    'gamma': 0.5}
    CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
    CFG.TRAIN.DATA = EasyDict()
    CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
    CFG.TRAIN.DATA.SHUFFLE = True

    CFG.VAL = EasyDict()
    CFG.VAL.INTERVAL = 1
    CFG.VAL.DATA = EasyDict()
    CFG.VAL.DATA.BATCH_SIZE = 64

    CFG.TEST = EasyDict()
    # 训练期不跑 test（on_training_end 自动加载最优 ckpt 跑最终 test）
    CFG.TEST.INTERVAL = 1000
    CFG.TEST.DATA = EasyDict()
    CFG.TEST.DATA.BATCH_SIZE = 64

    CFG.EVAL = EasyDict()
    CFG.EVAL.HORIZONS = list(range(1, OUTPUT_LEN + 1))
    CFG.EVAL.USE_GPU = True
    CFG.OUTPUT_DIR = None
    return CFG
