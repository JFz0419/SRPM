"""概率时间序列预测 runner 子包。

- ProbTSFRunner: 通用概率训练/评估 runner——支持概率头改造的点预测模型（NLL/分位数训练）
  以及一切遵守"概率评估契约"的自研模型；
- FMProbTSFRunner: Flow Matching 语义子类（RegionDiT 管线专用，S1/S2 两阶段）。

契约文档见 prob_tsf_runner.py 模块 docstring。
"""

from .fm_runner import FMProbTSFRunner
from .prob_tsf_runner import ProbTSFRunner

__all__ = ['ProbTSFRunner', 'FMProbTSFRunner']
