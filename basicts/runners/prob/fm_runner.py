"""Flow Matching 语义 runner（FMProbTSFRunner）—— RegionDiT 管线专用。

与基类 ProbTSFRunner 的差别只在语义约定（数据搬运/契约/反归一化全部复用）：

训练（两阶段，对应设计文档 §5）：
- STAGE='s1' 确定性预训练：模型 forward(train=True) 走回归头，返回
  {'prediction': z1 的解码预测, ...}；cfg.TRAIN.LOSS 配 masked_mae/masked_mse。
  起点仍是锚（模型内部），这一阶段同时是锚质量保证与点预测对照；
- STAGE='s2' 概率阶段（默认）：模型内部采样带噪起点 z0、流时间 t，并插值 x_t=(1-t)z0+t·z1
  + 速度回归字段的构造，返回 {'velocity': v_θ, 'z0': 锚, 'z1': 目标 latent, ...}；
  cfg.TRAIN.LOSS 配 fm_velocity_loss(velocity, z0, z1)（prob_baselines.losses）。
  t 的采样分布（U(0,1) vs logit-normal）由模型按 prob_cfg.T_DIST 决定。

评估：模型 forward(train=False) 内部跑 ODE 采样（步数 prob_cfg.ODE_STEPS、
样本数 prob_cfg.SAMPLE_NUM、锚上扰动 σ₀=prob_cfg.SIGMA0（固定归一化单位）），
返回混合契约字段（prediction_means [S,B,L,N,C] + prediction_sigmas），
闭式 crps_gmm 直接消费——CRPS_sum 等联合口径另走采样评估器。

STAGE 通过 cfg.PROB.STAGE 配置，runner 在构造时保证该键存在（默认 's2'），
模型侧统一从 prob_cfg['STAGE'] 读取，避免每个 arch 各自写默认值。
"""

from easydict import EasyDict
from easytorch.utils import is_master

from .prob_tsf_runner import ProbTSFRunner


class FMProbTSFRunner(ProbTSFRunner):
    """Flow Matching 概率预测 runner（RegionDiT：region bank VAE + DiT + 整流流）。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        # 保证 STAGE 键存在：s1 = 确定性预训练（回归头）；s2 = FM + 节点头（默认）
        if 'STAGE' not in self.prob_cfg:
            self.prob_cfg['STAGE'] = 's2'
        # FM 侧的默认概率超参（模型可逐项覆盖语义；这里只保证键有合理缺省）
        if 'SIGMA0_SCALE' in self.prob_cfg and 'SIGMA0' not in self.prob_cfg:
            raise ValueError('Use SIGMA0 in normalized units; SIGMA0_SCALE is obsolete')
        defaults = {
            'SAMPLE_NUM': 50,       # 推理时 ODE 采样条数 S（混合分量数）
            'ODE_METHOD': 'midpoint',  # midpoint: 2 NFE/step; euler/ab2: 1
            'ODE_STEPS': 20,        # Euler 中点积分步数（A3 消融轴：10/20/50）
            'SIGMA0': 0.5,          # fixed normalized source std, shared by train/eval
            'T_DIST': 'uniform',    # t 采样分布：uniform / logit_normal（G3，默认 U(0,1)）
        }
        for key, value in defaults.items():
            if key not in self.prob_cfg:
                self.prob_cfg[key] = value
        # val 轻量采样档（可选键 VAL_SAMPLE_NUM / VAL_ODE_STEPS，缺省不降档）：
        # Lightweight validation is optional; verify ranking against full sampling.
        # GBA uses full validation with the same sample count and solver as test.
        if any(k in self.prob_cfg for k in ('VAL_SAMPLE_NUM', 'VAL_ODE_STEPS', 'VAL_ODE_METHOD')):
            self._val_prob_override = {}
            if 'VAL_ODE_METHOD' in self.prob_cfg:
                self._val_prob_override['ODE_METHOD'] = self.prob_cfg['VAL_ODE_METHOD']
            if 'VAL_SAMPLE_NUM' in self.prob_cfg:
                self._val_prob_override['SAMPLE_NUM'] = int(self.prob_cfg['VAL_SAMPLE_NUM'])
            if 'VAL_ODE_STEPS' in self.prob_cfg:
                self._val_prob_override['ODE_STEPS'] = int(self.prob_cfg['VAL_ODE_STEPS'])
            self.logger.info('val 轻量采样档: {}（test 仍用 S={}, steps={}）'.format(
                self._val_prob_override, self.prob_cfg['SAMPLE_NUM'], self.prob_cfg['ODE_STEPS']))
        self.logger.info('FMProbTSFRunner prob_cfg: {}'.format(dict(self.prob_cfg)))

    def on_training_end(self, cfg, train_epoch=None):
        """Optionally test final in-memory weights without loading val-best."""
        selection = cfg.get('EVAL', {}).get('MODEL_SELECTION', 'best')
        if selection == 'best':
            return super().on_training_end(cfg, train_epoch)
        if selection != 'last':
            raise ValueError('EVAL.MODEL_SELECTION must be best or last')
        if is_master():
            self.tensorboard_writer.close()
        if 'TEST' in cfg:
            self.logger.info('Evaluating final training weights on the test set (no val-best reload).')
            # If an already-completed run was resumed, train() may pass 1 even
            # though load_model_resume restored the final epoch's weights.
            final_epoch = max(train_epoch or 0, self.start_epoch or 0)
            self.test_pipeline(cfg=cfg, train_epoch=final_epoch,
                               save_metrics=True, save_results=self.save_results)
