"""概率时间序列预测通用 runner（ProbTSFRunner）。

设计动机：BasicTS/easytorch 的 loss 与 metric 都通过 metric_forward 按**函数签名**
从 forward 返回的 dict 里自动取参（并自动注入 null_val）。因此概率模型不需要
专用训练循环——只需要让模型 forward 返回"概率字段"，loss/metric 按签名消费即可。
本 runner 在 SimpleTimeSeriesForecastingRunner 的数据搬运基础上，补齐三件事：

1. cfg.PROB 透传：模型 forward 若声明 prob_cfg 参数，则把 cfg['PROB']（EasyDict）
   传入——采样数 S、ODE 步数、σ₀ 尺度、STAGE 等概率超参全部走这里，不侵入模型构造；
2. 概率评估契约（train=False 时）：模型返回 dict 至少给出以下一种组合——
   a) 'prediction' [B,L,N,C]（混合/分布均值，点指标 MAE/RMSE 用）
      + 'prediction_sigma' [B,L,N,C]（单高斯 σ，闭式 CRPS/MIS/PICP 用）
   b) 'prediction' + 'prediction_means' [S,B,L,N,C] + 'prediction_sigmas'
      （S 分量等权高斯混合，闭式 crps_gmm 用——RegionDiT 两级不确定性形态）
   c) 'samples' [S,B,L,N,C]（采样口径；runner 自动折算 prediction=样本均值）
3. 反归一化时的概率字段处理：均值过 scaler.inverse_transform，
   σ 只乘 std（平移无关）——σ 不能加 mean。

训练（train=True）时模型返回什么完全由 cfg.TRAIN.LOSS 的签名决定：
- 点预测改造版（PointProbWrapper）→ gaussian_nll(prediction, prediction_sigma, target)
- RegionDiT FM 版 → fm_velocity_loss(velocity, z0, z1)（prob_baselines.losses）
- RegionDiT S1 确定性阶段 → masked_mae / masked_mse(prediction, target)

训练期指标跳过（本 runner 的关键补丁）：基类 train_iters 会对训练 forward 返回值
逐个计算 CFG.METRICS.FUNCS 里所有指标，而概率指标（如 crps_gmm 需要
prediction_means/prediction_sigmas）只在评估契约下存在——S2 训练 forward 只返回
velocity/z0/z1。本 runner 覆写 metric_forward：指标缺必需参数时返回 None，
train/val iters 对 None 跳过（训练期跑 ODE 采样来喂指标是错误代价，不做）。
评估期（train=False）契约字段齐备，所有指标正常计算。
"""

import inspect
import json
import os
from typing import Dict, Optional

import torch
from easydict import EasyDict
from easytorch.utils import master_only
from tqdm import tqdm

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


def _metric_args_available(metric_func, args: Dict, loss_args: Optional[Dict] = None) -> bool:
    """检查 metric_func 的必需形参（无默认值）能否由 args / loss_args / null_val 提供。

    基类 metric_forward 只做"从 forward 返回 dict 里筛选匹配键"，缺参时直接
    TypeError（正是 crps_gmm 在训练期崩掉的机制）。此处在调用前先判齐备性。
    模块级函数便于独立单测。
    """

    try:
        sig = inspect.signature(metric_func)
    except (ValueError, TypeError):                     # 无法内省的函数：交给基类原逻辑
        return True
    available = set(args.keys())
    if loss_args:
        available.update(loss_args.keys())
    available.add('null_val')                           # 基类会自动注入
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty \
                and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD) \
                and name not in available:
            return False
    return True


class _StrideSampler(torch.utils.data.Sampler):
    """等距子采样 sampler（顺序固定）：遍历 index ∈ {0, stride, 2·stride, ...}。

    用于 val 子采样——生成式管线 val 要跑 ODE 采样（成本 = 推理成本），
    选模型只需 CRPS 排序，等距抽取的固定子集既保代表性（覆盖整个时段）
    又保证各 epoch 的 val/CRPS 严格可比（集合与顺序都不变）。
    """

    def __init__(self, data_length: int, stride: int):
        assert stride >= 1
        self.data_length = data_length
        self.stride = stride

    def __iter__(self):
        return iter(range(0, self.data_length, self.stride))

    def __len__(self):
        return (self.data_length + self.stride - 1) // self.stride


class ProbTSFRunner(BaseTimeSeriesForecastingRunner):
    """概率预测通用 runner。用法见模块 docstring。"""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)
        # 概率超参（可选键：SAMPLE_NUM / ODE_STEPS / SIGMA0_SCALE / STAGE / T_DIST ...）
        self.prob_cfg = cfg.get('PROB', EasyDict())
        if not isinstance(self.prob_cfg, EasyDict):
            self.prob_cfg = EasyDict(self.prob_cfg)
        # val 轻量采样档（FM 系子类按 VAL_SAMPLE_NUM / VAL_ODE_STEPS 填充）：
        # val 只为选模型（CRPS 排序），可用小 S / 少步数的代理采样；
        # test 恒用正式口径。None = val 与 test 同口径（点模型改造版恒为 None）。
        self._val_prob_override: Optional[Dict] = None
        self._in_test = False                      # test() 期间置 True，_call_model 据此区分 val / test

    # ---------------------------------------------------------------- val 子采样
    def build_val_data_loader(self, cfg: Dict):
        """val dataloader：支持 PROB.VAL_STRIDE 等距子采样（缺省 1 = 全量）。

        动机：生成式管线的 val 成本 = ODE 推理成本（S×步数×2 次骨干前向/batch），
        远高于训练；选模型只需 CRPS 排序——等距抽取 1/VAL_STRIDE 的滑窗样本，
        子集固定（顺序 sampler）→ 各 epoch 的 val/CRPS 严格可比。
        DataLoader 的 sampler 与 shuffle 互斥，故此路径不复用 easytorch 的
        build_data_loader（其固定传 shuffle），字段口径与其保持一致。
        """

        from easytorch.core.data_loader import DataLoaderX

        loader = super().build_val_data_loader(cfg)
        stride = int(self.prob_cfg.get('VAL_STRIDE', 1))
        if stride <= 1:
            return loader
        dataset = loader.dataset
        data_cfg = cfg['VAL.DATA']
        n_total = len(dataset)
        n_sub = (n_total + stride - 1) // stride
        self.logger.info(
            'val 子采样 VAL_STRIDE={}: {} → {} 样本（等距、固定子集；test 不受影响）'.format(
                stride, n_total, n_sub))
        return (DataLoaderX if data_cfg.get('PREFETCH', False) else torch.utils.data.DataLoader)(
            dataset,
            collate_fn=data_cfg.get('COLLATE_FN', None),
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            sampler=_StrideSampler(n_total, stride),   # 顺序固定的等距子集
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False))

    # ---------------------------------------------------------------- 数据搬运
    def preprocessing(self, input_data: Dict) -> Dict:
        """scaler 变换（只动 flow 通道，ZScoreScaler 的语义）。"""

        input_data = dict(input_data)
        # Preserve observation masks before zero/NaN sentinels are normalized.
        for key in ('inputs', 'target'):
            raw = input_data[key][..., 0]
            valid = torch.isfinite(raw)
            if self.null_val is not None:
                valid = valid & (raw != self.null_val)
            input_data[key + '_valid'] = valid
            input_data[key] = input_data[key].clone()
        if self.scaler is not None:
            input_data['target'] = self.scaler.transform(input_data['target'])
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
        return input_data

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """按 cfg.MODEL.FORWARD_FEATURES 选输入通道。"""

        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """按 cfg.MODEL.TARGET_FEATURES 选目标通道（返回 [B,L,N,C2]）。"""

        data = data[:, :, :, self.target_features]
        return data

    def _call_model(self, history_data, future_data_4_dec,
                    batch_seen, epoch, train, history_mask=None, future_mask=None):
        """调用模型；若模型 forward 声明了 prob_cfg 形参则透传概率超参。
        val 阶段（train=False 且非 test）若配置了轻量采样档则降档透传——
        ODE 采样成本 = S × 步数 × 2（中点法），val 用小 S/少步数即大幅提速。"""

        try:
            params = inspect.signature(self.model.forward).parameters
        except (ValueError, TypeError):
            params = {}
        kwargs = dict(batch_seen=batch_seen, epoch=epoch, train=train)
        if 'prob_cfg' in params:
            prob_cfg = self.prob_cfg
            if (not train) and (not self._in_test) and self._val_prob_override:
                prob_cfg = dict(self.prob_cfg)               # 浅拷贝：只替换 val 档两项
                prob_cfg.update(self._val_prob_override)
            kwargs['prob_cfg'] = prob_cfg
        if 'history_mask' in params:
            kwargs['history_mask'] = history_mask
        if train and 'future_mask' in params:
            kwargs['future_mask'] = future_mask
        def invoke():
            return self.model(history_data=history_data,
                              future_data=future_data_4_dec, **kwargs)
        if not train and not self._in_test and 'VAL_SEED' in self.prob_cfg:
            # Same validation loader order => same noise per batch each epoch.
            # Restore CPU/CUDA RNG state so validation cannot change training draws.
            devices = list(range(torch.cuda.device_count())) if history_data.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                seed = int(self.prob_cfg['VAL_SEED']) + int(batch_seen or 0)
                torch.random.default_generator.manual_seed(seed)
                for device in devices:
                    torch.cuda.default_generators[device].manual_seed(seed)
                return invoke()
        return invoke()

    # ---------------------------------------------------------------- 训练期指标跳过
    def _metric_args_available(self, metric_func, args: Dict) -> bool:
        """实例薄封装（转发到模块级函数，loss_args 来自 runner 状态）。"""

        return _metric_args_available(metric_func, args, self.loss_args)

    def metric_forward(self, metric_func, args: Dict):
        """参数齐备检查版：缺必需参数（训练期的概率指标）返回 None，由
        train_iters / val_iters 跳过；齐备时走基类原逻辑。"""

        if not self._metric_args_available(metric_func, args):
            return None
        return super().metric_forward(metric_func, args)

    def train_iters(self, epoch: int, iter_index: int, data) -> torch.Tensor:
        """训练迭代：同基类，但跳过参数不齐（返回 None）的训练期指标。
        loss 始终走基类原逻辑（参数不齐应显式报错，而非静默跳过）。
        """

        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        loss = super().metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('train/loss', loss.item(), weight)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            if metric_item is None:                     # 训练期无概率字段 → 跳过
                continue
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), weight)
        return loss

    def val_iters(self, iter_index: int, data):
        """验证迭代：同基类，但①跳过参数不齐的指标；②loss 缺参时跳过
        （FM 系模型评估 forward 只返回采样契约字段，无 velocity/z0/z1——
        'val/loss' meter 会保持 nan，选模型由 CFG.METRICS.TARGET（如 val/CRPS）
        决定，不依赖 val/loss）。"""

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return['target'])
        if loss is not None:
            self.update_epoch_meter('val/loss', loss.item(), weight)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            if metric_item is None:
                continue
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item(), weight)

    # ---------------------------------------------------------------- 测试（概率字段透传）
    @staticmethod
    def _slice_horizon(eval_return: Dict, i: int) -> Dict:
        """把评估返回的各字段切到第 i 个 horizon，供逐 horizon 指标计算：
        [B,L,N,C] → [B,N,C]（切 dim=1）；[S,B,L,N,C] → [S,B,N,C]（切 dim=2）。
        非张量或形状不符的键原样保留（基类口径只切 prediction/target，
        概率字段必须同步切片，否则 crps_gmm 等在 test 上取不到参数）。
        """

        L = eval_return['target'].shape[1]
        out = {}
        for k, v in eval_return.items():
            if torch.is_tensor(v):
                if v.dim() == 4 and v.shape[1] == L:
                    v = v[:, i]                              # prediction/target/inputs/σ
                elif v.dim() == 5 and v.shape[2] == L:
                    v = v[:, :, i]                           # means/sigmas/samples
            out[k] = v
        return out

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False,
             save_results: bool = True) -> Dict:
        """测试过程：同基类，但把**完整概率字段**（prediction_means/sigmas/samples）
        传给指标，并支持逐 horizon 切片。基类 test 只传 {'prediction','target'}——
        概率指标（crps_gmm/mis/picp）在 test 上取不到参数，这是本覆写存在的理由。
        期间置 _in_test=True：test 恒用正式采样口径（val 的轻量档不生效）。
        """

        self._in_test = True
        try:
            for batch_idx, data in enumerate(tqdm(self.test_data_loader)):
                forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

                loss = self.metric_forward(self.loss, forward_return)
                weight = self._get_metric_weight(forward_return['target'])
                if loss is not None:
                    self.update_epoch_meter('test/loss', loss.item(), weight)

                if not self.if_evaluate_on_gpu:
                    eval_return = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                   for k, v in forward_return.items()}
                else:
                    eval_return = forward_return
                pred, target = eval_return['prediction'], eval_return['target']
                if save_results:
                    batch_data = {
                        'prediction': pred.detach().cpu().numpy(),
                        'target': target.detach().cpu().numpy(),
                        'inputs': eval_return['inputs'].detach().cpu().numpy()
                    }
                    self._save_test_results(batch_idx, batch_data)

                # 逐 horizon 指标（概率字段同步切片）
                for i in self.evaluation_horizons:
                    args_h = self._slice_horizon(eval_return, i)
                    weight_h = self._get_metric_weight(args_h['target'])
                    for metric_name, metric_func in self.metrics.items():
                        if metric_name.lower() == 'mase':
                            continue                          # MASE 需全部 horizon 后再算
                        metric_val = self.metric_forward(metric_func, args_h)
                        if metric_val is None:
                            continue
                        self.update_epoch_meter(f'test/{metric_name}@h{i+1}',
                                                metric_val.item(), weight_h)

                # 整体（全 horizon）指标
                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, eval_return)
                    if metric_item is None:
                        continue
                    self.update_epoch_meter(f'test/{metric_name}', metric_item.item(), weight)
        finally:
            self._in_test = False

        if save_metrics:
            def _get_meter(name):
                try:
                    return self.meter_pool.get_value(name)
                except Exception:                            # 跳过的指标无 meter 记录
                    return None
            metrics_results = {'overall': {k: _get_meter(f'test/{k}')
                                           for k in self.metrics.keys()}}
            for i in self.evaluation_horizons:
                metrics_results[f'horizon_{i+1}'] = {
                    k: _get_meter(f'test/{k}@h{i+1}') for k in self.metrics.keys()}
            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)


    # ---------------------------------------------------------------- 主入口
    def forward(self, data: Dict, epoch: Optional[int] = None,
                iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """训练/验证/测试统一入口（同 SimpleTSFRunner 的数据流，概率字段另作处理）。"""
        data = self.preprocessing(data)
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # [B, L, N, C]
        future_data = self.to_running_device(future_data)    # [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data).clone()

        if not train:
            # 评估时抹掉未来的 flow 通道（防泄漏；日历通道保留——未来已知）
            future_data_4_dec[..., 0] = float('nan')

        model_return = self._call_model(history_data, future_data_4_dec,
                                        iter_num, epoch, train,
                                        self.to_running_device(data['inputs_valid']),
                                        self.to_running_device(data['target_valid']) if train else None)

        # 解析模型返回
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)
        model_return['_target_valid'] = self.to_running_device(data['target_valid']).unsqueeze(-1)

        if not train:
            # 概率评估契约归一化（补默认字段、统一形状）
            self._normalize_prob_contract(model_return)

        model_return['_compute_intervals'] = not train and (
            self._in_test or self.prob_cfg.get('VAL_INTERVALS', True))

        model_return = self.postprocessing(model_return)
        return model_return

    # ---------------------------------------------------------------- 概率契约
    def _normalize_prob_contract(self, model_return: Dict) -> None:
        """评估契约归一（就地修改 model_return）：
        - 只有 'samples' [S,B,L,N,C]：折算 prediction=样本均值、
          prediction_means=samples（各分量 σ 用样本 std 近似，若无更好的头）；
        - 只有 'prediction_means'：prediction 默认取分量均值；
        - 校验 'prediction' 存在（点指标必需）。
        """

        if 'samples' in model_return and 'prediction_means' not in model_return:
            samples = model_return['samples']                     # [S,B,L,N,C]
            model_return['prediction_means'] = samples
            if 'prediction' not in model_return:
                model_return['prediction'] = samples.mean(dim=0)
            if 'prediction_sigmas' not in model_return:
                # 无节点头时退化为各分量的样本 std（跨 S，逐位置）
                model_return['prediction_sigmas'] = samples.std(dim=0, unbiased=False)
        elif 'prediction_means' in model_return and 'prediction' not in model_return:
            means = model_return['prediction_means']
            weights = model_return.get('prediction_weights')
            model_return['prediction'] = (means.mean(dim=0) if weights is None else
                                          (means * weights).sum(dim=0))
        assert 'prediction' in model_return, \
            '概率评估契约：模型评估时必须返回 prediction（或 samples/prediction_means）'

    # ---------------------------------------------------------------- 反归一化
    def postprocessing(self, input_data: Dict) -> Dict:
        """反归一化：prediction/target/inputs 走标准 inverse_transform；
        概率字段单独处理——均值正常反变换，σ 只乘 std（σ 平移无关，不能加 mean）。
        """

        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])
            if 'prediction_means' in input_data:
                means = input_data['prediction_means']           # [S,B,L,N,C]
                input_data['prediction_means'] = torch.stack(
                    [self.scaler.inverse_transform(means[s]) for s in range(means.shape[0])], dim=0)
            if 'prediction_sigma' in input_data:
                input_data['prediction_sigma'] = self._rescale_sigma(input_data['prediction_sigma'])
            if 'prediction_sigmas' in input_data:
                input_data['prediction_sigmas'] = self._rescale_sigma(input_data['prediction_sigmas'])
            if 'samples' in input_data:
                samples = input_data['samples']                  # [S,B,L,N,C]
                input_data['samples'] = torch.stack(
                    [self.scaler.inverse_transform(samples[s]) for s in range(samples.shape[0])], dim=0)
            if 'quantile_preds' in input_data:
                # 分位数随仿射变换走：q_real = mean + std * q_norm（逐档保序）
                input_data['quantile_preds'] = self._rescale_quantiles(
                    input_data['quantile_preds'])

        # Restore exact sentinels: inverse-transform roundoff must not turn an
        # originally missing zero into a small nonzero supervised target.
        target_valid = input_data.pop('_target_valid', None)
        if target_valid is not None:
            sentinel = self.null_val if self.null_val is not None else float('nan')
            input_data['target'] = torch.where(
                target_valid, input_data['target'],
                torch.full_like(input_data['target'], sentinel))

        # Compute mixture quantiles once per batch; horizon metrics reuse slices.
        alphas = self.prob_cfg.get('INTERVAL_ALPHAS', ())
        compute_intervals = input_data.pop('_compute_intervals', self._in_test)
        if compute_intervals and alphas:
            if 'prediction_means' in input_data:
                from basicts.metrics import gmm_intervals
                lower, upper = gmm_intervals(input_data['prediction_means'],
                                            input_data['prediction_sigmas'], alphas,
                                            weights=input_data.get('prediction_weights'))
                input_data['prediction_lower'] = lower
                input_data['prediction_upper'] = upper
            elif 'quantile_preds' in input_data:
                # 区间预测版（PointQuantileWrapper）：直接按分位档取区间
                from basicts.metrics import quantile_intervals
                lower, upper = quantile_intervals(input_data['quantile_preds'],
                                                  input_data['quantile_levels'], alphas)
                input_data['prediction_lower'] = lower
                input_data['prediction_upper'] = upper

        # subset forecasting（沿用基类语义）
        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]
            for key in ('prediction_sigma', 'prediction_means', 'prediction_sigmas', 'samples',
                        'prediction_weights', 'prediction_logits',
                        'quantile_preds', 'prediction_lower', 'prediction_upper'):
                if key in input_data:
                    input_data[key] = input_data[key].index_select(
                        -2, torch.as_tensor(self.target_time_series, device=input_data[key].device))
        return input_data

    def _rescale_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """σ_real = σ_norm * std（只乘平移无关的尺度项）。
        std 为标量（NORM_EACH_CHANNEL=False，GBA 即此情形）或带通道形状
        （取 flow 通道后按尾部广播）。"""

        std = getattr(self.scaler, 'std', None)
        if std is None:
            return sigma
        std = torch.as_tensor(std).to(device=sigma.device, dtype=sigma.dtype)
        if std.dim() == 0:
            return sigma * std
        # 形状 std（如 [1,N,C]）：截到 sigma 的通道维再尾部广播
        std_flow = std.unsqueeze(-1) if std.dim() == 2 else std
        return sigma * std_flow

    def _rescale_quantiles(self, quantile_preds: torch.Tensor) -> torch.Tensor:
        """分位数随仿射变换走：q_real = mean + std * q_norm（[Q,B,L,N,C]）。

        逐档保序（正的 std 乘加不改变沿 Q 的排序）；mean/std 为标量
        （GBA）或 [1,N]（逐节点）时均向 [Q,B,L,N,C] 自然广播。
        """

        std = getattr(self.scaler, 'std', None)
        mean = getattr(self.scaler, 'mean', None)
        if std is None:
            return quantile_preds
        q = quantile_preds
        std = torch.as_tensor(std).to(device=q.device, dtype=q.dtype)
        mean = torch.as_tensor(mean).to(device=q.device, dtype=q.dtype)
        if std.dim() == 0:
            return q * std + mean
        # 逐节点统计量（[N] 或 [1,N]）：显式对齐到 N 轴（dim=-2）——
        # 直接尾部广播会错对到 C 轴，产生多余的通道维
        std = std.reshape([1] * (q.dim() - 2) + [-1, 1])
        mean = mean.reshape([1] * (q.dim() - 2) + [-1, 1])
        return q * std + mean
