from .cls_metrics import accuracy
from .corr import masked_corr
from .huber import masked_huber
from .mae import masked_mae
from .mape import masked_mape
from .metric_meter import AvgMeter, RMSEMeter
from .mse import masked_mse
from .prob_metrics import (crps_gaussian, crps_gmm, crps_quantile,
                           mis_gaussian, picp_gaussian,
                           gmm_intervals, picp_interval, width_interval, mis_interval,
                           quantile_intervals, energy_score, variogram_score,
                           energy_score_metric, variogram_score_metric)
from .r_square import masked_r2
from .rmse import masked_rmse
from .smape import masked_smape
from .wape import masked_wape

ALL_METRICS = {
            'MAE': masked_mae,
            'MSE': masked_mse,
            'RMSE': masked_rmse,
            'MAPE': masked_mape,
            'WAPE': masked_wape,
            'SMAPE': masked_smape,
            'R2': masked_r2,
            'CORR': masked_corr,
            'HUBER': masked_huber,
            'Accuracy': accuracy,
            'CRPS_GAUSSIAN': crps_gaussian,
            'CRPS_GMM': crps_gmm,
            'CRPS_QUANTILE': crps_quantile,
            'MIS': mis_gaussian,
            'PICP': picp_gaussian,
            }

METRIC_METER = {
    'RMSE': RMSEMeter,
    'default': AvgMeter
}

__all__ = [
    'masked_mae',
    'masked_mse',
    'masked_rmse',
    'incremental_masked_rmse',
    'masked_mape',
    'masked_wape',
    'masked_smape',
    'masked_r2',
    'masked_corr',
    'masked_huber',
    'accuracy',
    'gmm_intervals', 'picp_interval', 'width_interval', 'mis_interval',
    'quantile_intervals',
    'crps_gaussian',
    'crps_gmm',
    'crps_quantile',
    'mis_gaussian',
    'picp_gaussian',
    'energy_score',
    'variogram_score',
    'energy_score_metric',
    'variogram_score_metric',
    'ALL_METRICS',
    'METRIC_METER'
]