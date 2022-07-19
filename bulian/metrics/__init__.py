# -*- coding: utf-8 -*-

"""Top-level package for bulian Metrics."""

import pandas as pd
# from tqdm import tqdm
from ..metrics import (
    column_pairs, demos, type, goal, multi_table, single_column, single_table,demo)
from ..metrics.demo import load_demo
from ..metrics.type import MetricType

__all__ = [
    'demos',
    'load_demo',
    'goal',
    'type',
    'multi_table',
    'column_pairs',
    'single_column',
    'single_table',
    'demo'
    # 'timeseries',
]


def compute_metrics(metrics, real_data, synthetic_data, metadata=None, plot=False,**kwargs):

    """Compute a collection of metrics on the given data.

    Args:
        metrics (list[sdmetrics.base.BaseMetric]):
            Metrics to compute.
        real_data:
            Data from the real dataset
        synthetic_data:
            Data from the synthetic dataset
        metadata (dict):
            Dataset metadata.
        **kwargs:
            Any additional arguments to pass to the metrics.

    Returns:
        pandas.DataFrame:
            Dataframe containing the metric scores, as well as information
            about each metric such as the min and max values and its goal.
    """
    # Only add metadata to kwargs if passed, to stay compatible
    # with metrics that do not expect a metadata argument
    if metadata is not None:
        kwargs['metadata'] = metadata

    scores = []
    for name, metric in (metrics.items()):
        error= None
        try:
            raw_score = metric.compute(real_data, synthetic_data, **kwargs)
            normalized_score = metric.normalize(raw_score)
        except Exception as err:
            raw_score = None
            normalized_score = None
            error = str(err)

        scores.append({
            'metric': name,
            'name': metric.name,
            'raw_score': raw_score,
            'normalized_score': normalized_score,
            'min_value': metric.min_value,
            'max_value': metric.max_value,
            'goal': metric.goal.name,
            'MetricType' : metric.MetricType.value,
            'error': error,
        })
    out = pd.DataFrame(scores)

    # if 'target' not in kwargs:
    #     out = out[out['MetricType']!=MetricType.EFFICACY.value].reset_index(drop=True)
    # else:
    #     out = out[out['MetricType']==MetricType.EFFICACY.value].reset_index(drop=True)
        
    return out


