import pandas as pd
import numpy as np

from bulian.metrics import compute_metrics
from bulian.metrics.single_table import SingleTableMetric
from bulian.metrics.utils import gauge, gauge_multi

privacyMetrics =[
    'CategoricalCAP',
    # 'CategoricalEnsemble',
    # 'CategoricalGeneralizedCAP',
    # 'CategoricalKNN',
    'CategoricalNB',
    # 'CategoricalPrivacyMetric',
    'CategoricalRF',
    # 'CategoricalSVM',
    # 'CategoricalZeroCAP',
    'NumericalLR',
    'NumericalMLP',
    'NumericalPrivacyMetric',
    # 'NumericalRadiusNearestNeighbor',
    # 'NumericalSVR',
]

def calculate_metrics(real_data, synthetic_data, target=None, key_fields=None, sensitive_fields=None):
    '''
        Helper function to calculate all metrics for single table data
    '''
    _OVERALL_SCORE_GRPS = ['Real vs Synthetic Dectection Metric',
                        'Statistical Test Metric',
                        'Distribution Similarity Metric',
                        'ML Efficacy Metric: R-Sq or F1',
                        'Privacy Metric',
                        ]

    metrics = SingleTableMetric.get_subclasses()

    nonPrivacy_metrics = {k:v for k,v in metrics.items() if k not in privacyMetrics}
    privacy_metrics = {k:v for k,v in metrics.items() if k in privacyMetrics}

    #### Non privacy metrics
    ML_Efficacy = None
    if target is not None:
        ML_Efficacy = compute_metrics(nonPrivacy_metrics,real_data, synthetic_data,target=target)
        ML_Efficacy = ML_Efficacy[ML_Efficacy['MetricType'].isin(['ML Efficacy Metric: R-Sq or F1'])].reset_index(drop=True)

    overall = compute_metrics(nonPrivacy_metrics,real_data, synthetic_data)        
    overall = overall[~overall['MetricType'].isin(['ML Efficacy Metric: R-Sq or F1'])].reset_index(drop=True)

    if ML_Efficacy is not None:
        o = pd.concat([ML_Efficacy,overall],0).reset_index(drop=True)
        del ML_Efficacy
    else:
        o = overall
    
    del overall

    #### Privacy metrics
    if (key_fields is not None) & (sensitive_fields is not None):
        oPriv = compute_metrics(privacy_metrics,real_data, synthetic_data,key_fields=key_fields,sensitive_fields=sensitive_fields)
        if len(oPriv)>0:
            o = pd.concat([o,oPriv],0)
            o = o.reset_index(drop=True)
        del oPriv

    o = o[~np.isnan(o['normalized_score'])]
    o_overall = o[o['MetricType'].isin(_OVERALL_SCORE_GRPS)]
    multi_metrics = o_overall.groupby('MetricType')['normalized_score'].mean().to_dict()

    try:
        avg_efficacy = 100*(o_overall['normalized_score'].mean())
    except:
         ValueError("Some of the Relevant metrics are NaN")
    
    gauge_fig, gauge_value = gauge(avg_efficacy, show_dashboard=True)
    if len(o)>0:
        gauge_multi_fig, gauge_multi_values = gauge_multi(multi_metrics, show_dashboard=True)
    else:
        gauge_multi_fig = None

    return o_overall, gauge_fig, gauge_multi_fig