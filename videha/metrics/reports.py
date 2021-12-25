import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import get_correlation_matrix, gauge, gauge_multi
import numpy as np
from ..metrics import compute_metrics
from ..metrics.single_table import *
import pandas as pd

privacyMetrics =[
    'CategoricalCAP',
    'CategoricalEnsemble',
    'CategoricalGeneralizedCAP',
    'CategoricalKNN',
    'CategoricalNB',
    'CategoricalPrivacyMetric',
    'CategoricalRF',
    'CategoricalSVM',
    'CategoricalZeroCAP',
    'NumericalLR',
    'NumericalMLP',
    'NumericalPrivacyMetric',
    'NumericalRadiusNearestNeighbor',
    'NumericalSVR',
]

def get_map(avg_efficacy):
    if avg_efficacy < 0.25:
        return 1
    elif avg_efficacy < 0.5:
        return 2
    elif avg_efficacy < 0.75:
        return 3
    elif avg_efficacy <= 1:
        return 4
    else:
         print(f"Avg Efficacy {avg_efficacy} is not within 0-1 bounds")
         return 0


def get_full_report(real_data, synthetic_data,discrete_columns,numeric_columns,target=None,key_fields=None,sensitive_fields=None):

    ### To do: check no ID columns
    metrics = SingleTableMetric.get_subclasses()

    nonPrivacy_metrics = {k:v for k,v in metrics.items() if k not in privacyMetrics}
    privacy_metrics = {k:v for k,v in metrics.items() if k in privacyMetrics}

    #### Non privacy metrics
    if target is not None:
        o = compute_metrics(nonPrivacy_metrics,real_data, synthetic_data,target=target)
    else:
        o = compute_metrics(nonPrivacy_metrics,real_data, synthetic_data)        

    #### Privacy metrics
    if (key_fields is not None) & (sensitive_fields is not None):
        oP = compute_metrics(privacy_metrics,real_data, synthetic_data,key_fields=key_fields,sensitive_fields=sensitive_fields)
        if len(oP)>0:
            try:
                o = pd.concat([o,oP],0)
                o = o.reset_index(drop=True)
            except:
                warnings.warn("No privary metrics to show!!")

    o = o[~np.isnan(o['normalized_score'])]
    o_min_0 = o[o['min_value']==0.0]    ### what about case where Max value is not 1 but inf
    o_min_neginf = o[o['min_value']==-np.inf]
    multi_metrics_min_0 = o_min_0.groupby('MetricType')['normalized_score'].mean().to_dict()
    multi_metrics_min_neginf = o_min_neginf.groupby('MetricType')['normalized_score'].mean().to_dict()

    efficiency_min_0 = 0
    efficiency_min_neginf = 0 

    if len(o_min_0)>0:
        efficiency_min_0 = (o_min_0.groupby('MetricType')['normalized_score'].mean()>0.5).sum()/(o_min_0['MetricType'].nunique())
    if len(o_min_neginf)>0:
        efficiency_min_neginf = (o_min_neginf.groupby('MetricType')['normalized_score'].mean()>0.0).sum()/(o_min_neginf['MetricType'].nunique())
    
    if (len(o_min_0)>0) & (len(o_min_neginf)>0):
        avg_efficiency = round(100*np.mean((efficiency_min_0,efficiency_min_neginf)))   ### Min: 0, max: 1
    elif (len(o_min_0)>0):
        avg_efficiency = round(100*efficiency_min_0)
    elif (len(o_min_neginf)>0):
        avg_efficiency = round(100*efficiency_min_neginf)
    else:
        raise ValueError("Relevant metrics are NaN")

    gauge(avg_efficiency)
    if len(o_min_neginf)>0:  #### check why this isn;t working when using ML Efficacy charts
        gauge_multi(multi_metrics_min_0)

    # print("\n----------------------------------------- CORRELATION ANALYSIS -----------------------------------------\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Correlation Analysis\n',fontsize = 24,y=1.07)
    syn_corr = get_correlation_matrix(df=synthetic_data, discrete_columns=discrete_columns)
    syn_mask = np.zeros_like(syn_corr, dtype=np.bool)
    syn_mask[np.triu_indices_from(syn_mask)] = True
    sns.heatmap(data=syn_corr,mask = syn_mask,cbar=False,ax=axes[0],cmap='YlGn')
    axes[0].set_title('Synthetic Data Correlation')

    real_corr = get_correlation_matrix(df=real_data, discrete_columns = discrete_columns)
    real_mask = np.zeros_like(real_corr, dtype=np.bool)
    real_mask[np.triu_indices_from(real_mask)] = True
    sns.heatmap(data=real_corr,mask = real_mask,cbar=False,ax=axes[1],cmap='YlGn')
    axes[1].set_title('Real Data Correlation')
    
    diff_corr = np.abs(real_corr)-np.abs(syn_corr)
    diff_mask = np.zeros_like(diff_corr, dtype=np.bool)
    diff_mask[np.triu_indices_from(diff_mask)] = True
    sns.heatmap(data=diff_corr,mask = diff_mask,ax=axes[2],cmap='YlGn')
    axes[2].set_title('Diff (Δ) of Absolute Correlations')
    plt.show()

    for i,numeric_feat in enumerate(numeric_columns):
        plt.figure(figsize=(20,4))
        fig = sns.kdeplot(synthetic_data[numeric_feat], shade=True,label='Synthetic Data')
        fig = sns.kdeplot(real_data[numeric_feat], shade=True, label='Real Data')
        if i == 0:
            plt.title(f"Density Distribution Analysis of Real vs Synthetic Data", fontsize = 26,y=1.3)
            fig.figure.suptitle(f"Numeric Density Distribution : {numeric_feat} ", fontsize = 16,y=1.02)
        else:
            fig.figure.suptitle(f"Numeric Density Distribution : {numeric_feat} ", fontsize = 16,y=1.02)

        plt.xlabel(f'{numeric_feat}', fontsize=10)
        plt.ylabel('Density Distribution', fontsize=10)
        plt.legend(loc='upper right')
        plt.show()

    # print("\n------------------------------------ CATEGORICAL FEATURE DISTRIBUTIONS ------------------------------------\n")
    for categ_feat in discrete_columns:
        plt.figure(figsize=(20,4))
        plt.hist(real_data[categ_feat], 
                label='Real Data',alpha=0.2,density=True)

        plt.hist(synthetic_data[categ_feat], 
                label='Synthetic Data',alpha=0.2,density=True)
        plt.ylabel('Mass Distribution', fontsize=10)
        plt.legend(loc='upper right',)
        plt.title(f'Categorical Density Distribution : {categ_feat}',fontsize=16,y=1.02)
        plt.tick_params(axis='x', rotation=90)
        plt.show()

# if __name__ == '__main__':
#     get_gauge(20)   