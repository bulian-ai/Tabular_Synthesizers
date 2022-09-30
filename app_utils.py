from math import ceil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from plotly.subplots import make_subplots
from bulian.metrics import compute_metrics
from bulian.metrics.single_table import SingleTableMetric
from bulian.metrics.utils import compute_pca, get_correlation_matrix

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

COLORSCALE = px.colors.sequential.Bluyl[::-1] + px.colors.sequential.Bluyl

def gauge(value, show_dashboard=False):
    try:
        value = np.ceil(value)
    except Exception as e:
        print(str(e))

    fig = go.Figure(go.Indicator(        
        mode = "gauge+number+delta",
        value = value,
        number = {'prefix': "<b>Overall Score<b>: ", 'font': {'size': 24,'family': "'Oswald', sans-serif",'color':'white'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 0, 'increasing': {'color': "grey"},'font': {'size': 60}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 0, 'tickcolor': "grey"},
            'bar': {'color': "grey",'thickness':0.3,},
            'bgcolor': "white",
            'borderwidth': 5,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 20], 'color': '#FF8C19'},
                {'range': [20, 40], 'color': '#FFFF80'},                
                {'range': [40, 60], 'color': '#BFFFB3'},
                {'range': [60, 80], 'color': '#BBFF33'},                
                {'range': [80, 101], 'color': '#1EB300'}],
            'threshold': {
                'line': {'color': "grey", 'width': 4},
                'thickness': 0.0,
                'value': 100}}))
    fig.update_layout(xaxis={'showgrid': False, 'showticklabels':True,'range':[0,1]},
                      yaxis={'showgrid': False, 'showticklabels':True,'range':[0,1]},
                     plot_bgcolor='rgba(0,0,0,0)',
                      font = {'color': "white", 'family': "'Oswald', sans-serif"})

    fig.update_layout(
    title={
        'text': "<b>Bulian AI Synthetic Data Quality Report<b>",
        'font': {'size': 32, 'family': "'Oswald', sans-serif", 'color':'#FAFAFA'},
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    if show_dashboard:
        return (fig, value)
    fig.show()

def gauge_multi(MeanDict, show_dashboard=False):
    CountEntities = len(MeanDict)
    sz = 12

    if CountEntities == 0:
        raise ValueError('No Metrics to plot!!')
        
    def getPlotName(key):
        if 'Distribution' in key:
            return 'Similarity Score'
        elif 'Dectection' in key:
            return 'Detection Score'
        elif 'Statistical' in key:
            return 'Statistical Score'
        elif 'Efficacy' in key:
            return 'ML Efficacy Score'
        elif 'Privacy' in key:
            return 'Privacy Score'
        elif 'Likelihood' in key:
            return 'Likelihood Score'

    if CountEntities == 3:
        start = 0
        end = 0.3
        delta = 0.05
        increase = 0.3
    elif CountEntities == 2:
        start = 0
        end = 0.475
        increase = 0.475
        delta = 0.05
    elif CountEntities == 1:
        start = 0
        end = 1
        increase = 0.
        delta = 0.
    else:
        start = 0
        end = 0.2
        delta = 0.05
        increase = 0.2
        
    trace = []
    i = 1
    values = []

    plot_count=0
    for k,v in MeanDict.items():
        if ('Distribution' in k) or ('Dectection' in k) or ('Statistical' in k) or ('Privacy' in k) or ('Efficacy' in k) or ('Likelihood' in k ): 
            plot_count += 1
    if plot_count>4:
        trace = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            ]
        )
        for i, item in enumerate(MeanDict.items(), start=1):
            k = item[0]
            value = item[1]
            if ('Distribution' in k) or ('Dectection' in k) or ('Statistical' in k) or ('Privacy' in k) or ('Efficacy' in k) or ('Likelihood' in k ): 
                name = getPlotName(k)
                values.append(
                    {name: round(value*100)}
                )
                plot = go.Indicator(        
                    mode = "gauge+number+delta",
                    value = round(value*100),
                    number = {'prefix': f"<b>{name}<b>: ", 'font': {'size': sz,'family': "'Oswald', sans-serif",'color':'#FAFAFA'}},
                    delta = {'reference': 0, 'increasing': {'color': "white"},'font': {'size': 12}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 0, 'tickcolor': "white"},
                        'bar': {'color': "grey",'thickness':0.3,},
                        'bgcolor': "white",
                        'borderwidth': 5,
                        'bordercolor': "white",
                        'steps': [
                            {'range': [0, 20], 'color': '#FF8C19'},
                            {'range': [20, 40], 'color': '#FFFF80'},                
                            {'range': [40, 60], 'color': '#BFFFB3'},
                            {'range': [60, 80], 'color': '#BBFF33'},                
                            {'range': [80, 101], 'color': '#1EB300'}],
                        'threshold': {
                            'line': {'color': "grey", 'width': 6},
                            'thickness': 0.0,
                            'value': 100}}
                )
                import math
                if i<4:
                    trace.add_trace(plot, 1, i)
                else:
                    trace.add_trace(plot, 2, i-3)
    else:
        for k,v in MeanDict.items():
            if ('Distribution' in k) or ('Dectection' in k) or ('Statistical' in k) or ('Privacy' in k) or ('Efficacy' in k) or ('Likelihood' in k ): 
                name = getPlotName(k)
                values.append(
                    {name: round(MeanDict[f'{k}']*100)}
                )
                temptrace = go.Indicator(        
                    mode = "gauge+number+delta",
                    value = round(MeanDict[f'{k}']*100),
                    number = {'prefix': f"<b>{name}<b>: ", 'font': {'size': sz,'family': "'Oswald', sans-serif",'color':'#FAFAFA'}},
                    domain = {'x': [start, end], 'y': [0, 1]},
                    delta = {'reference': 0, 'increasing': {'color': "white"},'font': {'size': 12}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 0, 'tickcolor': "white"},
                        'bar': {'color': "grey",'thickness':0.3,},
                        'bgcolor': "white",
                        'borderwidth': 5,
                        'bordercolor': "white",
                        'steps': [
                            {'range': [0, 20], 'color': '#FF8C19'},
                            {'range': [20, 40], 'color': '#FFFF80'},                
                            {'range': [40, 60], 'color': '#BFFFB3'},
                            {'range': [60, 80], 'color': '#BBFF33'},                
                            {'range': [80, 101], 'color': '#1EB300'}],
                        'threshold': {
                            'line': {'color': "grey", 'width': 6},
                            'thickness': 0.0,
                            'value': 100}}
                )
                trace.append(temptrace)
                start = end + delta
                end   = start + increase
                i += 1
                if i > 4:
                    break
    
    # layout and figure production
    layout = go.Layout(height = 300,
                       width  = 1500, 
                       autosize = False,
                       title = '')
    fig = go.Figure(data=trace, layout = layout)
    fig.update_layout(xaxis={'showgrid': False, 'showticklabels':True,'range':[0,1]},
                      yaxis={'showgrid': False, 'showticklabels':True,'range':[0,1]},
                     plot_bgcolor='rgba(0,0,0,0)',
                      font = {'color': "white", 'family': "'Oswald', sans-serif"})
    if show_dashboard:
        return fig, values
    fig.show()

def build_gauge_plots(real_data, synthetic_data, target=None, key_fields=None, sensitive_fields=None):
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

    return gauge_fig, gauge_multi_fig

def build_correlation_plot(real_data, synthetic_data, discrete_columns):
    correlation_fig = make_subplots(
        rows=1,
        cols=3,
        print_grid=False,
        shared_yaxes=True,
        subplot_titles=("Synthetic Data Correlation", "Real Data Correlation", "Absolute Diff (Δ) of Correlations"))
    syn_corr = get_correlation_matrix(df=synthetic_data, discrete_columns=discrete_columns)
    syn_mask = np.triu(np.ones_like(syn_corr, dtype=bool))
    chart = go.Heatmap(z=syn_corr.mask(syn_mask), x=syn_corr.columns.values, y=syn_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
    correlation_fig.add_trace(chart, 1, 1)

    real_corr = get_correlation_matrix(df=real_data, discrete_columns = discrete_columns)
    real_mask = np.triu(np.ones_like(real_corr, dtype=bool))
    chart = go.Heatmap(z=real_corr.mask(real_mask), x=real_corr.columns.values, y=real_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
    correlation_fig.add_trace(chart, 1, 2)

    diff_corr = np.abs(real_corr-syn_corr)
    diff_mask = np.triu(np.ones_like(diff_corr, dtype=bool))
    chart = go.Heatmap(z=diff_corr.mask(diff_mask), x=diff_corr.columns.values, y=diff_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
    correlation_fig.add_trace(chart, 1, 3)
    correlation_fig.update_yaxes(autorange='reversed')
    correlation_fig.update_layout(title=f'<b>Correlation Analysis</b>', title_x=0.5)
    return correlation_fig

def build_pca_plot(real_data, synthetic_data):
    real_pca = compute_pca(real_data)
    synthetic_pca = compute_pca(synthetic_data)

    real_pca_fig = go.Figure(data=go.Scattergl(
        x = real_pca['pc1'],
        y = real_pca['pc2'],
        mode='markers',
        name='Real Data',
        opacity=0.6,
        marker_color='#ffa114',
    ))

    synthetic_pca_fig = go.Figure(data=go.Scattergl(
        x = synthetic_pca['pc1'],
        y = synthetic_pca['pc2'],
        mode='markers',
        name='Synthetic Data',
        opacity=0.6,
        marker_color='#03b1fc',
    ))

    layout = go.Layout(
        height=700,
    )
    pca_fig = go.Figure(data=[real_pca_fig.data[0], synthetic_pca_fig.data[0]], layout=layout)
    pca_fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False, title='Component 1')
    pca_fig.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=False, title='Component 2')
    pca_fig.update_layout(title=f'<b>Prinicpal Component Analysis</b>', title_x=0.5)
    return pca_fig

def build_distribution_plots(real_data, synthetic_data, discrete_columns):
    '''
        Helper function to build numeric and categorical density distribution plots
    '''

    numeric_columns = []
    for x in real_data.columns:
        if x not in discrete_columns:
            numeric_columns.append(x)
    numeric_density_figures = []
    for i, numeric_feat in enumerate(numeric_columns):
        density_fig = ff.create_distplot(
            [synthetic_data[numeric_feat], real_data[numeric_feat]],
            group_labels=['Synthetic Data', 'Real Data'],
            show_hist=False,
            show_rug=False)

        density_fig.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, title='Value')
        density_fig.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, title='Density', showticksuffix='last')
        density_fig.update_layout(
            title=f"<b>Numeric Density Distribution : {numeric_feat}</b>",
            title_x=0.5
        )
        numeric_density_figures.append(density_fig)
    
    category_subplot_titles = []
    if len(discrete_columns)>0:
        for i, categ_feat in enumerate(discrete_columns):
            category_subplot_titles.append(f'{categ_feat}')
        category_feat_plot = make_subplots(
            rows=ceil(len(discrete_columns)/2),
            cols=2,
            subplot_titles=tuple(category_subplot_titles),
        )

        for i, categ_feat in enumerate(discrete_columns, start=1):
            real = go.Histogram(
                x=real_data[categ_feat],
                opacity=0.75,
                histnorm ='percent',
                name='Real Data',
                marker_color='#ffa114',
                legendgroup='Real Data',
                showlegend=True if i==1 else False,
                hovertemplate='%{x} - %{y:.1f}%'
            )
            synthetic = go.Histogram(
                x=synthetic_data[categ_feat],
                opacity=0.75, 
                histnorm ='percent',
                name='Synthetic Data', 
                marker_color='#03b1fc',
                legendgroup='Sythentic Data',
                showlegend=True if i==1 else False,
                hovertemplate='%{x} - %{y:.1f}%'
            )           
            data = [real, synthetic]

            if i%2==0:
                category_feat_plot.add_trace(data[0], ceil(i/2), 2)
                category_feat_plot.add_trace(data[1], ceil(i/2), 2)                    
            else:
                category_feat_plot.add_trace(data[0], ceil(i/2), 1)
                category_feat_plot.add_trace(data[1], ceil(i/2), 1)

        category_feat_plot.update_xaxes(showline=True, linewidth=1, linecolor='black')
        category_feat_plot.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False,title='Proportion %')
        category_feat_plot.update_layout(
            title=f'<b>Categorical Proportion Distribution</b>',
            height=1500,
            title_x=0.5
        )

        return numeric_density_figures, category_feat_plot