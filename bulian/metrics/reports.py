import enum
from re import L
from unicodedata import category
from .utils import (
    get_correlation_matrix,
    gauge, gauge_multi,
    compute_pca,
    count_memorized_lines,
    get_numeric_discrete_columns,
    remove_id_fields,
    get_column_name,
    get_metric_info
)
import numpy as np
from ..metrics import compute_metrics
from ..metrics.single_table import *
from ..metrics.multi_table import MultiTableMetric
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
import pandas as pd
from math import ceil
from datetime import datetime
from typing import Dict


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

# Custom color scale
COLORSCALE = px.colors.sequential.Bluyl[::-1] + px.colors.sequential.Bluyl

# To Do: We are not paying attention to whether the goal is to maximize or minimize; we assume it all maximize -- ignore, normalization of metric takes care of this
# Likelyhood metrics are not part of avg efficacy calcs, but are shown in graph

def get_full_report(real_data, synthetic_data, discrete_columns, 
    numeric_columns, target=None, key_fields=None, sensitive_fields=None, show_dashboard=False, port=8050):
    _OVERALL_SCORE_GRPS = ['Real vs Synthetic Dectection Metric',
                            'Statistical Test Metric',
                            'Distribution Similarity Metric',
                            'ML Efficacy Metric: R-Sq or F1',
                            'Privacy Metric',
                            ]

    import warnings
    warnings.filterwarnings('ignore')
    
    ### To do: check no ID columns
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
    if show_dashboard:
        gauge_fig, gauge_value = gauge(avg_efficacy, show_dashboard)
        if len(o)>0:
            gauge_multi_fig, gauge_multi_values = gauge_multi(multi_metrics, show_dashboard)
        else:
            gauge_multi_fig = None
    else:
        gauge(avg_efficacy)
        if len(o)>0:
            gauge_multi(multi_metrics)

    #print("\n----------------------------------------- CORRELATION ANALYSIS -----------------------------------------\n")
    if show_dashboard:
        app = JupyterDash(__name__)
        app.title = 'Bulian AI Synthetic Data Quality Report'
        app._favicon = 'apple-icon-57x57.png'

        colors = {
        'sub-background': '#FFFFFF',
        'background': '#FFFFFF',
        'text': '#000000'
        }

        gauge_fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text'],
        )
        if gauge_multi_fig:
            gauge_multi_fig.update_layout(
                plot_bgcolor=colors['sub-background'],
                paper_bgcolor=colors['sub-background'],
                font_color=colors['text'],
            )

        # Correlation Plots
        correlation_fig = make_subplots(
            rows=1,
            cols=3,
            print_grid=False,
            shared_yaxes=True,
            subplot_titles=("Synthetic Data Correlation", "Real Data Correlation", "Diff (Δ) of Absolute Correlations"))
        syn_corr = get_correlation_matrix(df=synthetic_data, discrete_columns=discrete_columns)
        syn_mask = np.zeros_like(syn_corr, dtype=np.bool)
        syn_mask[np.triu_indices_from(syn_mask)] = True
        syn_corr[syn_mask] = np.nan
        chart = go.Heatmap(z=syn_corr, x=syn_corr.columns, y=syn_corr.columns, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
        correlation_fig.add_trace(chart, 1, 1)

        real_corr = get_correlation_matrix(df=real_data, discrete_columns = discrete_columns)
        real_mask = np.zeros_like(real_corr, dtype=np.bool)
        real_mask[np.triu_indices_from(real_mask)] = True
        real_corr[real_mask] = np.nan
        chart = go.Heatmap(z=real_corr, x=real_corr.columns, y=real_corr.columns, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
        correlation_fig.add_trace(chart, 1, 2)

        diff_corr = np.abs(real_corr-syn_corr)
        diff_mask = np.zeros_like(diff_corr, dtype=np.bool)
        diff_mask[np.triu_indices_from(diff_mask)] = True
        diff_corr[diff_mask] = np.nan
        chart = go.Heatmap(z=diff_corr, x=diff_corr.columns, y=diff_corr.columns, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
        correlation_fig.add_trace(chart, 1, 3)
        correlation_fig.update_yaxes(autorange='reversed')
        correlation_fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['sub-background'],
            font_color=colors['text'],
            coloraxis_colorbar_x=-2
        )

        # Density Distribution
        if len(numeric_columns)>0:
            graph_objects = [
                html.Hr(),
                html.H1("Density Distribution Analysis of Real vs Synthetic Data", id="numeric_density_heading")
            ]
            for i, numeric_feat in enumerate(numeric_columns):
                density_fig = ff.create_distplot(
                    [synthetic_data[numeric_feat], real_data[numeric_feat]],
                    group_labels=['Synthetic Data', 'Real Data'],
                    show_hist=False,
                    show_rug=False)

                density_fig.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, title='Value')
                density_fig.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, title='Density', showticksuffix='last')
                density_fig.update_layout(
                    plot_bgcolor=colors['background'],
                    paper_bgcolor=colors['background'],
                    font_color=colors['text'],
                    title_x=0.5
                )
                graph_objects.append(
                    html.H3(f"Numeric Density Distribution : {numeric_feat}")
                )
                graph_objects.append(
                    dcc.Graph(
                        id=f'density-distribution-{i}',
                        figure=density_fig,
                        style={'width':'75%','margin-left':'auto', 'margin-right':'auto'}
                    )
                )
        
        # Categorical Count Plots
        if len(discrete_columns)>0:
            category_subplot_titles = []
            for i, categ_feat in enumerate(discrete_columns):
                category_subplot_titles.append(f'{categ_feat}')
            category_feat_plot = make_subplots(
                rows=ceil(len(discrete_columns)/2),
                cols=2,
                subplot_titles=tuple(category_subplot_titles),
                specs=[[{}, {}] for x in range(ceil(len(discrete_columns)/2))],
                vertical_spacing=0.11
            )

            for i, categ_feat in enumerate(discrete_columns, start=1):
                real = go.Histogram(
                    x=real_data[categ_feat],
                    opacity=0.75,
                    histnorm ='percent',
                    name='Real Data',
                    marker_color='#e04e14',
                    legendgroup='Real Data',
                    showlegend=True if i==1 else False,
                    hovertemplate='%{x} - %{y:.1f}%'
                )
                synthetic = go.Histogram(
                    x=synthetic_data[categ_feat],
                    histnorm ='percent',
                    opacity=0.75, 
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
                plot_bgcolor=colors['sub-background'],
                paper_bgcolor=colors['sub-background'],
                font_color=colors['text'],
                height=1500
            )
            graph_objects.append(html.Hr())
            graph_objects.append(html.H1('Categorical Proportion Distribution'))
            graph_objects.append(
                dcc.Graph(
                    id=f'category-feat',
                    figure=category_feat_plot,
                    style={'width':'75%','margin-left':'auto', 'margin-right':'auto', 'height':'100%'}
                )
            )


        # Detailed Metrics Table
        metrics_df = o[['metric', 'name', 'normalized_score', 'MetricType']]
        metrics_df = metrics_df.round(2)

        tables = [html.Hr(), html.H1("Detailed Metrics View")]
        tables.append(
            html.Div(
                children=[
                    dash_table.DataTable(
                        metrics_df.to_dict('records'),
                        filter_action="native",
                        columns=[{"name": get_column_name(i), "id": i} for i in metrics_df.columns],
                        sort_action="native",
                        sort_mode="multi",
                        style_header={
                            'backgroundColor': '#0a0c3d',
                            'color': 'white',
                            'text-transform': 'capitalize',
                            'font-weight': 'bold',
                            'font-size': '1.2em',
                            'textAlign': 'center',
                            'border': '1px solid #FAFAFA'                            
                        },
                        style_data={
                            'backgroundColor': '#0a0c3d',
                            'color': 'white',
                            'height': 'auto',
                            'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'border': '1px solid #FAFAFA',
                            'border-collapse': 'collapse'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'width':'85px',
                            'padding': '5px',
                            
                        },
                        style_cell_conditional=[
                            {'if': {'column_id': 'normalized_score'},
                            'width': '20%'},
                            {'if': {'column_id': 'name'},
                            'width': '30%'},
                        ],
                        tooltip_data=[{
                            'name': {'value': get_metric_info(row['metric']), 'type':'markdown'}
                        } for row in metrics_df.to_dict('records')]
                    )
                ],
                style={'width':'70%','margin-left':'auto', 'margin-right':'auto'}
            )
        )

        # Download PDF
        file_name = 'report-'+datetime.utcnow().strftime("%d/%m/%Y-%I:%M")+'.pdf'
        app.clientside_callback(
            """
            function(n_clicks){
                if(n_clicks > 0){
                    var elmnt = document.getElementById("graphs");
                    var opt = {
                        margin:[0,0,0,1],
                        filename: '"""+file_name+"""',
                        image: { type: 'jpeg', quality: 1 },
                        jsPDF: { unit: 'cm', format: 'a2', orientation: 'p', precision:40},
                        pagebreak: { mode: ['avoid-all'], before: 'hr' },
                    };
                    html2pdf().from(elmnt).set(opt).toPdf().get('pdf').then(function(pdf){
                        var image_data = 'data:image/png;base64,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';
                        pdf.setPage(2);
                        pdf.addImage(image_data, 'PNG', 39, 7);
                    }).save();
                }
            }
            """,
            Output('js','n_clicks'),
            Input('js','n_clicks'),
            running=[
                (Output('js', 'disabled'), True, False),
            ]
        )


        # Metric Cards
        if gauge_multi_fig:
            metric_info_cards = [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Similarity Score", className="card-title"),
                                html.P(
                                    "Distribution similarity between real and synthetic data; based on Kullback–Leibler divergence measures.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        id='cards',
                        color='warning'
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Privacy Score", className="card-title"),
                                html.P(
                                    "Numerical and categorical attack models and their ability to predict real data's sensitive attributes.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        color='warning',
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Detection Score", className="card-title"),
                                html.P(
                                    "Quantifies ability of ML algorithms to separate real vs synthetic data. Indicates deep structural stability.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        color='warning',
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Statistical Score", className="card-title"),
                                html.P(
                                    "Statistics based measure to quantify statistical distribution similarity. Based on K-S and C-S tests.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        color='warning',
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("ML Efficacy Score", className="card-title"),
                                html.P(
                                    "Compare the performance of ML models when run on the synthetic and real data. Based on F1 or R-sq scores.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        color='warning',
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Likelihood Score", className="card-title"),
                                html.P(
                                    "Metrics which learn the distribution of the real data and evaluate the likelihood of the synthetic data.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        color='warning',
                    ),
                    width=4
                )
                
                ]
        else:
            metric_info_cards = []

        for i, row in enumerate(metrics_df.to_dict('records')):
            metric_name = row['metric']
            info = get_metric_info(row['metric'])
            
            element = dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4(metric_name, className="card-title"),
                            html.P(
                                info,
                                className="card-text",
                            ),
                        ]
                    ),
                    color="warning"
                ),
                width=4,
            )
            metric_info_cards.append(element)
        metric_info_cards = [metric_info_cards[i:i+3] for i in range(0, len(metric_info_cards), 3)]

        metric_info_cards_rows = []
        for elements in metric_info_cards:
            metric_info_cards_rows.append(
                dbc.Row(elements)
            )
        metric_info_div = [
            html.Br(),
            html.H1("Metric Cards"),
            html.Div(
                children=metric_info_cards_rows,
                id='metric_info_div'
            ),
            html.A(
                id='docs_btn',
                href="https://docs.bulian.ai/bulianai-overview/api-docs/getting-started-with-bulian-ai/metrics",
                target="_blank",
                children=[
                    dbc.Button("View Docs", color="warning", className="me-1"),
                ],
            ),
        ]

        # Multi Gauge plots
        if gauge_multi_fig:
            gauge_multi_figures = []
            plot_count = len(gauge_multi_values)
            row = []
            row_2 = []
            if plot_count>4:
                for i, item in enumerate(gauge_multi_values, start=1):
                    name = list(item.keys())[0]
                    value = list(item.values())[0]
                    if i>3:
                            row_2.append(
                                    dbc.Col(daq.Gauge(
                                        id=f'gauge-{name}',
                                        label=name,
                                        labelPosition='bottom',
                                        value=int(value),
                                        max=100,
                                        min=0,
                                        showCurrentValue=True,
                                        color={"gradient":True,"ranges":{"red":[0,33],"yellow":[33,66],"green":[66,100]}},
                                        style={'font-weight':'bold', 'font-size': '1.2em'}
                                )
                            ))
                    else:
                        row.append(
                                dbc.Col(daq.Gauge(
                                    id=f'gauge-{name}',
                                    label=name,
                                    labelPosition='bottom',
                                    value=int(value),
                                    max=100,
                                    min=0,
                                    showCurrentValue=True,
                                    color={"gradient":True,"ranges":{"red":[0,33],"yellow":[33,66],"green":[66,100]}},
                                    style={'font-weight':'bold', 'font-size': '1.2em'}
                            )
                        ))
                gauge_multi_figures.append(dbc.Row(row, id='metrics_row'))
                gauge_multi_figures.append(dbc.Row(row_2, id='metrics_row_2'))
            else:
                for item in gauge_multi_values:
                    name = list(item.keys())[0]
                    value = list(item.values())[0]
                    row.append(
                            dbc.Col(daq.Gauge(
                                id=f'gauge-{name}',
                                label=name,
                                labelPosition='bottom',
                                value=int(value),
                                max=100,
                                min=0,
                                showCurrentValue=True,
                                color={"gradient":True,"ranges":{"red":[0,33],"yellow":[33,66],"green":[66,100]}},
                                style={'font-weight':'bold', 'font-size': '1.2em'}
                        )
                    ))
                gauge_multi_figures.append(dbc.Row(row, id='metrics_row'))

            gauge_multi_figures.append(
                html.A(
                    id='learn_more_btn',
                    href="#cards",
                    children=[
                        dbc.Button("Metric Cards", color="warning", className="me-1"),
                    ],
                ),
            )
        else:
            gauge_multi_figures = []
        
        # Data summary table
        table_header = [
            html.Thead(html.Tr([html.Th(""), html.Th("Real Data"), html.Th("Synthetic Data")]))
        ]
        row1 = html.Tr([html.Td("Row Count"), html.Td(len(real_data)), html.Td(len(synthetic_data))])
        row2 = html.Tr([html.Td("Column Count"), html.Td(len(real_data.columns)), html.Td(len(synthetic_data.columns))])
        row3 = html.Tr([html.Td("Duplicated lines"), html.Td(len(real_data)-len(real_data.drop_duplicates())), html.Td(count_memorized_lines(real_data, synthetic_data))])
        table_body = [html.Tbody([row1, row2, row3])]

        # Timestamp for report
        date_time = 'Generated on \n'+ datetime.utcnow().strftime("%d/%m/%Y, %I:%M %p") + ' UTC'

        # PCA plots
        real_pca = compute_pca(real_data)
        synthetic_pca = compute_pca(synthetic_data)

        real_pca_fig = go.Figure(data=go.Scattergl(
            x = real_pca['pc1'],
            y = real_pca['pc2'],
            mode='markers',
            name='Real Data',
            opacity=0.6,
            marker_color='#e04e14',
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
        pca_fig.update_layout(
            plot_bgcolor=colors['sub-background'],
            paper_bgcolor=colors['sub-background'],
            font_color=colors['text'],
        )

        # Main dash app layout
        app.layout = html.Div(
            style={
                'backgroundColor':colors['background'],
            },
            children=[
                dbc.Button("Download PDF", color="warning", className="me-1", id='js', n_clicks=0),
                html.Div(
                    id='graphs',
                    children=[
                        html.H1("Bulian AI Synthetic Data Quality Report", id='main_heading'),
                        html.H4(date_time, id='date_time'),
                        html.Div(
                            id='gauge-div',
                            children=[
                                daq.Gauge(
                                    id='gauge',
                                    label="Synthetic Data Quality Score",
                                    labelPosition='bottom',
                                    value=int(gauge_value),
                                    max=100,
                                    min=0,
                                    size=400,
                                    showCurrentValue=True,
                                    color={"gradient":True,"ranges":{"red":[0,33],"yellow":[33,66],"green":[66,100]}},
                                )],
                        ),
                        html.Div(
                            id='gauge-multi-div',
                            style={'align': 'center', 'margin-bottom':'2em'}, 
                            children=gauge_multi_figures
                        ),
                        html.H3('Data Summary Statistics', id='data_summary_heading'),
                        dbc.Table(table_header + table_body, bordered=True, id='data_summary_table', size='sm'),
                        html.Hr(),
                        html.Div(
                            id='correlation-div',
                            style={'align': 'center'},
                            children=[
                                html.H1("Correlation Analysis of Real vs Synthetic Data"),
                                dcc.Graph(
                                    id='correlation-graphs',
                                    figure=correlation_fig,
                                    style={'width':'75%','margin-left':'auto', 'margin-right':'auto'}
                                ),
                            ]
                        ),
                        html.Div(id="divider"),
                        html.Div(
                            id='pca-div',
                            style={'align': 'center'},
                            children=[
                                html.H1("PCA Overlap: Real vs Synthetic Data"),
                                dcc.Graph(
                                    id='pca-graph',
                                    figure=pca_fig,
                                )
                            ]
                        ),
                        dbc.Tooltip(
                            "Higher overlap between Real and Synthetic Data PCA components represents higher structural stability",
                            target="pca-div",
                        ),
                        dbc.Tooltip(
                            "The overall score represents the utility score or confidence score for Synthetic Datasets.",
                            target="gauge",
                        ),
                        dbc.Tooltip(
                            "Distribution similarity between Real and Synthetic Data; based on Kullback–Leibler divergence measures.",
                            target="gauge-Similarity Score",
                        ),
                        dbc.Tooltip(
                            "Numerical and categorical attack models and their ability to predict Real Data's sensitive attributes.",
                            target="gauge-Privacy Score",
                        ),
                        dbc.Tooltip(
                            "Quantifies ability of ML algorithms to separate Real vs Synthetic Data. Indicates deep structural stability.",
                            target="gauge-Detection Score",
                        ),
                        dbc.Tooltip(
                            "Statistics based measure to quantify statistical distribution similarity. Based on K-S and C-S tests.",
                            target="gauge-Statistical Score",
                        ),
                        dbc.Tooltip(
                            "Generic ML Efficacy metric that detects the type of ML Problem associated with the Dataset by analyzing the target column type and then applies all the metrics that are compatible with it.",
                            target="gauge-ML Efficacy Score",
                        ),
                        dbc.Tooltip(
                            "Metrics which learn the distribution of the Real Data and evaluate the likelihood of the Synthetic Data belonging to the learned distribution.",
                            target="gauge-Likelihood Score",
                        ),
                    ]+graph_objects+tables+metric_info_div
                )
            ])
        app.run_server(debug=False, port=port)
    else:
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
        correlation_fig.show()


        # PCA plot
        real_pca = compute_pca(real_data)
        synthetic_pca = compute_pca(synthetic_data)

        real_pca_fig = go.Figure(data=go.Scattergl(
            x = real_pca['pc1'],
            y = real_pca['pc2'],
            mode='markers',
            name='Real Data',
            opacity=0.6,
            marker_color='#e04e14',
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
        pca_fig.show()

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
            density_fig.show()

        # Categorical Count Plots
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
                    marker_color='#e04e14',
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
            category_feat_plot.show()

def get_multi_table_report(real_data, synthetic_data, metadata, numeric_features:Dict[str, str]=None, discrete_features:Dict[str, str]=None, show_dashboard=False, port=8050):
    """Multi Table Data Quality Report.
    This API provides data quality report metrics for multi table /relational methods

    Args:
        real_data (Dict(pandas.DataFrame)):
            Dictionary containing related tables as pandas.DataFrame

        synthetic_data (pandas.DataFrame):
            Dictionary containing related tables as pandas.DataFrame
        
        metadata (bulian.metadata.dataset.Metadata):
            Metadata object for the related relational data tables

        numeric_features (Dict):
            Dictionary with keys as table name and values as a list of numeric column names
    
        discrete_features (Dict):
            Dictionary with keys as table name and values as a list of discrete column names
        
        show_dashboard (Boolean):
            Enable or disable dashboard application

        port (int):
            Change the default port for the Dash local server
    """
    metrics = MultiTableMetric.get_subclasses()
    table_names = list(real_data.keys())

    overall = compute_metrics(metrics, real_data, synthetic_data, metadata)
    overall = overall.dropna(subset=['normalized_score'])

    real_data = remove_id_fields(real_data, metadata)
    synthetic_data = remove_id_fields(synthetic_data, metadata)

    try:
        avg_efficacy = 100*(overall['normalized_score'].mean())
    except ValueError:
        ValueError("Some of the Relevant metrics are NaN")
    multi_metrics = overall.groupby('MetricType')['normalized_score'].mean().to_dict()

    if show_dashboard:
        gauge_fig, gauge_value = gauge(avg_efficacy, show_dashboard)
        if len(multi_metrics)>0:
            gauge_multi_fig, gauge_multi_values = gauge_multi(multi_metrics, show_dashboard)
    else:
        gauge(avg_efficacy, show_dashboard)
        if len(multi_metrics)>0:
            gauge_multi(multi_metrics, show_dashboard)

    if show_dashboard:
        app = JupyterDash(__name__)
        app.title = 'Bulian AI Synthetic Data Quality Report'
        app._favicon = 'apple-icon-57x57.png'
        
        # Timestamp for report
        date_time = 'Generated on \n'+ datetime.utcnow().strftime("%d/%m/%Y, %I:%M %p") + ' UTC'

        # Multi Gauge plots
        if gauge_multi_fig:
            gauge_multi_figures = []
            plot_count = len(gauge_multi_values)
            row = []
            row_2 = []
            if plot_count>4:
                for i, item in enumerate(gauge_multi_values, start=1):
                    name = list(item.keys())[0]
                    value = list(item.values())[0]
                    if i>3:
                            row_2.append(
                                    dbc.Col(daq.Gauge(
                                        id=f'gauge-{name}',
                                        label=name,
                                        labelPosition='bottom',
                                        value=int(value),
                                        max=100,
                                        min=0,
                                        showCurrentValue=True,
                                        color={"gradient":True,"ranges":{"red":[0,33],"yellow":[33,66],"green":[66,100]}},
                                        style={'font-weight':'bold', 'font-size': '1.2em'}
                                )
                            ))
                    else:
                        row.append(
                                dbc.Col(daq.Gauge(
                                    id=f'gauge-{name}',
                                    label=name,
                                    labelPosition='bottom',
                                    value=int(value),
                                    max=100,
                                    min=0,
                                    showCurrentValue=True,
                                    color={"gradient":True,"ranges":{"red":[0,33],"yellow":[33,66],"green":[66,100]}},
                                    style={'font-weight':'bold', 'font-size': '1.2em'}
                            )
                        ))
                gauge_multi_figures.append(dbc.Row(row, id='metrics_row'))
                gauge_multi_figures.append(dbc.Row(row_2, id='metrics_row_2'))
            else:
                for item in gauge_multi_values:
                    name = list(item.keys())[0]
                    value = list(item.values())[0]
                    row.append(
                            dbc.Col(daq.Gauge(
                                id=f'gauge-{name}',
                                label=name,
                                labelPosition='bottom',
                                value=int(value),
                                max=100,
                                min=0,
                                showCurrentValue=True,
                                color={"gradient":True,"ranges":{"red":[0,33],"yellow":[33,66],"green":[66,100]}},
                                style={'font-weight':'bold', 'font-size': '1.2em'}
                        )
                    ))
                gauge_multi_figures.append(dbc.Row(row, id='metrics_row'))
            gauge_multi_figures.append(
                html.A(
                    id='learn_more_btn',
                    href="#cards",
                    children=[
                        dbc.Button("Metric Cards", color="warning", className="me-1"),
                    ],
                ),
            )

        # Correlation Plots
        correlation_figures = []
        for i, table_name in enumerate(table_names):
            current_real_data = real_data[table_name]
            current_synthetic_data = synthetic_data[table_name]

            if not discrete_features:
                discrete_columns = get_numeric_discrete_columns(current_real_data)[1]
            else:
                discrete_columns = discrete_features[table_name]
                    
            correlation_fig = make_subplots(
                rows=1,
                cols=3,
                print_grid=False,
                shared_yaxes=True,
                subplot_titles=("Synthetic Data Correlation", "Real Data Correlation", "Absolute Diff (Δ) of Correlations"))
            syn_corr = get_correlation_matrix(df=current_synthetic_data, discrete_columns=discrete_columns)
            syn_mask = np.triu(np.ones_like(syn_corr, dtype=bool))
            chart = go.Heatmap(z=syn_corr.mask(syn_mask), x=syn_corr.columns.values, y=syn_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
            correlation_fig.add_trace(chart, 1, 1)

            real_corr = get_correlation_matrix(df=current_real_data, discrete_columns=discrete_columns)
            real_mask = np.triu(np.ones_like(real_corr, dtype=bool))
            chart = go.Heatmap(z=real_corr.mask(real_mask), x=real_corr.columns.values, y=real_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
            correlation_fig.add_trace(chart, 1, 2)

            diff_corr = np.abs(real_corr-syn_corr)
            diff_mask = np.triu(np.ones_like(diff_corr, dtype=bool))
            chart = go.Heatmap(z=diff_corr.mask(diff_mask), x=diff_corr.columns.values, y=diff_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
            correlation_fig.add_trace(chart, 1, 3)
            correlation_fig.update_yaxes(autorange='reversed')
            correlation_fig.update_layout(title=f'<b>Table - "{table_name}"</b>', title_x=0.5)
            if i!=0:
                correlation_fig.update_traces(showscale=False)
            correlation_figures.append(dcc.Graph(
                            id=f'correlation-graphs-{i}',
                            figure=correlation_fig,
                            style={'width':'75%','margin-left':'auto', 'margin-right':'auto'}
            ))

        # PCA Plots
        subplot_titles = []
        for name in table_names:
            subplot_titles.append(f'<b>Table - "{name}"</b>')

        pca_plot = make_subplots(
            rows=ceil(len(table_names)/2),
            cols=2,
            subplot_titles=tuple(subplot_titles)
        )

        for i, table_name in enumerate(table_names, start=1):
            current_real_data = real_data[table_name]
            current_synthetic_data = synthetic_data[table_name]

            # Removing datetime column
            current_real_data = current_real_data.select_dtypes(exclude=['datetime64'])
            current_synthetic_data = current_synthetic_data.select_dtypes(exclude=['datetime64'])

            real_pca = compute_pca(current_real_data)
            synthetic_pca = compute_pca(current_synthetic_data)

            real_pca_fig = go.Scattergl(
                x = real_pca['pc1'],
                y = real_pca['pc2'],
                mode='markers',
                name='Real Data',
                opacity=0.6,
                marker_color='#e04e14',
                showlegend=True if i==1 else False
            )
            synthetic_pca_fig = go.Scattergl(
                x = synthetic_pca['pc1'],
                y = synthetic_pca['pc2'],
                mode='markers',
                name='Synthetic Data',
                opacity=0.6,
                marker_color='#03b1fc',
                showlegend=True if i==1 else False
            )
            if i%2==0:
                pca_plot.add_trace(real_pca_fig, ceil(i/2), 2)
                pca_plot.add_trace(synthetic_pca_fig, ceil(i/2), 2)                    
            else:
                pca_plot.add_trace(real_pca_fig, ceil(i/2), 1)
                pca_plot.add_trace(synthetic_pca_fig, ceil(i/2), 1)
        pca_plot.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False, title='Component 1')
        pca_plot.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=False, title='Component 2')
        pca_plot.update_layout(height=300*len(table_names))

        # Numeric Desnity
        numeric_plots = []
        for i, table_name in enumerate(table_names):
            current_real_data = real_data[table_name]
            current_synthetic_data = synthetic_data[table_name]

            if not numeric_features:
                numeric_columns = get_numeric_discrete_columns(current_real_data)[0]
            else:
                numeric_columns = numeric_features[table_name]
            
            if len(numeric_columns)==0:
                break
            
            numeric_subplot_titles = []
            for i, numeric_feat in enumerate(numeric_columns):
                numeric_subplot_titles.append(f'{numeric_feat}')

            numeric_subplot = make_subplots(
                rows=len(numeric_columns),
                cols=1,
                subplot_titles=tuple(numeric_subplot_titles),
                vertical_spacing=(1/(len(numeric_columns)-1)) if len(numeric_columns)>1 else 0
            )

            for i, numeric_feat in enumerate(numeric_columns, start=1):
                density_fig = ff.create_distplot(
                    [current_synthetic_data[numeric_feat], current_real_data[numeric_feat]],
                    group_labels=['Synthetic Data', 'Real Data'],
                    show_hist=False,
                    show_rug=False)
                numeric_subplot.add_trace(density_fig.data[0], i, 1)
                numeric_subplot.add_trace(density_fig.data[1], i, 1)
            numeric_subplot.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, title='Value')
            numeric_subplot.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, title='Density', showticksuffix='last')
            numeric_subplot.update_layout(title=f'<b>Table - "{table_name}"</b>', title_x=0.5)
            numeric_plots.append(dcc.Graph(
                figure=numeric_subplot,
                style={'width':'75%','margin-left':'auto', 'margin-right':'auto'}
            ))

        # Categorical Count Plots
        categorical_count_plots = []
        for table_name in table_names:
            current_real_data = real_data[table_name]
            current_synthetic_data = synthetic_data[table_name]

            if not discrete_features:
                discrete_columns = get_numeric_discrete_columns(current_real_data)[1]
            else:
                discrete_columns = discrete_features[table_name]

            if len(discrete_columns)==0:
                break

            category_subplot_titles = []
            for i, categ_feat in enumerate(discrete_columns):
                category_subplot_titles.append(f'{categ_feat}')
            category_feat_plot = make_subplots(
                rows=ceil(len(discrete_columns)/2),
                cols=2,
                subplot_titles=tuple(category_subplot_titles),
                specs=[[{}, {}] for x in range(ceil(len(discrete_columns)/2))],
                vertical_spacing=(1/(ceil(len(discrete_columns)/2-1))) if len(discrete_columns)>2 else 0
            )

            for i, categ_feat in enumerate(discrete_columns, start=1):
                real = go.Histogram(
                    x=current_real_data[categ_feat],
                    opacity=0.75,
                    histnorm ='percent',
                    name='Real Data',
                    marker_color='#e04e14',
                    legendgroup='Real Data',
                    showlegend=True if i==1 else False,
                    hovertemplate='%{x} - %{y:.1f}%'
                )
                synthetic = go.Histogram(
                    x=current_synthetic_data[categ_feat],
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
                title=f'<b>Table - "{table_name}"</b>',
                title_x=0.5
            )
            categorical_count_plots.append(
                dcc.Graph(
                    figure=category_feat_plot,
                    style={'width':'75%','margin-left':'auto', 'margin-right':'auto'}
                )
            )

        # Detailed Metrics Table
        metrics_df = overall[['metric', 'name', 'normalized_score', 'MetricType']]
        metrics_df = metrics_df.round(2)

        metric_table = [html.Hr(), html.H1("Detailed Metrics View")]
        metric_table.append(
            html.Div(
                children=[
                    dash_table.DataTable(
                        metrics_df.to_dict('records'),
                        filter_action="native",
                        columns=[{"name": get_column_name(i), "id": i} for i in metrics_df.columns],
                        sort_action="native",
                        sort_mode="multi",
                        style_header={
                            'backgroundColor': '#0a0c3d',
                            'color': 'white',
                            'text-transform': 'capitalize',
                            'font-weight': 'bold',
                            'font-size': '1.2em',
                            'textAlign': 'center',
                            'border': '1px solid #FAFAFA'                            
                        },
                        style_data={
                            'backgroundColor': '#0a0c3d',
                            'color': 'white',
                            'height': 'auto',
                            'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'border': '1px solid #FAFAFA',
                            'border-collapse': 'collapse'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'width':'85px',
                            'padding': '5px',
                            
                        },
                        style_cell_conditional=[
                            {'if': {'column_id': 'normalized_score'},
                            'width': '20%'},
                            {'if': {'column_id': 'name'},
                            'width': '30%'},
                        ]
                    )
                ],
                style={'width':'70%','margin-left':'auto', 'margin-right':'auto'}
            )
        )

        # Metric Cards
        if gauge_multi_fig:
            metric_info_cards = [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Detection Score", className="card-title"),
                                html.P(
                                    "Quantifies ability of ML algorithms to separate real vs synthetic data. Indicates deep structural stability.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        color='warning',
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Statistical Score", className="card-title"),
                                html.P(
                                    "Statistics based measure to quantify statistical distribution similarity. Based on K-S and C-S tests.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        color='warning',
                    ),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Likelihood Score", className="card-title"),
                                html.P(
                                    "Metrics which learn the distribution of the real data and evaluate the likelihood of the synthetic data.",
                                    className="card-text",
                                    style={'font-size': '14px'}
                                ),
                            ]
                        ),
                        color='warning',
                    ),
                    width=4
                )
                
                ]
        else:
            metric_info_cards = []

        for i, row in enumerate(metrics_df.to_dict('records')):
            metric_name = row['metric']
            info = get_metric_info(row['metric'])
            
            element = dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4(metric_name, className="card-title"),
                            html.P(
                                info,
                                className="card-text",
                            ),
                        ]
                    ),
                    color="warning"
                ),
                width=4,
            )
            metric_info_cards.append(element)
        metric_info_cards = [metric_info_cards[i:i+3] for i in range(0, len(metric_info_cards), 3)]

        metric_info_cards_rows = []
        for elements in metric_info_cards:
            metric_info_cards_rows.append(
                dbc.Row(elements)
            )
        metric_info_div = [
            html.Br(),
            html.H1("Metric Cards"),
            html.A(id='cards'),
            html.Div(
                children=metric_info_cards_rows,
                id='metric_info_div'
            ),
            html.A(
                id='docs_btn',
                href="https://docs.bulian.ai/bulianai-overview/api-docs/getting-started-with-bulian-ai/metrics",
                target="_blank",
                children=[
                    dbc.Button("View Docs", color="warning", className="me-1"),
                ],
            ),
        ]

        # Download PDF
        # TODO - Check other libraries with 
        file_name = 'report-'+datetime.utcnow().strftime("%d/%m/%Y-%I:%M")+'.pdf'
        app.clientside_callback(
            """
            function(n_clicks){
                if(n_clicks > 0){
                    var elmnt = document.getElementById("graphs");
                    var opt = {
                        margin:[0,0,0,1],
                        filename: '"""+file_name+"""',
                        image: { type: 'jpeg', quality: 1 },
                        jsPDF: { unit: 'cm', format: 'a2', orientation: 'p', precision:40},
                        html2canvas:  { dpi: 192, letterRendering: true},
                        pagebreak: { mode: ['avoid-all'], before: 'hr' },
                    };
                    html2pdf().from(elmnt).set(opt).toPdf().get('pdf').then(function(pdf){
                        var image_data = 'data:image/png;base64,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';
                        pdf.setPage(2);
                        pdf.addImage(image_data, 'PNG', 39.5, 6.7);
                    }).save();
                }
            }
            """,
            Output('js','n_clicks'),
            Input('js','n_clicks'),
            running=[
                (Output('js', 'disabled'), True, False),
            ]
        )

        # Data summary table
        summary_tables = []
        for i, table in enumerate(table_names):
            if i==0:
                table_header = [
                    html.Thead(html.Tr([html.Th(f'Table - "{table}"'), html.Th("Real Data"), html.Th("Synthetic Data")]))
                ]
            else:
                table_header = [
                    html.Thead(html.Tr([html.Th(f'Table - "{table}"')]))
                ]
            row1 = html.Tr([html.Td("Row Count"), html.Td(len(real_data[table])), html.Td(len(synthetic_data[table]))])
            row2 = html.Tr([html.Td("Column Count"), html.Td(len(real_data[table].columns)), html.Td(len(synthetic_data[table].columns))])
            row3 = html.Tr([html.Td("Duplicated lines"), html.Td(len(real_data[table])-len(real_data[table].drop_duplicates())), html.Td(count_memorized_lines(real_data[table], synthetic_data[table]))])
            table_body = [html.Tbody([row1, row2, row3])]
            summary_tables += table_header+table_body
        
        app.layout = html.Div(
            children=[
                dbc.Button("Download PDF", color="warning", className="me-1", id='js', n_clicks=0),
                html.Div(
                    id='graphs',
                    children=[
                        html.H1("Bulian AI Synthetic Data Quality Report", id='main_heading'),
                        html.H4(date_time, id='date_time'),
                        html.Div(
                            id='gauge-div',
                            children=[
                                daq.Gauge(
                                    id='gauge',
                                    label="Synthetic Data Quality Score",
                                    labelPosition='bottom',
                                    value=int(gauge_value),
                                    max=100,
                                    min=0,
                                    size=400,
                                    showCurrentValue=True,
                                    color={"gradient":True,"ranges":{"red":[0,33],"yellow":[33,66],"green":[66,100]}},
                                )],
                        ),
                        html.Div(
                            id='gauge-multi-div',
                            style={'align': 'center', 'margin-bottom':'2em'}, 
                            children=gauge_multi_figures
                        ),
                        html.H3('Data Summary Statistics', id='data_summary_heading'),
                        dbc.Table(summary_tables, bordered=True, id='data_summary_table', size='sm'),
                        html.Hr(),
                        html.Div(
                            id='correlation-div',
                            style={'align': 'center'},
                            children=[
                                html.H1("Correlation Analysis of Real vs Synthetic Data"),
                                html.Div(children=correlation_figures)
                            ]
                        ),
                        html.Div(id="divider"),
                        html.Div(
                            id='pca-div',
                            style={'align': 'center'},
                            children=[
                                html.H1("PCA Overlap: Real vs Synthetic Data"),
                                dcc.Graph(
                                    id='pca-graph',
                                    figure=pca_plot
                                )
                            ]
                        ),
                        html.Hr(),
                        html.H1("Density Distribution Analysis of Real vs Synthetic Data", id="numeric_density_heading") if len(numeric_plots)>0 else html.H1(),
                        html.Div(
                            children=numeric_plots
                        ),
                        html.Hr(),
                        html.H1("Categorical Proportion Distribution") if len(categorical_count_plots)>0 else html.H1(),
                        html.Div(
                            children=categorical_count_plots
                        ),
                        dbc.Tooltip(
                            "Higher overlap between Real and Synthetic Data PCA components represents higher structural stability",
                            target="pca-graph",
                        ),
                        dbc.Tooltip(
                            "The overall score represents the utility score or confidence score for Synthetic Datasets.",
                            target="gauge",
                        ),
                        dbc.Tooltip(
                            "Quantifies ability of ML algorithms to separate Real vs Synthetic Data. Indicates deep structural stability.",
                            target="gauge-Detection Score",
                        ),
                        dbc.Tooltip(
                            "Statistics based measure to quantify statistical distribution similarity. Based on K-S and C-S tests.",
                            target="gauge-Statistical Score",
                        ),
                        dbc.Tooltip(
                            "Metrics which learn the distribution of the Real Data and evaluate the likelihood of the Synthetic Data belonging to the learned distribution.",
                            target="gauge-Likelihood Score",
                        )]+metric_table+metric_info_div
                )]
        )
        app.run_server(debug=True, port=port)
    else:
        # Correlation Plots
        for table_name in table_names:
            current_real_data = real_data[table_name]
            current_synthetic_data = synthetic_data[table_name]

            if not numeric_features:
                numeric_columns = get_numeric_discrete_columns(current_real_data)[0]
            else:
                numeric_columns = numeric_features[table_name]

            if not discrete_features:
                discrete_columns = get_numeric_discrete_columns(current_real_data)[1]
            else:
                discrete_columns = discrete_features[table_name]
                    
            correlation_fig = make_subplots(
                rows=1,
                cols=3,
                print_grid=False,
                shared_yaxes=True,
                subplot_titles=("Synthetic Data Correlation", "Real Data Correlation", "Absolute Diff (Δ) of Correlations"))
            syn_corr = get_correlation_matrix(df=current_synthetic_data, discrete_columns=discrete_columns)
            syn_mask = np.triu(np.ones_like(syn_corr, dtype=bool))
            chart = go.Heatmap(z=syn_corr.mask(syn_mask), x=syn_corr.columns.values, y=syn_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
            correlation_fig.add_trace(chart, 1, 1)

            real_corr = get_correlation_matrix(df=current_real_data, discrete_columns = discrete_columns)
            real_mask = np.triu(np.ones_like(real_corr, dtype=bool))
            chart = go.Heatmap(z=real_corr.mask(real_mask), x=real_corr.columns.values, y=real_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
            correlation_fig.add_trace(chart, 1, 2)

            diff_corr = np.abs(real_corr-syn_corr)
            diff_mask = np.triu(np.ones_like(diff_corr, dtype=bool))
            chart = go.Heatmap(z=diff_corr.mask(diff_mask), x=diff_corr.columns.values, y=diff_corr.columns.values, hoverongaps=False, colorscale=COLORSCALE, zmin=-2, zmax=2)
            correlation_fig.add_trace(chart, 1, 3)
            correlation_fig.update_yaxes(autorange='reversed')
            correlation_fig.update_layout(title=f'<b>Correlation Analysis for "{table_name}" table</b>', title_x=0.5)
            correlation_fig.show()
        
        # PCA Plots
        pca_plot = make_subplots(
            rows=ceil(len(table_names)/2),
            cols=2,
            subplot_titles=tuple(table_names)
        )

        for i, table_name in enumerate(table_names, start=1):
            current_real_data = real_data[table_name]
            current_synthetic_data = synthetic_data[table_name]

            # Removing datetime column
            current_real_data = current_real_data.select_dtypes(exclude=['datetime64'])
            current_synthetic_data = current_synthetic_data.select_dtypes(exclude=['datetime64'])

            real_pca = compute_pca(current_real_data)
            synthetic_pca = compute_pca(current_synthetic_data)

            real_pca_fig = go.Scattergl(
                x = real_pca['pc1'],
                y = real_pca['pc2'],
                mode='markers',
                name='Real Data',
                opacity=0.6,
                marker_color='#e04e14',
                showlegend=True if i==1 else False
            )
            synthetic_pca_fig = go.Scattergl(
                x = synthetic_pca['pc1'],
                y = synthetic_pca['pc2'],
                mode='markers',
                name='Synthetic Data',
                opacity=0.6,
                marker_color='#03b1fc',
                showlegend=True if i==1 else False
            )
            if i%2==0:
                pca_plot.add_trace(real_pca_fig, ceil(i/2), 2)
                pca_plot.add_trace(synthetic_pca_fig, ceil(i/2), 2)                    
            else:
                pca_plot.add_trace(real_pca_fig, ceil(i/2), 1)
                pca_plot.add_trace(synthetic_pca_fig, ceil(i/2), 1)

        pca_plot.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=False, title='Component 1')
        pca_plot.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=False, title='Component 2')
        pca_plot.update_layout(title=f'<b>Prinicpal Component Analysis</b>', height=300*len(table_names), title_x=0.5)
        pca_plot.show()

        # Numeric Desnity
        for table_name in table_names:
            current_real_data = real_data[table_name]
            current_synthetic_data = synthetic_data[table_name]

            if not numeric_features:
                numeric_columns = get_numeric_discrete_columns(current_real_data)[0]
            else:
                numeric_columns = numeric_features[table_name]
            
            if len(numeric_columns)==0:
                break
            
            numeric_subplot_titles = []
            for i, numeric_feat in enumerate(numeric_columns):
                numeric_subplot_titles.append(f'{numeric_feat}')

            numeric_subplot = make_subplots(
                rows=len(numeric_columns),
                cols=1,
                subplot_titles=tuple(numeric_subplot_titles),
                vertical_spacing=(1/(len(numeric_columns)-1)) if len(numeric_columns)>1 else 0
            )

            for i, numeric_feat in enumerate(numeric_columns, start=1):
                density_fig = ff.create_distplot(
                    [current_synthetic_data[numeric_feat], current_real_data[numeric_feat]],
                    group_labels=['Synthetic Data', 'Real Data'],
                    show_hist=False,
                    show_rug=False)
                numeric_subplot.add_trace(density_fig.data[0], i, 1)
                numeric_subplot.add_trace(density_fig.data[1], i, 1)
            numeric_subplot.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, title='Value')
            numeric_subplot.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False, title='Density', showticksuffix='last')
            numeric_subplot.update_layout(title=f'<b>Numerical Density Distribution for "{table_name}" table</b>', title_x=0.5)
            numeric_subplot.show()

        # Categorical Count Plots    
        for table_name in table_names:
            current_real_data = real_data[table_name]
            current_synthetic_data = synthetic_data[table_name]

            if not discrete_features:
                discrete_columns = get_numeric_discrete_columns(current_real_data)[1]
            else:
                discrete_columns = discrete_features[table_name]

            if len(discrete_columns)==0:
                break

            category_subplot_titles = []
            for i, categ_feat in enumerate(discrete_columns):
                category_subplot_titles.append(f'{categ_feat}')
            category_feat_plot = make_subplots(
                rows=ceil(len(discrete_columns)/2),
                cols=2,
                subplot_titles=tuple(category_subplot_titles),
                specs=[[{}, {}] for x in range(ceil(len(discrete_columns)/2))],
                vertical_spacing=(1/(ceil(len(discrete_columns)/2-1))) if len(discrete_columns)>2 else 0
            )

            for i, categ_feat in enumerate(discrete_columns, start=1):
                real = go.Histogram(
                    x=current_real_data[categ_feat],
                    opacity=0.75,
                    histnorm ='percent',
                    name='Real Data',
                    marker_color='#e04e14',
                    legendgroup='Real Data',
                    showlegend=True if i==1 else False,
                    hovertemplate='%{x} - %{y:.1f}%'
                )
                synthetic = go.Histogram(
                    x=current_synthetic_data[categ_feat],
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
                title=f'<b>Categorical Proportion Distribution for "{table_name}" table</b>',
                title_x=0.5
            )
            category_feat_plot.show()