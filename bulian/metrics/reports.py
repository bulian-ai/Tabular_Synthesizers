from ast import excepthandler
import enum
from re import L
from unicodedata import category
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import get_correlation_matrix, gauge, gauge_multi
import numpy as np
from ..metrics import compute_metrics
from ..metrics.single_table import *
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

METRIC_INFO = {
    'BNLogLikelihood': 'Average log likelihood across all the rows in the synthetic dataset.',
    'LogisticDetection': 'Detection Metric based on a LogisticRegression from scikit-learn.',
    'SVCDetection': 'Detection Metric based on a SVC from scikit-learn.',
    'BinaryDecisionTreeClassifier': 'ML Efficacy Metric for binary classifications problems, based on a DecisionTreeClassifier from scikit-learn.',
    'BinaryAdaBoostClassifier': 'ML Efficacy Metric for binary classifications problems, based on an AdaBoostClassifier from scikit-learn.',
    'BinaryLogisticRegression': 'ML Efficacy Metric for binary classifications problems, based on a LogisticRegression from scikit-learn.',
    'BinaryMLPClassifier': 'ML Efficacy Metric for binary classifications problems, based on an MLPClassifier from scikit-learn.',
    'MulticlassDecisionTreeClassifier': 'ML Efficacy Metric for multiclass classifications problems, based on a DecisionTreeClassifier from scikit-learn.',
    'MulticlassMLPClassifier': 'ML Efficacy Metric for multiclass classifications problems, based on an MLPClassifier from scikit-learn.',
    'LinearRegression': 'ML Efficacy Metric for regression problems, based on a LinearRegression from scikit-learn.',
    'MLPRegressor': 'ML Efficacy Metric for regression problems, based on an MLPRegressor from scikit-learn.',
    'GMLogLikelihood': 'Average log likelihood of multiple GMMs fit over real data and scored synthetic data.',
    'CSTest': 'Chi-Squared test to compare the distributions of two categorical columns.',
    'KSTest': 'Kolmogorov-Smirnov test to compare the distributions of two numerical columns',
    'KSTestExtended': 'KSTest on all the RDT transformed numerical variables.',
    'CategoricalCAP': 'Privacy Metric for categorical columns, based on the Correct Attribution Probability method.',
    'CategoricalZeroCAP': 'Privacy Metric for categorical columns, based on the Correct Attribution Probability method.',
    'CategoricalGeneralizedCAP': 'Privacy Metric for categorical columns, based on the Correct Attribution Probability method.',
    'CategoricalNB': 'Privacy Metric for categorical columns, based on CategoricalNB from scikit-learn.',
    'CategoricalKNN': 'Privacy Metric for categorical columns, based on KNeighborsClassifier from scikit-learn.',
    'CategoricalRF': 'Privacy Metric for categorical columns, based on RandomForestClassifier from scikit-learn.',
    'CategoricalSVM': 'Privacy Metric for categorical columns, based on SVMClassifier from scikit-learn.',
    'CategoricalEnsemble': 'Privacy Metric for categorical columns, based on an ensemble of categorical Privacy Metrics.',
    'NumericalLR': 'Privacy Metric for numerical columns, based on LinearRegression from scikit-learn.',
    'NumericalMLP': 'Privacy Metric for numerical columns, based on MLPRegressor from scikit-learn.',
    'NumericalSVR': 'Privacy Metric for numerical columns, based on SVR from scikit-learn.',
    'NumericalRadiusNearestNeighbor': 'Privacy Metric for numerical columns, based on an implementation of the Radius Nearest Neighbor method.',
    'ContinuousKLDivergence': 'KL-Divergence Metric applied to all possible pairs of numerical columns.',
    'DiscreteKLDivergence': 'KL-Divergence Metric applied to all the possible pairs of categorical and boolean columns.',
    'MLEfficacy': 'Generic ML Efficacy metric that detects the type of ML Problem associated with the dataset'
}

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

def get_column_name(name):
    name = name.replace('_', ' ')
    if name == 'MetricType':
        return 'Metric Type'
    return name

def get_metric_info(metric_name):
    if metric_name in METRIC_INFO:
        return str(METRIC_INFO[metric_name])
    return None

# To Do: We are not pating attention to whether the goal is to maximize or minimize; we assume it all maximize
def get_full_report(real_data, synthetic_data, discrete_columns, 
    numeric_columns, target=None, key_fields=None, sensitive_fields=None, show_dashboard=False):
    _OVERALL_SCORE_GRPS = ['Real vs Synthetic Dectection Metric','Statistical Test Metric','Distribution Similarity Metric']

    import warnings
    warnings.filterwarnings('ignore')
    
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
        oPriv = compute_metrics(privacy_metrics,real_data, synthetic_data,key_fields=key_fields,sensitive_fields=sensitive_fields)
        if len(oPriv)>0:
            o = pd.concat([o,oPriv],0)
            o = o.reset_index(drop=True)

    o = o[~np.isnan(o['normalized_score'])]
    o_overall = o[o['MetricType'].isin(_OVERALL_SCORE_GRPS)]
    # o_min_0 = o[o['min_value']==0.0]    ### what about case where Max value is not 1 but inf, for now we assume we have covered it
    # o_min_neginf = o[o['min_value']==-np.inf]
    multi_metrics = o.groupby('MetricType')['normalized_score'].mean().to_dict()
    # multi_metrics_min_0 = o_min_0.groupby('MetricType')['normalized_score'].mean().to_dict()
    # multi_metrics_min_neginf = o_min_neginf.groupby('MetricType')['normalized_score'].mean().to_dict()
    
    # efficiency_min_0 = 0
    # efficiency_min_neginf = 0 

    # if len(o_min_0)>0:
    #     efficiency_min_0 = (o_min_0.groupby('MetricType')['normalized_score'].mean()>0.5).sum()/(o_min_0['MetricType'].nunique())
    # if len(o_min_neginf)>0:
    #     efficiency_min_neginf = (o_min_neginf.groupby('MetricType')['normalized_score'].mean()>0.0).sum()/(o_min_neginf['MetricType'].nunique())
    
    # if (len(o_min_0)>0) & (len(o_min_neginf)>0):
    #     avg_efficiency = round(100*np.mean((efficiency_min_0,efficiency_min_neginf)))   ### Min: 0, max: 1
    # elif (len(o_min_0)>0):
    #     avg_efficiency = round(100*efficiency_min_0)
    # elif (len(o_min_neginf)>0):
    #     avg_efficiency = round(100*efficiency_min_neginf)
    # else:
    #     raise ValueError("Relevant metrics are NaN")


    try:
        avg_efficiency = 100*(o_overall['normalized_score'].mean())
        # print('avg_efficiency',avg_efficiency)
    except:
         ValueError("Some of the Relevant metrics are NaN")
    if show_dashboard:
        gauge_fig, gauge_value = gauge(avg_efficiency, show_dashboard)
        if len(o)>0:
            gauge_multi_fig, gauge_multi_values = gauge_multi(multi_metrics, show_dashboard)
        else:
            gauge_multi_fig = None
    else:
        gauge(avg_efficiency)
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
        chart = go.Heatmap(z=syn_corr, x=syn_corr.columns, y=syn_corr.columns, hoverongaps=False, colorscale=px.colors.diverging.RdYlGn, zmin=-1, zmax=1)
        correlation_fig.add_trace(chart, 1, 1)

        real_corr = get_correlation_matrix(df=real_data, discrete_columns = discrete_columns)
        real_mask = np.zeros_like(real_corr, dtype=np.bool)
        real_mask[np.triu_indices_from(real_mask)] = True
        real_corr[real_mask] = np.nan
        chart = go.Heatmap(z=real_corr, x=real_corr.columns, y=real_corr.columns, hoverongaps=False, colorscale=px.colors.diverging.RdYlGn, zmin=-1, zmax=1)
        correlation_fig.add_trace(chart, 1, 2)

        diff_corr = np.abs(real_corr)-np.abs(syn_corr)
        diff_mask = np.zeros_like(diff_corr, dtype=np.bool)
        diff_mask[np.triu_indices_from(diff_mask)] = True
        diff_corr[diff_mask] = np.nan
        chart = go.Heatmap(z=diff_corr, x=diff_corr.columns, y=diff_corr.columns, hoverongaps=False, colorscale=px.colors.diverging.RdYlGn, zmin=-1, zmax=1)
        correlation_fig.add_trace(chart, 1, 3)

        correlation_fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['sub-background'],
            font_color=colors['text'],
            coloraxis_colorbar_x=-2
        )

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
                name='Real Data',
                marker_color='#e04e14',
                legendgroup='Real Data',
                showlegend=True if i==1 else False
            )
            synthetic = go.Histogram(
                x=synthetic_data[categ_feat],
                opacity=0.75, 
                name='Synthetic Data', 
                marker_color='#03b1fc', 
                legendgroup='Sythentic Data',
                showlegend=True if i==1 else False
            )           
            data = [real, synthetic]

            if i%2==0:
                category_feat_plot.add_trace(data[0], ceil(i/2), 2)
                category_feat_plot.add_trace(data[1], ceil(i/2), 2)                    
            else:
                category_feat_plot.add_trace(data[0], ceil(i/2), 1)
                category_feat_plot.add_trace(data[1], ceil(i/2), 1)

        category_feat_plot.update_xaxes(showline=True, linewidth=1, linecolor='black')
        category_feat_plot.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
        category_feat_plot.update_layout(
            plot_bgcolor=colors['sub-background'],
            paper_bgcolor=colors['sub-background'],
            font_color=colors['text'],
            height=1500
        )
        graph_objects.append(html.Hr())
        graph_objects.append(html.H1('Categorical Count Distribution'))
        graph_objects.append(
            dcc.Graph(
                id=f'category-feat',
                figure=category_feat_plot,
                style={'width':'75%','margin-left':'auto', 'margin-right':'auto', 'height':'100%'}
            )
        )

        metrics_df = o[['metric', 'name', 'normalized_score', 'MetricType']]
        metrics_df = metrics_df.round(2)

        tables = [html.H1("Detailed Metrics View")]
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
                        pdf.setPage(1);
                        pdf.addImage(image_data, 'PNG', 39, 38.7);
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
                )]
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
            html.Hr(),
            html.H1("Metric Cards"),
            html.Div(
                children=metric_info_cards_rows,
                id='metric_info_div'
            ),
            html.A(
                id='docs_btn',
                href="https://docs.bulian.ai/bulianai-overview/api-docs/getting-started-with-bulian-ai/metrics/single-table-metrics",
                target="_blank",
                children=[
                    dbc.Button("Learn More", color="warning", className="me-1"),
                ],
            ),
        ]

        if gauge_multi_fig:
            gauge_multi_figures = []
            row = []
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
        else:
            gauge_multi_figures = []
        
        table_header = [
            html.Thead(html.Tr([html.Th(""), html.Th("Real Data"), html.Th("Synthetic Data")]))
        ]

        row1 = html.Tr([html.Td("Row Count"), html.Td(len(real_data)), html.Td(len(synthetic_data))])
        row2 = html.Tr([html.Td("Column Count"), html.Td(len(real_data.columns)), html.Td(len(synthetic_data.columns))])

        table_body = [html.Tbody([row1, row2,])]

        date_time = 'Generated on \n'+ datetime.utcnow().strftime("%d/%m/%Y, %I:%M %p")
        app.layout = html.Div(
            style={
                'backgroundColor':colors['background'],
            },
            children=[
                dbc.Button("Download PDF", color="warning", className="me-1", id='js', n_clicks=0),
                html.A(
                    id='learn_more_btn',
                    href="#cards",
                    children=[
                        dbc.Button("Learn More", color="warning", className="me-1"),
                    ],
                ),
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
                        dbc.Table(table_header + table_body, bordered=True, id='data_summary_table', size='sm'),
                        html.Div(id="divider"),
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
                        dbc.Tooltip(
                            "The overall score represents the utility score or confidence score for synthetic datasets.",
                            target="gauge",
                        ),
                        dbc.Tooltip(
                            "Distribution similarity between real and synthetic data; based on Kullback–Leibler divergence measures.",
                            target="gauge-Similarity Score",
                        ),
                        dbc.Tooltip(
                            "Numerical and categorical attack models and their ability to predict real data's sensitive attributes.",
                            target="gauge-Privacy Score",
                        ),
                        dbc.Tooltip(
                            "Quantifies ability of ML algorithms to separate real vs synthetic data. Indicates deep structural stability.",
                            target="gauge-Detection Score",
                        ),
                        dbc.Tooltip(
                            "Statistics based measure to quantify statistical distribution similarity. Based on K-S and C-S tests.",
                            target="gauge-Statistical Score",
                        ),
                        dbc.Tooltip(
                            "Generic ML Efficacy metric that detects the type of ML Problem associated with the dataset by analyzing the target column type and then applies all the metrics that are compatible with it.",
                            target="gauge-ML Efficacy Score",
                        ),
                    ]+graph_objects+tables+metric_info_div
                )
            ])
        app.run_server(debug=True)
    else:
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

        #print("\n------------------------------------ CATEGORICAL FEATURE DISTRIBUTIONS ------------------------------------\n")
        for categ_feat in discrete_columns:
            plt.figure(figsize=(20,4))
            plt.hist(real_data[categ_feat], 
                    label='Real Data',alpha=0.2,density=True)

            plt.hist(synthetic_data[categ_feat], 
                    label='Synthetic Data',alpha=0.2,density=True)
            plt.ylabel('Mass Distribution', fontsize=10)
            plt.legend(loc='upper right',)
            plt.title(f'Categorical Count Distribution : {categ_feat}',fontsize=16,y=1.02)
            plt.tick_params(axis='x', rotation=90)
            plt.show()

# if __name__ == '__main__':
#     get_gauge(20)   