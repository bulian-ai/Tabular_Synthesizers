import streamlit as st
import pandas as pd
import warnings

from streamlit_elements import elements, mui, html
from app_utils import build_correlation_plot, build_distribution_plots, build_gauge_plots, build_pca_plot
from app_components import footer

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Main App
st.set_page_config(
    page_title='Bulian AI',
    layout='wide'
)

st.image(image='assets/logo.png', width=400)
st.subheader('Safe, artificial data that acts like your production data')
st.markdown('Generate safe, realistic, and scalable synthetic data on demand. Safely share it across teams, businesses, and borders.')

with st.expander('Dive into Bulian AI\'s synthetic data models'):
    col1, col2, col3 = st.columns(3)
    with col1:
        with elements('card_frame'):
            mui.Card(
                mui.Typography(mui.icon.ModelTraining(), ' Twin Synthesizer', variant="h5", sx={'padding':'12px', 'font-weight':'bold', 'text-align':'center'}),
                mui.Typography(
                    'GAN models inherits from BaseSynthesizer class; generate non-privacy preserving synthetic datasets given an input python pandas.DataFrame and column list broken by numeric and categorical columns passes as python list.',
                    sx={
                        'text-algin':'justify',
                    }
                ),
                sx={
                    "padding":"12px",
                    "height":"190px",
                },
                variant="outlined"
            )
    with col2:
        with elements('card_frame_2'):
            mui.Card(
                mui.Typography(mui.icon.Security(), ' Private Twin Synthesizer', variant="h5", sx={'padding':'12px', 'font-weight':'bold', 'text-align':'center'}),
                mui.Typography(
                    'Differentially private GAN models inherits from BaseSynthesizerPrivate class and generate privacy preserving synthetic datasets given an input python pandas.DataFrame and column list broken by numeric and categorical columns passes as python list.',
                    sx={
                        'text-algin':'justify',
                    }
                ),
                sx={
                    "padding":"12px",
                    "height":"190px",
                },
                variant="outlined"              
            )
    with col3:
        with elements('card_frame_3'):
            mui.Card(
                mui.Typography(mui.icon.MultipleStop(), ' Relational Model (HMA)', variant="h5", sx={'padding':'12px', 'font-weight':'bold', 'text-align':'center'}),
                mui.Typography(
                    'Gaussian Copula based HMA1 models inherits from BaseRelationalModel class and generates multi-table complex synthetic datasets given an input metadata python dictand relational tables as dict.',
                    sx={
                        'text-algin':'justify',
                    }
                ),
                sx={
                    "padding":"12px",
                    "height":"190px",
                },
                variant="outlined"          
            )
    with elements('info'):
        mui.Alert('Currently only Twin and Private Twin Synthesizers have been implemented on the app', severity="info")

with st.expander('Examples'):
    col1, col2 = st.columns([3,1])
    with col1:
        with elements("notebook-links"):
            mui.List(
                mui.ListItem(
                    mui.ListItemIcon(
                        mui.icon.DoubleArrow,
                        sx={
                            'color':'#fc3',
                        }
                    ),
                    mui.ListItemText(
                        html.a(
                            'single-table-trainer-notebook-CPU',
                            css={
                                "color":"#fafafa",
                                "text-decoration":"none",
                                "&:hover": {
                                    "color": "#fc3"
                                }
                            },
                            target='_blank',
                            href='https://github.com/bulian-ai/public_docs/blob/main/notebooks/single_table_demo_CPU.ipynb'),
                    )
                ),
                mui.ListItem(
                    mui.ListItemIcon(
                        mui.icon.DoubleArrow,
                        sx={
                            'color':'#fc3'
                        }
                    ),
                    mui.ListItemText(
                        html.a(
                            'single-table-trainer-notebook-GPU',
                            css={
                                "color":"#fafafa",
                                "text-decoration":"none",
                                "&:hover": {
                                    "color": "#fc3"
                                }
                            },
                            target='_blank',
                            href='https://github.com/bulian-ai/public_docs/blob/main/notebooks/single_table_demo_GPU.ipynb'),
                    )
                ),
                mui.ListItem(
                    mui.ListItemIcon(
                        mui.icon.DoubleArrow,
                        sx={
                            'color':'#fc3'
                        }
                    ),
                    mui.ListItemText(
                        html.a(
                            'Finance churn class imbalance trainer notebook ',
                            css={
                                "color":"#fafafa",
                                "text-decoration":"none",
                                "&:hover": {
                                    "color": "#fc3"
                                }
                            },
                            target='_blank',
                            href='https://github.com/bulian-ai/public_docs/blob/main/notebooks/Boost_financial_churn_models.ipynb'),
                    )
                ),
                mui.ListItem(
                    mui.ListItemIcon(
                        mui.icon.DoubleArrow,
                        sx={
                            'color':'#fc3'
                        }
                    ),
                    mui.ListItemText(
                        html.a(
                            'Relational HMA1 Trainer',
                            css={
                                "color":"#fafafa",
                                "text-decoration":"none",
                                "&:hover": {
                                    "color": "#fc3"
                                }
                            },
                            target='_blank',
                            href='https://github.com/bulian-ai/public_docs/blob/main/notebooks/relational_demo.ipynb'),
                    )
                ),
                mui.ListItem(
                    mui.ListItemIcon(
                        mui.icon.DoubleArrow,
                        sx={
                            'color':'#fc3'
                        }
                    ),
                    mui.ListItemText(
                        html.a(
                            'Metadata Tutorial',
                            css={
                                "color":"#fafafa",
                                "text-decoration":"none",
                                "&:hover": {
                                    "color": "#fc3"
                                }
                            },
                            target='_blank',
                            href='https://github.com/bulian-ai/public_docs/blob/main/notebooks/Relational%20Metadata.ipynb'),
                    )
                )
        )
    with col2:
        with elements('colab-links'):
            mui.List(
                mui.ListItem(
                    mui.ListItemText(
                        'View in ',
                        html.a(
                            'Colab',
                            css={
                                "color":"#fc3",
                                "text-decoration":"none",
                            },
                            target='_blank',
                            href='https://colab.research.google.com/github/bulian-ai/public_docs/blob/main/notebooks/single_table_demo_CPU.ipynb'),
                    )
                ),
                mui.ListItem(
                    mui.ListItemText(
                        'View in ',
                        html.a(
                            'Colab',
                            css={
                                "color":"#fc3",
                                "text-decoration":"none",
                            },
                            target='_blank',
                            href='https://colab.research.google.com/github/bulian-ai/public_docs/blob/main/notebooks/single_table_demo_CPU.ipynb'),
                    )
                ),
                mui.ListItem(
                    mui.ListItemText(
                        'View in ',
                        html.a(
                            'Colab',
                            css={
                                "color":"#fc3",
                                "text-decoration":"none",
                            },
                            target='_blank',
                            href='https://colab.research.google.com/github/bulian-ai/public_docs/blob/main/notebooks/single_table_demo_CPU.ipynb'),
                    )
                ),
                mui.ListItem(
                    mui.ListItemText(
                        'View in ',
                        html.a(
                            'Colab',
                            css={
                                "color":"#fc3",
                                "text-decoration":"none",
                            },
                            target='_blank',
                            href='https://colab.research.google.com/github/bulian-ai/public_docs/blob/main/notebooks/single_table_demo_CPU.ipynb'),
                    )
                ),
                mui.ListItem(
                    mui.ListItemText(
                        'View in ',
                        html.a(
                            'Colab',
                            css={
                                "color":"#fc3",
                                "text-decoration":"none",
                            },
                            target='_blank',
                            href='https://colab.research.google.com/github/bulian-ai/public_docs/blob/main/notebooks/single_table_demo_CPU.ipynb'),
                    )
                ),
            )

with elements("docs_button"):
    mui.Button(
        mui.icon.EmojiPeople,
        mui.icon.DoubleArrow,
        html.a(
            "Learn more about Bulian AI on our Docs",
            href='https://docs.bulian.ai/bulianai-overview/',
            target='_blank',
            css={
                'text-decoration':'none',
                'color':'#fafafa',
            }),
        sx={
            'background': 'linear-gradient(285.02deg,#3347ff -48.87%,#f33 77.92%)',
            'color':'#FAFAFA'
        }
    )

st.markdown('<hr>', unsafe_allow_html=True)

with elements('app_heading'):
    mui.Typography('Discover the Product', variant='h4', sx={
        'color': '#fc3',
        'font-weight': 'bold',
        'margin-bottom':'20px',
        'text-align':'center'
    })
st.markdown('<img src="https://i.imgur.com/Vv8KTDZ.png" class="flow_chart"/>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file is not None:
    real_data = pd.read_csv(uploaded_file)
    st.session_state['real_data'] = real_data

if uploaded_file:
    with st.expander('See Real Data'):
        st.dataframe(real_data.head())

# Using object notation
select_box_option = st.sidebar.selectbox(
    "Select Model",
    ("Twin Synthesizer", "Private Twin Synthesizer")
)

discrete_columns = []
if uploaded_file:
    discrete_columns = st.sidebar.multiselect(
        'Select Discrete Columns',
        options=real_data.columns,
        help='List of discrete columns to be used to generate the Conditional Vector. This list should contatin the column names.'
    )


if select_box_option == 'Private Twin Synthesizer':
    from bulian.Tabular.synthesizers import PrivateTwinSynthesizer as synth_model
if select_box_option == 'Twin Synthesizer':
    from bulian.Tabular.synthesizers import TwinSynthesizer as synth_model


sample_count = st.sidebar.number_input(
    'No. of Samples',
    min_value=10,
    help='No. of synthetic data rows to be generated.'
)

if select_box_option == 'Private Twin Synthesizer':
    e_factor = st.sidebar.number_input(
        'Epsilon',
        value=10,
        help='We perform generator iterations until our privacy constraint, has been reached. Generally, high-utilizty datasets need higher privacy budgets. Defaults to `1`.'
    )
else:
    e_factor = st.sidebar.number_input(
        'Epochs',
        value=10,
        help='Number of training epochs.'
    )

batch_size = st.sidebar.number_input(
    'Batch Size',
    value=200,
    help='Modify the batch size used for model training.'
)

device = st.sidebar.selectbox('Select Device', ('cpu', 'cuda'), help='Select cuda if GPU is available otherwise cpu')
device = 'cpu' # Streamlit cloud does not support GPU

if uploaded_file:
    define_target = st.sidebar.checkbox('Specify target column?', value=False, help='Name of the column to use as the target.')
    if define_target:
        target = st.sidebar.selectbox('Target Column', real_data.columns)

if not uploaded_file:
    run_model_button = st.sidebar.button('Run Model', disabled=True)
else:
    run_model_button = st.sidebar.button('Run Model', disabled=False)

if run_model_button:
    with st.spinner('Generating synthetic data...'):
        if not uploaded_file:
            st.error("File not uploaded")
        else:
            model = synth_model(batch_size=batch_size, device=device)
            if select_box_option=='Private Twin Synthesizer':
                model.fit(data=real_data, discrete_columns=discrete_columns, update_epsilon=e_factor)
            else:
                model.fit(data=real_data, discrete_columns=discrete_columns, epochs=e_factor)
            synthetic_data = model.sample(sample_count)
            st.session_state['synthetic_data'] = synthetic_data
            st.success("Synthetic data generated")
    with st.spinner('Generating data quality report...'):
        if 'synthetic_data' in st.session_state:
            expander = st.expander('See Synthetic Data')
            expander.download_button(
                'Download as CSV',
                data=st.session_state['synthetic_data'].to_csv().encode('utf-8'),
                file_name='synthetic_data.csv',
                mime='text/csv'
            )
            expander.dataframe(st.session_state['synthetic_data'].head())

        gauge_fig, gauge_multi_fig = build_gauge_plots(
            real_data=st.session_state['real_data'],
            synthetic_data=st.session_state['synthetic_data'],
            target=target if define_target else None
        )
        st.plotly_chart(figure_or_data=gauge_fig, use_container_width=True)
        st.plotly_chart(figure_or_data=gauge_multi_fig, use_container_width=True)
        
        correlation_fig = build_correlation_plot(
            real_data=st.session_state['real_data'],
            synthetic_data=st.session_state['synthetic_data'],
            discrete_columns=discrete_columns
        )
        st.plotly_chart(figure_or_data=correlation_fig, use_container_width=True)

        pca_fig = build_pca_plot(
            real_data=st.session_state['real_data'],
            synthetic_data=st.session_state['synthetic_data']
        )
        st.plotly_chart(figure_or_data=pca_fig, use_container_width=True)

        numeric_density_figs, categorical_plot = build_distribution_plots(
            real_data=st.session_state['real_data'],
            synthetic_data=st.session_state['synthetic_data'],
            discrete_columns=discrete_columns
        )

        for plot in numeric_density_figs:
            st.plotly_chart(figure_or_data=plot, use_container_width=True)
        
        st.plotly_chart(figure_or_data=categorical_plot, use_container_width=True)

st.markdown(
    '''<style>
        .streamlit-expanderHeader{
            font-size:1.2em;
            font-weight:bold;
            color:#fc3;
        }
        .flow_chart{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 70%;
        }
        a {
        text-decoration: none !important;
        color: #fc3;
        }
    </style>''',
    unsafe_allow_html=True
)
st.markdown(body='<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css" integrity="sha512-xh6O/CkQoPOWDdYTDqeRdPCVd1SpvCA9XXcUnZS2FmJNp1coAFzvtCN9BmamE+4aHK8yyUHUSCcJHgXloTyT2A==" crossorigin="anonymous" referrerpolicy="no-referrer" />', unsafe_allow_html=True)
