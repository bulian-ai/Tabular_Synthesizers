import streamlit as st
import pandas as pd
import warnings
from utils import calculate_metrics

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# Main App
st.set_page_config(
    page_title='Bulian AI',
    initial_sidebar_state='expanded',
    layout='wide'
)

st.title("Bulian AI Demo")

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
        options=real_data.columns
    )


if select_box_option == 'Private Twin Synthesizer':
    from bulian.Tabular.synthesizers import PrivateTwinSynthesizer as synth_model
if select_box_option == 'Twin Synthesizer':
    from bulian.Tabular.synthesizers import TwinSynthesizer as synth_model


sample_count = st.sidebar.number_input(
    'No. of Samples',
    min_value=10
)

epochs = st.sidebar.number_input(
    'Epochs',
    value=10,
)

batch_size = st.sidebar.number_input(
    'Batch Size',
    value=200
)

device = st.sidebar.selectbox('Select Device', ('cpu', 'cuda'))

if uploaded_file:
    define_target = st.sidebar.checkbox('Define Target', value=False)
    if define_target:
        target = st.sidebar.selectbox('Target Column', real_data.columns)

if not uploaded_file:
    run_model_button = st.sidebar.button('Run Model', disabled=True)
else:
    run_model_button = st.sidebar.button('Run Model', disabled=False)

if run_model_button:
    with st.spinner('Working...'):
        if not uploaded_file:
            st.error("File not uploaded")
        else:
            model = synth_model(batch_size=batch_size, device=device)
            model.fit(data=real_data, discrete_columns=discrete_columns, epochs=epochs)
            synthetic_data = model.sample(sample_count)
            st.session_state['synthetic_data'] = synthetic_data
            st.success("Synthetic data generated")

            if 'synthetic_data' in st.session_state:
                expander = st.expander('See Synthetic Data')
                expander.download_button(
                    'Download as CSV',
                    data=st.session_state['synthetic_data'].to_csv().encode('utf-8'),
                    file_name='synthetic_data.csv',
                    mime='text/csv'
                )
                expander.dataframe(st.session_state['synthetic_data'].head())

            overall_metrics, gauge_fig, gauge_multi_fig = calculate_metrics(
                real_data=st.session_state['real_data'],
                synthetic_data=st.session_state['synthetic_data'],
                target=target if define_target else None
            )
            st.plotly_chart(figure_or_data=gauge_fig, use_container_width=True)
            st.plotly_chart(figure_or_data=gauge_multi_fig, use_container_width=True)


            metrics_df = overall_metrics[['metric', 'normalized_score', 'MetricType']]
            metrics_df = metrics_df.round(2)
            st.dataframe(data=metrics_df)
